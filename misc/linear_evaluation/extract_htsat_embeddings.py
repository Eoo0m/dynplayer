"""
Phase 1: Extract HTS-AT embeddings for all tracks and save to disk.
This is a one-time operation - embeddings can then be reused for classifier training.
"""

import sys
import os

# Add HTS-AT paths
HTSAT_DIR = os.path.join(os.path.dirname(__file__), "..", "HTS-Audio-Transformer")
sys.path.insert(0, HTSAT_DIR)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import csv
import librosa
from tqdm import tqdm

import config as htsat_config
from model.htsat import HTSAT_Swin_Transformer


class AudioDataset(Dataset):
    """Dataset for extracting embeddings from all audio files."""

    def __init__(self, track_ids, audio_dir, sample_rate=32000, chunk_duration=10, max_chunks=3):
        self.track_ids = track_ids
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_length = sample_rate * chunk_duration  # 10초 = 320000 샘플
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        audio_path = self.audio_dir / f"{track_id}.mp3"

        try:
            waveform, _ = librosa.load(
                audio_path, sr=self.sample_rate, mono=True,
                duration=self.chunk_duration * self.max_chunks
            )

            chunks = []
            chunk_lengths = []
            for i in range(self.max_chunks):
                start = i * self.chunk_length
                end = start + self.chunk_length
                if start < len(waveform):
                    chunk = waveform[start:end]
                    chunk_len = len(chunk)
                    if len(chunk) < self.chunk_length:
                        chunk = np.pad(chunk, (0, self.chunk_length - len(chunk)))
                    chunks.append(chunk)
                    chunk_lengths.append(chunk_len)

            if len(chunks) == 0:
                chunks = [np.zeros(self.chunk_length, dtype=np.float32)]
                chunk_lengths = [self.sample_rate]

            num_chunks = len(chunks)

        except Exception as e:
            chunks = [np.zeros(self.chunk_length, dtype=np.float32)]
            chunk_lengths = [self.sample_rate]
            num_chunks = 1

        return {
            "track_id": track_id,
            "chunks": torch.FloatTensor(np.stack(chunks)),
            "chunk_lengths": torch.LongTensor(chunk_lengths),
            "num_chunks": num_chunks,
        }


def collate_fn(batch):
    max_num_chunks = max(item['num_chunks'] for item in batch)
    chunk_length = batch[0]['chunks'].shape[1]

    all_chunks = []
    all_chunk_lengths = []
    all_num_chunks = []
    track_ids = []

    for item in batch:
        chunks = item['chunks']
        chunk_lengths = item['chunk_lengths']
        num_chunks = item['num_chunks']

        if num_chunks < max_num_chunks:
            pad_chunks = torch.zeros(max_num_chunks - num_chunks, chunk_length)
            chunks = torch.cat([chunks, pad_chunks], dim=0)
            pad_lengths = torch.zeros(max_num_chunks - num_chunks, dtype=torch.long)
            chunk_lengths = torch.cat([chunk_lengths, pad_lengths])

        all_chunks.append(chunks)
        all_chunk_lengths.append(chunk_lengths)
        all_num_chunks.append(num_chunks)
        track_ids.append(item['track_id'])

    return {
        'track_ids': track_ids,
        'chunks': torch.stack(all_chunks),
        'chunk_lengths': torch.stack(all_chunk_lengths),
        'num_chunks': torch.tensor(all_num_chunks, dtype=torch.long),
    }


def load_htsat(checkpoint_path, device):
    model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config=htsat_config,
        depths=htsat_config.htsat_depth,
        embed_dim=htsat_config.htsat_dim,
        patch_stride=(htsat_config.htsat_stride[0], htsat_config.htsat_stride[1]),
        num_heads=htsat_config.htsat_num_head,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("sed_model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model


def register_feature_hook(model):
    feature_hook = {"features": None}

    def hook_fn(module, input, output):
        x = input[0]
        B, C, F, T = x.shape
        feature_hook["features"] = x.reshape(B, C, -1)

    model.tscam_conv.register_forward_hook(hook_fn)
    return feature_hook


def extract_chunk_embedding(model, audio_chunk, chunk_length, device, feature_hook):
    feature_hook["features"] = None
    _ = model(audio_chunk)

    x = feature_hook["features"]
    B, C, T = x.shape

    time_ratio = audio_chunk.shape[1] / T
    frame_lengths = (chunk_length.float() / time_ratio).long().clamp(min=1, max=T)

    mask = torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).expand_as(x).float()

    x_mean = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
    x_masked = x.masked_fill(mask == 0, -1e9)
    x_max = x_masked.max(dim=2)[0]

    chunk_emb = torch.cat([x_mean, x_max], dim=1)
    return chunk_emb


def extract_embeddings(model, chunks, chunk_lengths, num_chunks, device, feature_hook):
    B, max_num_chunks, chunk_length = chunks.shape

    all_chunk_embs = []
    for i in range(max_num_chunks):
        chunk_i = chunks[:, i, :].to(device)
        length_i = chunk_lengths[:, i].to(device)
        emb_i = extract_chunk_embedding(model, chunk_i, length_i, device, feature_hook)
        all_chunk_embs.append(emb_i)

    all_chunk_embs = torch.stack(all_chunk_embs, dim=1)

    chunk_mask = torch.arange(max_num_chunks, device=device).unsqueeze(0) < num_chunks.unsqueeze(1)
    chunk_mask = chunk_mask.unsqueeze(-1).float()

    embeddings = (all_chunk_embs * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp(min=1)
    return embeddings


def load_track_ids():
    """Load all track IDs from genre CSV."""
    csv_path = Path(__file__).parent.parent / "data" / "spotify_genre_info_top60.csv"
    audio_dir = Path("/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node")

    track_ids = []
    genres = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['genres']:
                track_id = row['track_id']
                if (audio_dir / f"{track_id}.mp3").exists():
                    track_ids.append(track_id)
                    genres.append(row['genres'])

    # Build genre mapping
    unique_genres = sorted(set(genres))
    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}

    # Create track_id to genre_idx mapping
    track_to_genre_idx = {tid: genre_to_idx[g] for tid, g in zip(track_ids, genres)}

    print(f"Found {len(track_ids):,} tracks with audio")
    print(f"Genres: {len(unique_genres)}")

    return track_ids, track_to_genre_idx, genre_to_idx, idx_to_genre


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 8  # Smaller batch for memory
    output_dir = Path(__file__).parent.parent / "data" / "audio_embeddings"

    # Clean output directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HTS-AT
    print("\n=== Loading HTS-AT ===")
    checkpoint_path = Path(HTSAT_DIR) / "HTSAT_AudioSet_Saved_6.ckpt"
    encoder = load_htsat(checkpoint_path, device)
    feature_hook = register_feature_hook(encoder)

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"HTS-AT parameters: {num_params:,}")

    # Load track IDs
    print("\n=== Loading Track IDs ===")
    audio_dir = Path("/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node")
    track_ids, track_to_genre_idx, genre_to_idx, idx_to_genre = load_track_ids()

    # Create dataset
    dataset = AudioDataset(track_ids, audio_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # Extract embeddings
    print("\n=== Extracting Embeddings ===")
    all_embeddings = []
    all_track_ids = []
    all_genre_idxs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            chunks = batch["chunks"]
            chunk_lengths = batch["chunk_lengths"]
            num_chunks = batch["num_chunks"].to(device)
            batch_track_ids = batch["track_ids"]

            embeddings = extract_embeddings(
                encoder, chunks, chunk_lengths, num_chunks, device, feature_hook
            )

            all_embeddings.append(embeddings.cpu().numpy())
            all_track_ids.extend(batch_track_ids)
            all_genre_idxs.extend([track_to_genre_idx[tid] for tid in batch_track_ids])

    # Concatenate
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_track_ids = np.array(all_track_ids)
    all_genre_idxs = np.array(all_genre_idxs)

    print(f"\nEmbeddings shape: {all_embeddings.shape}")

    # Save
    print("\n=== Saving ===")
    np.save(output_dir / "htsat_embeddings.npy", all_embeddings)
    np.save(output_dir / "track_ids.npy", all_track_ids)
    np.save(output_dir / "genre_idxs.npy", all_genre_idxs)

    # Save mappings
    import pickle
    with open(output_dir / "genre_to_idx.pkl", 'wb') as f:
        pickle.dump(genre_to_idx, f)
    with open(output_dir / "idx_to_genre.pkl", 'wb') as f:
        pickle.dump(idx_to_genre, f)

    print(f"Saved to {output_dir}")
    print(f"  - htsat_embeddings.npy: {all_embeddings.shape}")
    print(f"  - track_ids.npy: {len(all_track_ids)}")
    print(f"  - genre_idxs.npy: {len(all_genre_idxs)}")
    print(f"  - genre_to_idx.pkl: {len(genre_to_idx)} genres")

    print("\nDone! Now run train_genre_htsat_fast.py for classifier training.")


if __name__ == "__main__":
    main()
