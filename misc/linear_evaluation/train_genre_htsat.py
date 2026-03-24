"""
Genre classification using HTS-AT (Hierarchical Token-Semantic Audio Transformer)
Pretrained on AudioSet, 768-dim embeddings
"""

import sys
import os

# Add HTS-AT paths - must add parent first for relative imports to work
HTSAT_DIR = os.path.join(os.path.dirname(__file__), "..", "HTS-Audio-Transformer")
sys.path.insert(0, HTSAT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import librosa

# Import HTS-AT
import config as htsat_config
from model.htsat import HTSAT_Swin_Transformer


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AudioGenreDataset(Dataset):
    def __init__(
        self, track_ids, genres, genre_to_idx, audio_dir, sample_rate=32000, chunk_duration=10, max_chunks=3
    ):
        # HTS-AT는 10초 chunk만 지원 -> 최대 30초를 3개 chunk로 분할
        self.track_ids = track_ids
        self.genres = genres
        self.genre_to_idx = genre_to_idx
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_length = sample_rate * chunk_duration  # 10초 = 320000 샘플
        self.max_chunks = max_chunks  # 최대 3개 chunk (30초)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        genre = self.genres[idx]
        genre_idx = self.genre_to_idx[genre]

        audio_path = self.audio_dir / f"{track_id}.mp3"

        try:
            # 전체 오디오 로드 (최대 30초)
            waveform, _ = librosa.load(
                audio_path, sr=self.sample_rate, mono=True, duration=self.chunk_duration * self.max_chunks
            )

            # 10초 chunk들로 분할
            chunks = []
            chunk_lengths = []
            for i in range(self.max_chunks):
                start = i * self.chunk_length
                end = start + self.chunk_length
                if start < len(waveform):
                    chunk = waveform[start:end]
                    chunk_len = len(chunk)
                    # 10초 미만이면 패딩
                    if len(chunk) < self.chunk_length:
                        chunk = np.pad(chunk, (0, self.chunk_length - len(chunk)))
                    chunks.append(chunk)
                    chunk_lengths.append(chunk_len)

            # chunk가 없으면 (빈 오디오) 기본값
            if len(chunks) == 0:
                chunks = [np.zeros(self.chunk_length, dtype=np.float32)]
                chunk_lengths = [self.sample_rate]

            num_chunks = len(chunks)

        except:
            chunks = [np.zeros(self.chunk_length, dtype=np.float32)]
            chunk_lengths = [self.sample_rate]
            num_chunks = 1

        return {
            "chunks": torch.FloatTensor(np.stack(chunks)),  # [num_chunks, chunk_length]
            "chunk_lengths": torch.LongTensor(chunk_lengths),  # [num_chunks]
            "num_chunks": num_chunks,
            "genre_idx": genre_idx,
        }


def collate_fn(batch):
    """Custom collate function for chunked audio"""
    # 각 샘플의 chunk 수가 다를 수 있음 (1~3개)
    max_num_chunks = max(item['num_chunks'] for item in batch)
    chunk_length = batch[0]['chunks'].shape[1]  # 320000

    all_chunks = []
    all_chunk_lengths = []
    all_num_chunks = []
    genre_idxs = []

    for item in batch:
        chunks = item['chunks']  # [num_chunks, chunk_length]
        chunk_lengths = item['chunk_lengths']  # [num_chunks]
        num_chunks = item['num_chunks']

        # max_num_chunks에 맞춰 패딩
        if num_chunks < max_num_chunks:
            pad_chunks = torch.zeros(max_num_chunks - num_chunks, chunk_length)
            chunks = torch.cat([chunks, pad_chunks], dim=0)
            pad_lengths = torch.zeros(max_num_chunks - num_chunks, dtype=torch.long)
            chunk_lengths = torch.cat([chunk_lengths, pad_lengths])

        all_chunks.append(chunks)
        all_chunk_lengths.append(chunk_lengths)
        all_num_chunks.append(num_chunks)
        genre_idxs.append(item['genre_idx'])

    return {
        'chunks': torch.stack(all_chunks),  # [B, max_num_chunks, chunk_length]
        'chunk_lengths': torch.stack(all_chunk_lengths),  # [B, max_num_chunks]
        'num_chunks': torch.tensor(all_num_chunks, dtype=torch.long),  # [B]
        'genre_idx': torch.tensor(genre_idxs, dtype=torch.long)  # [B]
    }


class GenreClassifier(nn.Module):
    """Linear classifier on HTS-AT embeddings (mean+max pooling)"""

    def __init__(self, embedding_dim=768 * 2, num_genres=192):
        super().__init__()
        # embedding_dim = 768*2 for mean+max concatenation
        self.classifier = nn.Linear(embedding_dim, num_genres)

    def forward(self, x):
        return self.classifier(x)


def load_htsat(checkpoint_path, device):
    """Load pretrained HTS-AT"""
    # Create model with config
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'sed_model.' prefix if present
        state_dict = {k.replace("sed_model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    return model


def extract_chunk_embedding(model, audio_chunk, chunk_length, device, feature_hook):
    """Extract embedding from a single 10-second chunk with length masking"""
    feature_hook["features"] = None
    output = model(audio_chunk)

    # Get features before pooling from hook
    x = feature_hook["features"]  # (B, C, T)
    B, C, T = x.shape

    # Create mask based on chunk_length
    time_ratio = audio_chunk.shape[1] / T
    frame_lengths = (chunk_length.float() / time_ratio).long().clamp(min=1, max=T)

    # Create mask [B, T]
    mask = torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).expand_as(x).float()  # [B, C, T]

    # Masked mean pooling
    x_mean = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (B, C)

    # Masked max pooling
    x_masked = x.masked_fill(mask == 0, -1e9)
    x_max = x_masked.max(dim=2)[0]  # (B, C)

    chunk_emb = torch.cat([x_mean, x_max], dim=1)  # (B, C*2)
    return chunk_emb


def extract_embeddings(model, chunks, chunk_lengths, num_chunks, device, feature_hook):
    """
    Extract embeddings from multiple 10-second chunks and pool them.

    Args:
        chunks: [B, max_num_chunks, chunk_length]
        chunk_lengths: [B, max_num_chunks]
        num_chunks: [B]
    Returns:
        embeddings: [B, C*2]
    """
    B, max_num_chunks, chunk_length = chunks.shape
    C = 768  # HTS-AT hidden dim

    # Process each chunk position
    all_chunk_embs = []  # will be [max_num_chunks, B, C*2]

    for i in range(max_num_chunks):
        chunk_i = chunks[:, i, :].to(device)  # [B, chunk_length]
        length_i = chunk_lengths[:, i].to(device)  # [B]
        emb_i = extract_chunk_embedding(model, chunk_i, length_i, device, feature_hook)  # [B, C*2]
        all_chunk_embs.append(emb_i)

    # Stack: [max_num_chunks, B, C*2] -> [B, max_num_chunks, C*2]
    all_chunk_embs = torch.stack(all_chunk_embs, dim=1)  # [B, max_num_chunks, C*2]

    # Create mask for valid chunks: [B, max_num_chunks]
    chunk_mask = torch.arange(max_num_chunks, device=device).unsqueeze(0) < num_chunks.unsqueeze(1)
    chunk_mask = chunk_mask.unsqueeze(-1).float()  # [B, max_num_chunks, 1]

    # Mean pooling over valid chunks
    embeddings = (all_chunk_embs * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp(min=1)  # [B, C*2]

    return embeddings


def register_feature_hook(model):
    """Register hook to capture features before pooling (from tscam_conv output)"""
    feature_hook = {"features": None}

    def hook_fn(module, input, output):
        # Capture tscam_conv output: (B, 527, freq, time)
        # After flatten in forward: (B, 527, T)
        # We want features before this conv, which is the input
        # input[0] shape: (B, 768, freq, time)
        x = input[0]
        B, C, F, T = x.shape
        # Flatten to (B, C, F*T) for pooling
        feature_hook["features"] = x.reshape(B, C, -1)

    # Register hook on tscam_conv layer to get its input (768-dim features)
    model.tscam_conv.register_forward_hook(hook_fn)

    return feature_hook


def load_genre_data():
    csv_path = Path(__file__).parent.parent / "data" / "spotify_genre_info_frequent.csv"
    audio_dir = Path(
        "/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node"
    )

    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["genres"]:
                track_id = row["track_id"]
                if (audio_dir / f"{track_id}.mp3").exists():
                    data.append({"track_id": track_id, "genre": row["genres"]})

    df = pd.DataFrame(data)
    print(f"Tracks with audio: {len(df):,}")

    genres = sorted(df["genre"].unique())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}

    print(f"Genres: {len(genres)}")
    return df, genre_to_idx, idx_to_genre


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 16  # Smaller batch for HTS-AT (large transformer)
    num_epochs = 10
    lr = 1e-3

    # Load HTS-AT
    print("\n=== Loading HTS-AT ===")
    checkpoint_path = Path(HTSAT_DIR) / "HTSAT_AudioSet_Saved_6.ckpt"
    encoder = load_htsat(checkpoint_path, device)

    # Register hook for mean+max pooling
    feature_hook = register_feature_hook(encoder)

    # Freeze encoder - only train classifier
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"HTS-AT parameters: {num_params:,} (frozen)")

    # Load data
    print("\n=== Loading Data ===")
    df, genre_to_idx, idx_to_genre = load_genre_data()

    # Split 80:10:10
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["genre"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["genre"]
    )
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    audio_dir = Path(
        "/Users/eomjoonseo/dynplayer_crawler/preprocessing/preview_audio_node"
    )

    train_dataset = AudioGenreDataset(
        train_df["track_id"].values, train_df["genre"].values, genre_to_idx, audio_dir
    )
    val_dataset = AudioGenreDataset(
        val_df["track_id"].values, val_df["genre"].values, genre_to_idx, audio_dir
    )
    test_dataset = AudioGenreDataset(
        test_df["track_id"].values, test_df["genre"].values, genre_to_idx, audio_dir
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        worker_init_fn=seed_worker, generator=g, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn
    )

    # Classifier (HTS-AT embedding dim is 768*2 for mean+max)
    print("\n=== Classifier ===")
    classifier = GenreClassifier(embedding_dim=768 * 2, num_genres=len(genre_to_idx)).to(
        device
    )
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # Training
    print("\n=== Training ===")
    best_val_acc = 0

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        total_loss = 0
        all_preds, all_labels = [], []
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            chunks = batch["chunks"]  # [B, max_num_chunks, chunk_length]
            chunk_lengths = batch["chunk_lengths"]  # [B, max_num_chunks]
            num_chunks = batch["num_chunks"].to(device)  # [B]
            labels = batch["genre_idx"].to(device)

            with torch.no_grad():
                embeddings = extract_embeddings(encoder, chunks, chunk_lengths, num_chunks, device, feature_hook)

            optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                acc = accuracy_score(all_labels, all_preds)
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} Acc: {acc:.4f}"
                )

        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = total_loss / num_batches

        # Validation
        classifier.eval()
        val_preds, val_labels, val_logits_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                chunks = batch["chunks"]
                chunk_lengths = batch["chunk_lengths"]
                num_chunks = batch["num_chunks"].to(device)
                labels = batch["genre_idx"].to(device)
                embeddings = extract_embeddings(encoder, chunks, chunk_lengths, num_chunks, device, feature_hook)
                logits = classifier(embeddings)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_logits_all.append(logits.cpu())

        val_logits_all = torch.cat(val_logits_all, dim=0)
        val_labels_tensor = torch.tensor(val_labels)
        top5_preds = val_logits_all.topk(5, dim=1).indices
        val_top5_acc = (top5_preds == val_labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item()

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Val Top5: {val_top5_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'classifier': classifier.state_dict(),
                'genre_to_idx': genre_to_idx,
                'idx_to_genre': idx_to_genre,
            }, Path(__file__).parent / "genre_htsat_best.pt")
            print(f"  Saved best model")

    # Test
    print("\n=== Test ===")
    checkpoint = torch.load(Path(__file__).parent / "genre_htsat_best.pt")
    classifier.load_state_dict(checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    test_preds, test_labels, test_logits_all = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            chunks = batch["chunks"]
            chunk_lengths = batch["chunk_lengths"]
            num_chunks = batch["num_chunks"].to(device)
            labels = batch["genre_idx"].to(device)
            embeddings = extract_embeddings(encoder, chunks, chunk_lengths, num_chunks, device, feature_hook)
            logits = classifier(embeddings)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_logits_all.append(logits.cpu())

    test_logits_all = torch.cat(test_logits_all, dim=0)
    test_labels_tensor = torch.tensor(test_labels)
    top5_preds = test_logits_all.topk(5, dim=1).indices
    test_top5_acc = (top5_preds == test_labels_tensor.unsqueeze(1)).any(dim=1).float().mean().item()

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="weighted")
    print(f"Test Acc: {test_acc:.4f}, Test Top5: {test_top5_acc:.4f}, Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()
