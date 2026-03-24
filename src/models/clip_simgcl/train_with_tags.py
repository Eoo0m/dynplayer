"""
CLIP training with both captions AND individual tags as text supervision.

Uses SimGCL playlist embeddings (64-dim).
Text inputs:
  1. Captions (GPT-generated descriptions)
  2. Individual tags (each tag creates a separate training pair)

Example: Playlist A with tags ["jazz", "chill", "study"] creates 3 separate pairs:
  - ("jazz" embedding, playlist A embedding)
  - ("chill" embedding, playlist A embedding)
  - ("study" embedding, playlist A embedding)

This allows direct matching: searching "jazz" finds playlists tagged with jazz.
Cosine similarity is used.
"""

import random
import os
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_simgcl.model import CaptionPlaylistCLIP


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def embed_text_batch(client, texts, model="text-embedding-3-large"):
    """Batch text embedding using OpenAI API"""
    res = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in res.data])


def load_simgcl_playlist_embeddings(simgcl_dir, base_dir):
    """Load SimGCL playlist (user) embeddings."""
    print("\n=== Loading SimGCL Playlist Embeddings ===")
    full_path = os.path.join(base_dir, simgcl_dir)
    playlist_embs = np.load(f"{full_path}/model_loo_playlist_embeddings.npy")
    playlist_ids = np.load(f"{full_path}/model_loo_playlist_ids.npy", allow_pickle=True)

    playlist_index = {str(pid): i for i, pid in enumerate(playlist_ids)}
    playlist_ids = np.array([str(pid) for pid in playlist_ids])

    print(f"SimGCL playlist embeddings: {playlist_embs.shape}")
    print(f"Total playlists: {len(playlist_ids)}")

    return playlist_embs, playlist_ids, playlist_index


class PlaylistMultiTextDataset(Dataset):
    """
    Dataset that provides (text_emb, playlist_emb) pairs.
    Each playlist can have multiple text embeddings (caption and/or tags).
    """

    def __init__(self, text_playlist_pairs, playlist_embs, playlist_index):
        """
        Args:
            text_playlist_pairs: list of (text_emb, playlist_id) tuples
            playlist_embs: numpy array of playlist embeddings
            playlist_index: dict mapping playlist_id to index
        """
        self.pairs = text_playlist_pairs
        self.playlist_embs = playlist_embs
        self.playlist_index = playlist_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text_emb, pid = self.pairs[idx]
        playlist_emb = self.playlist_embs[self.playlist_index[pid]]
        return (
            torch.tensor(text_emb, dtype=torch.float32),
            torch.tensor(playlist_emb, dtype=torch.float32),
        )


def train_test_split(playlist_ids, caption_dict, tag_dict, test_ratio=0.2, seed=42):
    """Split playlists into train/test sets."""
    random.seed(seed)
    np.random.seed(seed)

    # Playlists with at least caption or tags
    valid_ids = [pid for pid in playlist_ids if pid in caption_dict or pid in tag_dict]

    if test_ratio == 0:
        print(f"Train playlists: {len(valid_ids):,}")
        print("Test playlists: 0 (test_ratio=0)")
        return valid_ids, [], {}, {}

    shuffled = valid_ids.copy()
    random.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]

    # For test, we only use captions for evaluation (like original)
    test_captions = {pid: caption_dict[pid] for pid in test_ids if pid in caption_dict}

    print(f"Train playlists: {len(train_ids):,}")
    print(f"Test playlists: {len(test_ids):,} (with captions: {len(test_captions)})")

    return train_ids, test_ids, test_captions


def recall_at_k(model, test_captions, playlist_embs, playlist_index, device, k_list=[1, 5, 10, 20]):
    """Recall@K evaluation using cosine similarity."""
    model.eval()

    all_playlist_ids = list(playlist_index.keys())
    all_projected = []

    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(all_playlist_ids), batch_size):
            batch_ids = all_playlist_ids[i:i+batch_size]
            batch_embs = torch.stack([
                torch.tensor(playlist_embs[playlist_index[pid]], dtype=torch.float32)
                for pid in batch_ids
            ]).to(device)

            projected = model.playlist_proj(batch_embs)
            all_projected.append(projected)

    all_projected = torch.cat(all_projected, dim=0)  # Already L2 normalized

    pid_to_idx = {pid: i for i, pid in enumerate(all_playlist_ids)}
    recall_scores = {k: [] for k in k_list}

    for test_pid, caption_emb in tqdm(test_captions.items(), desc="Evaluating Recall@K"):
        if test_pid not in pid_to_idx:
            continue

        with torch.no_grad():
            caption_tensor = torch.tensor(caption_emb, dtype=torch.float32).unsqueeze(0).to(device)
            projected_caption = model.caption_proj(caption_tensor)  # Already L2 normalized

            # Cosine similarity (embeddings are normalized)
            similarities = torch.matmul(projected_caption, all_projected.T).squeeze(0)

            max_k = max(k_list)
            _, top_indices = torch.topk(similarities, k=max_k)
            top_indices = top_indices.cpu().numpy()

        target_idx = pid_to_idx[test_pid]

        for k in k_list:
            if target_idx in top_indices[:k]:
                recall_scores[k].append(1.0)
            else:
                recall_scores[k].append(0.0)

    avg_recall = {k: np.mean(scores) for k, scores in recall_scores.items()}
    return avg_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simgcl-dir", default="models/simgcl_randneg/outputs/min5_win10")
    parser.add_argument("--playlist-csv", default="clip copy/playlists_with_captions_tags_filtered.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="clip_simgcl/exp_with_tags")
    parser.add_argument("--playlist-grad-scale", type=float, default=0.1,
                        help="Gradient scale for playlist projector")
    parser.add_argument("--tag-weight", type=float, default=1.0,
                        help="Weight for tag samples relative to captions (1.0 = equal)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = os.path.join(args.output_dir, "clip")

    # Base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # Load SimGCL embeddings
    playlist_embs, playlist_ids, playlist_index = load_simgcl_playlist_embeddings(
        args.simgcl_dir, base_dir
    )
    playlist_dim = playlist_embs.shape[1]

    # Load playlist data
    print("\n=== Loading Playlist Data ===")
    playlist_df = pd.read_csv(os.path.join(base_dir, args.playlist_csv))
    print(f"Total playlist records: {len(playlist_df)}")

    # Filter to playlists in SimGCL
    playlist_df = playlist_df[playlist_df["playlist_id"].isin(playlist_index)]
    print(f"Playlists in SimGCL: {len(playlist_df)}")

    # Extract captions
    caption_df = playlist_df[playlist_df["caption"].notna() & (playlist_df["caption"] != "")]
    captions = {row["playlist_id"]: row["caption"] for _, row in caption_df.iterrows()}
    print(f"Playlists with captions: {len(captions)}")

    # Extract individual tags (use filtered tags column if available)
    def parse_tags_list(row):
        # Use tags_filtered column if available, otherwise use tags
        tags_str = row.get("tags_filtered", row.get("tags", "[]"))
        if pd.isna(tags_str) or tags_str == "[]":
            return []
        try:
            tags = ast.literal_eval(tags_str)
            if isinstance(tags, list) and len(tags) > 0:
                return tags  # Return individual tags as list
            return []
        except:
            return []

    playlist_df["tags_list"] = playlist_df.apply(parse_tags_list, axis=1)

    # Build playlist -> tags mapping (individual tags)
    playlist_tags = {}
    all_unique_tags = set()
    for _, row in playlist_df.iterrows():
        tags_list = row["tags_list"]
        if tags_list:
            playlist_tags[row["playlist_id"]] = tags_list
            all_unique_tags.update(tags_list)

    print(f"Playlists with tags: {len(playlist_tags)}")
    print(f"Unique tags: {len(all_unique_tags)}")

    # ======================================================
    # Embed captions
    # ======================================================
    print("\n=== Embedding Captions ===")
    caption_pids = list(captions.keys())
    caption_texts = [captions[pid] for pid in caption_pids]

    caption_embeddings = []
    BATCH = 512
    for i in tqdm(range(0, len(caption_texts), BATCH), desc="Embedding captions"):
        batch = caption_texts[i:i+BATCH]
        embs = embed_text_batch(client, batch)
        caption_embeddings.append(embs)

    caption_embeddings = np.vstack(caption_embeddings)
    caption_emb_dict = {pid: caption_embeddings[i] for i, pid in enumerate(caption_pids)}
    print(f"Caption embeddings: {caption_embeddings.shape}")

    # ======================================================
    # Embed individual tags (each unique tag once)
    # ======================================================
    print("\n=== Embedding Individual Tags ===")
    unique_tags_list = sorted(list(all_unique_tags))
    print(f"Embedding {len(unique_tags_list)} unique tags...")

    tag_embeddings_list = []
    for i in tqdm(range(0, len(unique_tags_list), BATCH), desc="Embedding tags"):
        batch = unique_tags_list[i:i+BATCH]
        embs = embed_text_batch(client, batch)
        tag_embeddings_list.append(embs)

    tag_embeddings_array = np.vstack(tag_embeddings_list)
    # Map: tag text -> embedding
    tag_text_to_emb = {tag: tag_embeddings_array[i] for i, tag in enumerate(unique_tags_list)}
    print(f"Tag embeddings: {tag_embeddings_array.shape}")

    caption_dim = caption_embeddings.shape[1]  # 3072

    # ======================================================
    # Train/Test Split
    # ======================================================
    print(f"\n=== Train/Test Split (test_ratio={args.test_ratio}) ===")
    train_ids, test_ids, test_captions = train_test_split(
        list(playlist_index.keys()), caption_emb_dict, playlist_tags,
        test_ratio=args.test_ratio, seed=args.seed
    )

    # ======================================================
    # Build training pairs (text_emb, playlist_id)
    # Each individual tag creates a separate pair
    # ======================================================
    print("\n=== Building Training Pairs ===")
    train_pairs = []
    caption_pair_count = 0
    tag_pair_count = 0

    for pid in train_ids:
        # Add caption pair
        if pid in caption_emb_dict:
            train_pairs.append((caption_emb_dict[pid], pid))
            caption_pair_count += 1

        # Add individual tag pairs (each tag = separate pair)
        if pid in playlist_tags:
            for tag in playlist_tags[pid]:
                if tag in tag_text_to_emb:
                    train_pairs.append((tag_text_to_emb[tag], pid))
                    tag_pair_count += 1

    print(f"Total training pairs: {len(train_pairs)}")
    print(f"  - From captions: {caption_pair_count}")
    print(f"  - From individual tags: {tag_pair_count}")

    # ======================================================
    # Model & Dataset
    # ======================================================
    model = CaptionPlaylistCLIP(
        caption_dim, playlist_dim, out_dim=args.output_dim, temperature=0.07
    ).to(device)

    print("\n" + "=" * 80)
    print("CLIP: Text (Caption + Tags) <-> Playlist Embedding (SimGCL)")
    print("=" * 80)
    print(f"Playlist dim: {playlist_dim} (SimGCL)")
    print(f"Text dim: {caption_dim} (OpenAI)")
    print(f"Output dim: {args.output_dim}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Playlist grad scale: {args.playlist_grad_scale}")

    dataset = PlaylistMultiTextDataset(train_pairs, playlist_embs, playlist_index)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW([
        {"params": model.caption_proj.parameters(), "lr": args.lr},
        {"params": model.playlist_proj.parameters(), "lr": args.lr * args.playlist_grad_scale},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Caption proj LR: {args.lr}, Playlist proj LR: {args.lr * args.playlist_grad_scale}")
    print(f"\nDataset size: {len(dataset)} pairs")

    # ======================================================
    # Training
    # ======================================================
    best_recall = 0.0
    best_recall_scores = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        avg_pos = 0
        avg_neg = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for text_emb, playlist_emb in pbar:
            text_emb = text_emb.to(device)
            playlist_emb = playlist_emb.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, pos_sim, neg_sim = model(text_emb, playlist_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_pos += pos_sim
            avg_neg += neg_sim

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pos": f"{pos_sim:.4f}",
                "neg": f"{neg_sim:.4f}",
            })

        scheduler.step()

        batches = len(loader)
        print(
            f"[Epoch {epoch+1}] loss={total_loss/batches:.4f} | "
            f"pos_sim={avg_pos/batches:.4f} | neg_sim={avg_neg/batches:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Evaluation
        if args.test_ratio > 0 and ((epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs):
            print(f"\n--- Evaluation at Epoch {epoch+1} ---")
            recall_scores = recall_at_k(
                model, test_captions, playlist_embs, playlist_index, device, k_list=[1, 5, 10, 20]
            )

            print("Recall@K:")
            for k, score in recall_scores.items():
                print(f"  Recall@{k}: {score:.4f}")

            if recall_scores[10] > best_recall:
                best_recall = recall_scores[10]
                best_recall_scores = recall_scores.copy()
                print(f"\n  New best Recall@10: {best_recall:.4f}")
                torch.save(model.state_dict(), f"{output_prefix}_best.pt")

    # ======================================================
    # Final Evaluation
    # ======================================================
    if args.test_ratio > 0:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        final_recall = recall_at_k(
            model, test_captions, playlist_embs, playlist_index, device, k_list=[1, 5, 10, 20, 50]
        )

        print("\nFinal Recall@K:")
        for k, score in final_recall.items():
            print(f"  Recall@{k}: {score:.4f}")

        if best_recall_scores:
            print(f"\nBest Epoch Recall@K:")
            for k, score in best_recall_scores.items():
                print(f"  Recall@{k}: {score:.4f}")
        print("=" * 60)

    # Save model
    torch.save(model.state_dict(), f"{output_prefix}.pt")
    print(f"\nModel saved: {output_prefix}.pt")

    # ======================================================
    # Save projected playlist embeddings
    # ======================================================
    print("\n=== Generating Projected Playlist Embeddings ===")
    model.eval()

    all_projected = []
    batch_size_infer = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(playlist_ids), batch_size_infer)):
            batch_ids = playlist_ids[i:i+batch_size_infer]
            batch_embs = torch.stack([
                torch.tensor(playlist_embs[playlist_index[pid]], dtype=torch.float32)
                for pid in batch_ids
            ]).to(device)

            projected = model.playlist_proj(batch_embs).cpu().numpy()
            all_projected.append(projected)

    all_projected = np.vstack(all_projected)

    np.save(f"{output_prefix}_playlist_ids.npy", playlist_ids)
    np.save(f"{output_prefix}_playlist_projected.npy", all_projected)

    print(f"\nSaved playlist embeddings: {all_projected.shape}")
    print(f"   - {output_prefix}_playlist_ids.npy")
    print(f"   - {output_prefix}_playlist_projected.npy")


if __name__ == "__main__":
    main()
