"""
CLIP training using SimGCL playlist embeddings and GPT captions.

Uses pre-trained SimGCL user (playlist) embeddings (64-dim) instead of track average embeddings.
Captions are embedded using OpenAI text-embedding-3-large.
"""

import random
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path for model import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_simgcl.model import PlaylistCaptionDataset, CaptionPlaylistCLIP


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


def load_simgcl_playlist_embeddings(simgcl_dir):
    """Load SimGCL playlist (user) embeddings."""
    print("\n=== Loading SimGCL Playlist Embeddings ===")

    playlist_embs = np.load(f"{simgcl_dir}/model_loo_playlist_embeddings.npy")
    playlist_ids = np.load(
        f"{simgcl_dir}/model_loo_playlist_ids.npy", allow_pickle=True
    )

    # Create index mapping
    playlist_index = {str(pid): i for i, pid in enumerate(playlist_ids)}
    playlist_ids = np.array([str(pid) for pid in playlist_ids])

    print(f"SimGCL playlist embeddings: {playlist_embs.shape}")
    print(f"Total playlists: {len(playlist_ids)}")

    return playlist_embs, playlist_ids, playlist_index


def train_test_split(playlist_ids, caption_emb_dict, test_ratio=0.2, seed=42):
    """Split playlists into train/test sets."""
    random.seed(seed)
    np.random.seed(seed)

    # Filter to playlists with captions
    valid_ids = [pid for pid in playlist_ids if pid in caption_emb_dict]

    if test_ratio == 0:
        print(f"Train playlists: {len(valid_ids):,}")
        print("Test playlists: 0 (test_ratio=0, using all data for training)")
        return valid_ids, [], {}

    # Shuffle and split
    shuffled = valid_ids.copy()
    random.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]

    test_captions = {pid: caption_emb_dict[pid] for pid in test_ids}

    print(f"Train playlists: {len(train_ids):,}")
    print(f"Test playlists: {len(test_ids):,}")

    return train_ids, test_ids, test_captions


def recall_at_k(
    model,
    test_captions,
    playlist_embs,
    playlist_index,
    device,
    k_list=[1, 5, 10, 20],
):
    """
    Recall@K evaluation: how well can we find the correct playlist from caption.
    """
    model.eval()

    # Project all playlist embeddings
    all_playlist_ids = list(playlist_index.keys())
    all_projected = []

    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(all_playlist_ids), batch_size):
            batch_ids = all_playlist_ids[i : i + batch_size]
            batch_embs = torch.stack(
                [
                    torch.tensor(
                        playlist_embs[playlist_index[pid]], dtype=torch.float32
                    )
                    for pid in batch_ids
                ]
            ).to(device)

            projected = model.playlist_proj(batch_embs)
            all_projected.append(projected)

    all_projected = torch.cat(all_projected, dim=0)  # (N, out_dim)

    # Playlist ID mapping
    pid_to_idx = {pid: i for i, pid in enumerate(all_playlist_ids)}

    # Evaluate test captions
    recall_scores = {k: [] for k in k_list}

    for test_pid, caption_emb in tqdm(
        test_captions.items(), desc="Evaluating Recall@K"
    ):
        if test_pid not in pid_to_idx:
            continue

        with torch.no_grad():
            caption_tensor = (
                torch.tensor(caption_emb, dtype=torch.float32).unsqueeze(0).to(device)
            )
            projected_caption = model.caption_proj(caption_tensor)  # (1, out_dim)

            # Compute similarity with all playlists
            similarities = torch.matmul(projected_caption, all_projected.T).squeeze(
                0
            )  # (N,)

            # Top-K
            max_k = max(k_list)
            _, top_indices = torch.topk(similarities, k=max_k)
            top_indices = top_indices.cpu().numpy()

        # Target index
        target_idx = pid_to_idx[test_pid]

        # Compute recall for each K
        for k in k_list:
            if target_idx in top_indices[:k]:
                recall_scores[k].append(1.0)
            else:
                recall_scores[k].append(0.0)

    # Average
    avg_recall = {k: np.mean(scores) for k, scores in recall_scores.items()}

    return avg_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simgcl-dir",
        default="simgcl_randneg/outputs/min5_win10",
        help="Directory containing SimGCL embeddings",
    )
    parser.add_argument(
        "--playlist-csv",
        default="clip copy/playlists_with_captions.csv",
        help="CSV with playlist captions",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, help="Output directory for this experiment")
    parser.add_argument("--playlist-grad-scale", type=float, default=1.0,
                        help="Gradient scale for playlist projector (default: 1.0, set to 0.1 for text-fast)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = os.path.join(args.output_dir, "clip")

    # ======================================================
    # 1) Load SimGCL playlist embeddings
    # ======================================================
    playlist_embs, playlist_ids, playlist_index = load_simgcl_playlist_embeddings(
        args.simgcl_dir
    )
    playlist_dim = playlist_embs.shape[1]  # 64

    # ======================================================
    # 2) Load playlist captions from CSV
    # ======================================================
    print("\n=== Loading Playlist Captions ===")
    playlist_df = pd.read_csv(args.playlist_csv)
    print(f"Total playlist records: {len(playlist_df)}")

    # Filter playlists with captions and in SimGCL
    playlist_df = playlist_df[playlist_df["caption"].notna()]
    playlist_df = playlist_df[playlist_df["caption"] != ""]
    playlist_df = playlist_df[playlist_df["playlist_id"].isin(playlist_index)]
    print(f"Playlists with captions (in SimGCL): {len(playlist_df)}")

    # ======================================================
    # 3) Embed captions with GPT
    # ======================================================
    print("\n=== Embedding Captions with GPT ===")

    all_playlist_ids = playlist_df["playlist_id"].tolist()
    all_captions = playlist_df["caption"].tolist()

    caption_embeddings = []
    BATCH = 512
    for i in tqdm(range(0, len(all_captions), BATCH), desc="Embedding captions"):
        batch_captions = all_captions[i : i + BATCH]
        batch_embs = embed_text_batch(client, batch_captions)
        caption_embeddings.append(batch_embs)

    caption_embeddings = np.vstack(caption_embeddings)
    caption_emb_dict = {
        pid: caption_embeddings[i] for i, pid in enumerate(all_playlist_ids)
    }

    caption_dim = caption_embeddings.shape[1]  # 3072
    print(f"Caption embeddings: {caption_embeddings.shape}")
    print(f"Caption dim: {caption_dim}")

    # ======================================================
    # 4) Train/Test Split
    # ======================================================
    print(f"\n=== Train/Test Split (test_ratio={args.test_ratio}) ===")
    train_ids, test_ids, test_captions = train_test_split(
        all_playlist_ids, caption_emb_dict, test_ratio=args.test_ratio, seed=args.seed
    )

    # ======================================================
    # 5) Model & Dataset
    # ======================================================
    model = CaptionPlaylistCLIP(
        caption_dim, playlist_dim, out_dim=args.output_dim, temperature=0.07
    ).to(device)

    print("\n" + "=" * 80)
    print("CLIP: Caption (GPT) <-> Playlist Embedding (SimGCL)")
    print("=" * 80)
    print(f"Playlist dim: {playlist_dim} (SimGCL)")
    print(f"Caption dim: {caption_dim} (OpenAI)")
    print(f"Output dim: {args.output_dim}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Playlist grad scale: {args.playlist_grad_scale}")

    # Dataset (train only)
    dataset = PlaylistCaptionDataset(
        train_ids, caption_emb_dict, playlist_embs, playlist_index
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Separate parameter groups: playlist_proj gets scaled learning rate
    optimizer = torch.optim.AdamW([
        {"params": model.caption_proj.parameters(), "lr": args.lr},
        {"params": model.playlist_proj.parameters(), "lr": args.lr * args.playlist_grad_scale},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Caption proj LR: {args.lr}, Playlist proj LR: {args.lr * args.playlist_grad_scale}")

    print(f"\nDataset size: {len(dataset)} playlists")

    # ======================================================
    # 6) Training
    # ======================================================
    best_recall = 0.0
    best_recall_scores = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        avg_pos = 0
        avg_neg = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for caption_emb, playlist_emb in pbar:
            caption_emb = caption_emb.to(device)
            playlist_emb = playlist_emb.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, pos_sim, neg_sim = model(caption_emb, playlist_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_pos += pos_sim
            avg_neg += neg_sim

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "pos": f"{pos_sim:.4f}",
                    "neg": f"{neg_sim:.4f}",
                }
            )

        scheduler.step()

        batches = len(loader)
        print(
            f"[Epoch {epoch+1}] loss={total_loss/batches:.4f} | "
            f"pos_sim={avg_pos/batches:.4f} | neg_sim={avg_neg/batches:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Evaluation
        if args.test_ratio > 0 and (
            (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs
        ):
            print(f"\n--- Evaluation at Epoch {epoch+1} ---")
            recall_scores = recall_at_k(
                model,
                test_captions,
                playlist_embs,
                playlist_index,
                device,
                k_list=[1, 5, 10, 20],
            )

            print("Recall@K:")
            for k, score in recall_scores.items():
                print(f"  Recall@{k}: {score:.4f}")

            # Best model save
            if recall_scores[10] > best_recall:
                best_recall = recall_scores[10]
                best_recall_scores = recall_scores.copy()
                print(f"\n  New best Recall@10: {best_recall:.4f}")
                torch.save(model.state_dict(), f"{output_prefix}_best.pt")

    # ======================================================
    # 7) Final Evaluation
    # ======================================================
    if args.test_ratio > 0:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        final_recall = recall_at_k(
            model,
            test_captions,
            playlist_embs,
            playlist_index,
            device,
            k_list=[1, 5, 10, 20, 50],
        )

        print("\nFinal Recall@K:")
        for k, score in final_recall.items():
            print(f"  Recall@{k}: {score:.4f}")

        if best_recall_scores:
            print(f"\nBest Epoch Recall@K:")
            for k, score in best_recall_scores.items():
                print(f"  Recall@{k}: {score:.4f}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Training completed (no evaluation, test_ratio=0)")
        print("=" * 60)

    # Save model
    torch.save(model.state_dict(), f"{output_prefix}.pt")
    print(f"\nModel saved: {output_prefix}.pt")

    # ======================================================
    # 8) Save projected playlist embeddings
    # ======================================================
    print("\n=== Generating Projected Playlist Embeddings ===")
    model.eval()

    all_projected = []
    batch_size_infer = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(playlist_ids), batch_size_infer)):
            batch_ids = playlist_ids[i : i + batch_size_infer]

            batch_embs = torch.stack(
                [
                    torch.tensor(
                        playlist_embs[playlist_index[pid]], dtype=torch.float32
                    )
                    for pid in batch_ids
                ]
            ).to(device)

            projected = model.playlist_proj(batch_embs).cpu().numpy()
            all_projected.append(projected)

    all_projected = np.vstack(all_projected)

    # Save
    np.save(f"{output_prefix}_playlist_ids.npy", playlist_ids)
    np.save(f"{output_prefix}_playlist_projected.npy", all_projected)

    print(f"\nSaved playlist embeddings: {all_projected.shape}")
    print(f"   - {output_prefix}_playlist_ids.npy")
    print(f"   - {output_prefix}_playlist_projected.npy")


if __name__ == "__main__":
    main()
