"""
CLIP training with dot product contrastive loss + L2 regularization.

Unlike the cosine similarity version, this version:
1. Uses dot product (no normalization) for similarity
2. Adds L2 regularization on embedding norms to reflect popularity implicitly
   (popular tracks/playlists will have larger norms)
"""

import random
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PlaylistCaptionDataset(Dataset):
    def __init__(self, playlist_ids, caption_emb_dict, playlist_embs, playlist_index):
        self.playlist_ids = [
            pid for pid in playlist_ids
            if pid in caption_emb_dict and pid in playlist_index
        ]
        self.caption_emb_dict = caption_emb_dict
        self.playlist_embs = playlist_embs
        self.playlist_index = playlist_index

    def __len__(self):
        return len(self.playlist_ids)

    def __getitem__(self, idx):
        playlist_id = self.playlist_ids[idx]
        caption_emb = self.caption_emb_dict[playlist_id]
        playlist_emb = self.playlist_embs[self.playlist_index[playlist_id]]
        return (
            torch.tensor(caption_emb, dtype=torch.float32),
            torch.tensor(playlist_emb, dtype=torch.float32),
        )


class ProjectionMLPDot(nn.Module):
    """Projection MLP without final L2 normalization (for dot product)."""

    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.proj_in(x)
        h = self.activation(h)
        h = self.norm(h)

        residual = h
        h = self.block1(h)
        h = h + residual

        residual = h
        h = self.block2(h)
        h = h + residual

        h = self.proj_out(h)
        # NO L2 normalization - use raw embeddings for dot product
        return h


class CaptionPlaylistCLIPDot(nn.Module):
    """CLIP model using dot product similarity + L2 regularization."""

    def __init__(self, caption_dim, playlist_dim, out_dim=512, temperature=0.07, l2_weight=0.01):
        super().__init__()
        self.caption_proj = ProjectionMLPDot(caption_dim, out_dim)
        self.playlist_proj = ProjectionMLPDot(playlist_dim, out_dim)
        self.temperature = temperature
        self.l2_weight = l2_weight

    def forward(self, caption, playlist):
        """
        caption: (B, caption_dim)
        playlist: (B, playlist_dim)

        Returns:
            loss: contrastive loss + L2 regularization
            pos_sim: average positive similarity
            neg_sim: average negative similarity
        """
        z_caption = self.caption_proj(caption)  # (B, out_dim)
        z_playlist = self.playlist_proj(playlist)  # (B, out_dim)

        # Dot product similarity matrix
        sim_raw = torch.matmul(z_caption, z_playlist.T)  # (B, B)

        # Temperature scaling for loss
        sim_matrix = sim_raw / self.temperature

        # Contrastive loss (InfoNCE, bidirectional)
        labels = torch.arange(len(z_caption), device=z_caption.device)
        loss_c2p = F.cross_entropy(sim_matrix, labels)
        loss_p2c = F.cross_entropy(sim_matrix.T, labels)
        contrastive_loss = (loss_c2p + loss_p2c) / 2

        # L2 regularization on embedding norms
        l2_caption = (z_caption.norm(2, dim=1) ** 2).mean()
        l2_playlist = (z_playlist.norm(2, dim=1) ** 2).mean()
        l2_loss = self.l2_weight * (l2_caption + l2_playlist) / 2

        total_loss = contrastive_loss + l2_loss

        # Monitoring
        pos_sim = torch.diagonal(sim_raw).mean()
        neg_sim = (sim_raw.sum() - torch.diagonal(sim_raw).sum()) / (
            sim_raw.numel() - len(z_caption)
        )

        return total_loss, pos_sim.item(), neg_sim.item(), contrastive_loss.item(), l2_loss.item()


def embed_text_batch(client, texts, model="text-embedding-3-large"):
    res = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in res.data])


def load_simgcl_playlist_embeddings(simgcl_dir, base_dir):
    print("\n=== Loading SimGCL Playlist Embeddings ===")
    full_path = os.path.join(base_dir, simgcl_dir)
    playlist_embs = np.load(f"{full_path}/model_loo_playlist_embeddings.npy")
    playlist_ids = np.load(f"{full_path}/model_loo_playlist_ids.npy", allow_pickle=True)
    playlist_index = {str(pid): i for i, pid in enumerate(playlist_ids)}
    playlist_ids = np.array([str(pid) for pid in playlist_ids])
    print(f"SimGCL playlist embeddings: {playlist_embs.shape}")
    return playlist_embs, playlist_ids, playlist_index


def train_test_split(playlist_ids, caption_emb_dict, test_ratio=0.2, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    valid_ids = [pid for pid in playlist_ids if pid in caption_emb_dict]

    if test_ratio == 0:
        print(f"Train playlists: {len(valid_ids):,}")
        return valid_ids, [], {}

    shuffled = valid_ids.copy()
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]
    test_captions = {pid: caption_emb_dict[pid] for pid in test_ids}

    print(f"Train playlists: {len(train_ids):,}")
    print(f"Test playlists: {len(test_ids):,}")
    return train_ids, test_ids, test_captions


def recall_at_k(model, test_captions, playlist_embs, playlist_index, device, k_list=[1, 5, 10, 20], use_cosine=False):
    """Recall@K evaluation. use_cosine=True for cosine similarity, False for dot product."""
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

    all_projected = torch.cat(all_projected, dim=0)

    # Normalize for cosine similarity
    if use_cosine:
        all_projected = F.normalize(all_projected, dim=-1)

    pid_to_idx = {pid: i for i, pid in enumerate(all_playlist_ids)}

    recall_scores = {k: [] for k in k_list}

    for test_pid, caption_emb in tqdm(test_captions.items(), desc=f"Evaluating Recall@K ({'cos' if use_cosine else 'dot'})"):
        if test_pid not in pid_to_idx:
            continue

        with torch.no_grad():
            caption_tensor = torch.tensor(caption_emb, dtype=torch.float32).unsqueeze(0).to(device)
            projected_caption = model.caption_proj(caption_tensor)

            # Normalize for cosine similarity
            if use_cosine:
                projected_caption = F.normalize(projected_caption, dim=-1)

            # Similarity (dot or cosine)
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
    parser.add_argument("--playlist-csv", default="clip copy/playlists_with_captions.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--playlist-grad-scale", type=float, default=1.0)
    parser.add_argument("--l2-weight", type=float, default=0.01,
                        help="L2 regularization weight on embedding norms (default: 0.01)")
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = os.path.join(args.output_dir, "clip")

    # Base directory (parent of clip_simgcl)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # Load SimGCL embeddings
    playlist_embs, playlist_ids, playlist_index = load_simgcl_playlist_embeddings(args.simgcl_dir, base_dir)
    playlist_dim = playlist_embs.shape[1]

    # Load captions
    print("\n=== Loading Playlist Captions ===")
    playlist_df = pd.read_csv(os.path.join(base_dir, args.playlist_csv))
    print(f"Total playlist records: {len(playlist_df)}")

    playlist_df = playlist_df[playlist_df["caption"].notna()]
    playlist_df = playlist_df[playlist_df["caption"] != ""]
    playlist_df = playlist_df[playlist_df["playlist_id"].isin(playlist_index)]
    print(f"Playlists with captions (in SimGCL): {len(playlist_df)}")

    # Embed captions
    print("\n=== Embedding Captions with GPT ===")
    all_playlist_ids = playlist_df["playlist_id"].tolist()
    all_captions = playlist_df["caption"].tolist()

    caption_embeddings = []
    BATCH = 512
    for i in tqdm(range(0, len(all_captions), BATCH), desc="Embedding captions"):
        batch_captions = all_captions[i:i+BATCH]
        batch_embs = embed_text_batch(client, batch_captions)
        caption_embeddings.append(batch_embs)

    caption_embeddings = np.vstack(caption_embeddings)
    caption_emb_dict = {pid: caption_embeddings[i] for i, pid in enumerate(all_playlist_ids)}
    caption_dim = caption_embeddings.shape[1]
    print(f"Caption embeddings: {caption_embeddings.shape}")

    # Train/Test split
    print(f"\n=== Train/Test Split (test_ratio={args.test_ratio}) ===")
    train_ids, test_ids, test_captions = train_test_split(
        all_playlist_ids, caption_emb_dict, test_ratio=args.test_ratio, seed=args.seed
    )

    # Model
    model = CaptionPlaylistCLIPDot(
        caption_dim, playlist_dim, out_dim=args.output_dim,
        temperature=args.temperature, l2_weight=args.l2_weight
    ).to(device)

    print("\n" + "=" * 80)
    print("CLIP (Dot Product + L2 Regularization)")
    print("=" * 80)
    print(f"Playlist dim: {playlist_dim}")
    print(f"Caption dim: {caption_dim}")
    print(f"Output dim: {args.output_dim}")
    print(f"Temperature: {args.temperature}")
    print(f"L2 weight: {args.l2_weight}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Playlist grad scale: {args.playlist_grad_scale}")

    # Dataset
    dataset = PlaylistCaptionDataset(train_ids, caption_emb_dict, playlist_embs, playlist_index)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.caption_proj.parameters(), "lr": args.lr},
        {"params": model.playlist_proj.parameters(), "lr": args.lr * args.playlist_grad_scale},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Dataset size: {len(dataset)} playlists")

    # Training
    best_recall = 0.0
    best_recall_scores = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_contrastive = 0
        total_l2 = 0
        avg_pos = 0
        avg_neg = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for caption_emb, playlist_emb in pbar:
            caption_emb = caption_emb.to(device)
            playlist_emb = playlist_emb.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, pos_sim, neg_sim, contrastive, l2 = model(caption_emb, playlist_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += contrastive
            total_l2 += l2
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
            f"[Epoch {epoch+1}] loss={total_loss/batches:.4f} "
            f"(cont={total_contrastive/batches:.4f}, l2={total_l2/batches:.4f}) | "
            f"pos_sim={avg_pos/batches:.4f} | neg_sim={avg_neg/batches:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Evaluation
        if args.test_ratio > 0 and ((epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs):
            print(f"\n--- Evaluation at Epoch {epoch+1} ---")

            # Dot product evaluation only
            recall_dot = recall_at_k(
                model, test_captions, playlist_embs, playlist_index, device, k_list=[1, 5, 10, 20], use_cosine=False
            )

            print("Recall@K (dot):", {k: round(v, 4) for k, v in recall_dot.items()})

            # Use dot product Recall@10 for best model selection
            if recall_dot[10] > best_recall:
                best_recall = recall_dot[10]
                best_recall_scores = recall_dot.copy()
                print(f"\n  ✨ New best Recall@10: {best_recall:.4f}")
                torch.save(model.state_dict(), f"{output_prefix}_best.pt")

    # Final evaluation
    if args.test_ratio > 0:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        final_recall = recall_at_k(
            model, test_captions, playlist_embs, playlist_index, device, k_list=[1, 5, 10, 20, 50], use_cosine=False
        )

        print("\nFinal Recall@K (dot):")
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

    # Save projected embeddings
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
