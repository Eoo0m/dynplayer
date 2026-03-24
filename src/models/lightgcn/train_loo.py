"""
Train LightGCN with LOO (Leave-One-Out) evaluation - Dot Product Version

Uses dot product similarity with L2 regularization.
Original LightGCN paper style.

Usage:
    python lightgcn_dot/train_loo.py --dataset min5_win10
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
import yaml
import random

from model import LightGCN, bpr_loss


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_negative(num_items, pos_items, n_samples=1):
    """Sample negative items (not in pos_items)"""
    neg_items = []
    for _ in range(n_samples):
        neg = np.random.randint(0, num_items)
        while neg in pos_items:
            neg = np.random.randint(0, num_items)
        neg_items.append(neg)
    return neg_items


def evaluate_retrieval(model, train_dict, test_dict, num_items, device, k_list=[10, 20], use_cosine=False):
    """Evaluate using LOO test set"""
    model.eval()

    with torch.no_grad():
        playlist_emb, track_emb = model()
        playlist_emb = playlist_emb.cpu().numpy()
        track_emb = track_emb.cpu().numpy()
        if use_cosine:
            playlist_emb = playlist_emb / (np.linalg.norm(playlist_emb, axis=1, keepdims=True) + 1e-10)
            track_emb = track_emb / (np.linalg.norm(track_emb, axis=1, keepdims=True) + 1e-10)

    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    for playlist, test_items in tqdm(test_dict.items(), desc="Evaluating", leave=False):
        if playlist not in train_dict or len(test_items) == 0 or len(train_dict[playlist]) == 0:
            continue

        train_items = train_dict[playlist]
        p_emb = playlist_emb[playlist]
        scores = track_emb @ p_emb

        # Mask train items
        for ti in train_items:
            scores[ti] = -np.inf

        # Get top-k
        max_k = max(k_list)
        topk = np.argpartition(scores, -max_k)[-max_k:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        # Compute metrics
        rel = np.array([1 if i in test_items else 0 for i in topk])
        for k in k_list:
            r = rel[:k]
            recall = r.sum() / len(test_items)

            dcg = (r / np.log2(np.arange(2, len(r) + 2))).sum()
            n_relevant = min(len(test_items), k)
            idcg = (np.ones(n_relevant) / np.log2(np.arange(2, n_relevant + 2))).sum()
            ndcg = dcg / idcg if idcg > 0 else 0.0

            recalls[k].append(recall)
            ndcgs[k].append(ndcg)

    return {
        f"recall@{k}": float(np.mean(v)) if v else 0.0 for k, v in recalls.items()
    } | {f"ndcg@{k}": float(np.mean(v)) if v else 0.0 for k, v in ndcgs.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["min2_win10", "min5_win10"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    # Find config.yaml relative to script
    if args.config:
        config_path = args.config
    else:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"

    config = load_config(config_path)
    dataset_config = config["data"]["datasets"][args.dataset]

    seed = config["training"]["seed"]
    embedding_dim = config["training"]["embedding_dim"]
    epochs = config["training"]["epochs"]
    eval_every = config["training"]["eval_every"]
    k_values = config["training"]["k_values"]

    lgcn_config = config["lightgcn_loo"]
    n_layers = lgcn_config["n_layers"]
    batch_size = lgcn_config["batch_size"]
    lr = lgcn_config["lr"]
    samples_per_user = lgcn_config["samples_per_user"]
    reg_weight = lgcn_config["reg_weight"]

    script_dir = Path(__file__).parent
    data_dir = script_dir / "outputs" / args.dataset
    output_prefix = data_dir / "model_loo"

    # Early stopping config
    patience_epochs = 15  # Stop if no improvement for 15 epochs
    patience_evals = patience_epochs // eval_every  # Number of evals without improvement

    print(f"=== LightGCN LOO Training (Dot Product) ===")
    print(f"Dataset: {args.dataset} ({dataset_config['description']})")
    print(f"Data dir: {data_dir}")
    print(f"Embedding dim: {embedding_dim}, Layers: {n_layers}, Epochs: {epochs}")
    print(f"BPR: dot product, L2 reg={reg_weight}, Adam optimizer")
    print(f"Early stopping: patience={patience_epochs} epochs ({patience_evals} evals)")

    set_seed(seed)

    # Device setup
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load LOO split
    split = np.load(data_dir / "loo_split.npz", allow_pickle=True)
    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()
    num_users = int(split["num_users"])
    num_items = int(split["num_items"])
    playlist_to_idx = split["playlist_to_idx"].item()
    track_to_idx = split["track_to_idx"].item()
    idx_to_playlist = {v: k for k, v in playlist_to_idx.items()}
    idx_to_track = {v: k for k, v in track_to_idx.items()}

    # Load bipartite graph
    graph_sp = sp.load_npz(data_dir / "train_graph.npz")

    print(f"Playlists: {num_users:,}")
    print(f"Tracks: {num_items:,}")
    print(f"Train playlists: {len(train_dict):,}")
    print(f"Test playlists: {len(test_dict):,}")
    print(f"Graph shape: {graph_sp.shape}, nnz: {graph_sp.nnz:,}")

    # Convert graph to PyTorch sparse tensor
    graph_sp = graph_sp.tocoo()
    indices = torch.LongTensor(np.vstack([graph_sp.row, graph_sp.col]))
    values = torch.FloatTensor(graph_sp.data)
    shape = torch.Size(graph_sp.shape)
    graph_tensor = torch.sparse_coo_tensor(indices, values, shape, device=device)

    # Build model
    model = LightGCN(num_users, num_items, embedding_dim, n_layers=n_layers).to(device)
    model.setup_graph(graph_tensor)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (Adam for dot product)
    opt = optim.Adam(model.parameters(), lr=lr)

    metrics_history = []
    best_recall = 0.0
    best_epoch = 0
    evals_without_improvement = 0
    best_model_state = None

    # Training loop
    for ep in range(1, epochs + 1):
        model.train()

        # Create training samples
        playlist_ids = []
        pos_track_ids = []
        neg_track_ids = []

        for playlist, tracks in train_dict.items():
            for _ in range(samples_per_user):
                pos_track = np.random.choice(tracks)
                neg_track = sample_negative(num_items, tracks, n_samples=1)[0]
                playlist_ids.append(playlist)
                pos_track_ids.append(pos_track)
                neg_track_ids.append(neg_track)

        # Shuffle
        indices = np.random.permutation(len(playlist_ids))
        playlist_ids = np.array(playlist_ids)[indices]
        pos_track_ids = np.array(pos_track_ids)[indices]
        neg_track_ids = np.array(neg_track_ids)[indices]

        total_loss = 0.0
        n_batches = 0

        # Mini-batch training
        for i in tqdm(range(0, len(playlist_ids), batch_size), desc=f"Epoch {ep}/{epochs}", leave=False):
            batch_playlists = torch.LongTensor(playlist_ids[i:i+batch_size]).to(device)
            batch_pos_tracks = torch.LongTensor(pos_track_ids[i:i+batch_size]).to(device)
            batch_neg_tracks = torch.LongTensor(neg_track_ids[i:i+batch_size]).to(device)

            # Get embeddings
            p_emb, pos_t_emb, neg_t_emb, p_ego, pos_t_ego, neg_t_ego = model.get_embedding(
                batch_playlists, batch_pos_tracks, batch_neg_tracks
            )

            # Dot product BPR
            pos_scores = (p_emb * pos_t_emb).sum(dim=1)
            neg_scores = (p_emb * neg_t_emb).sum(dim=1)
            loss_bpr = bpr_loss(pos_scores, neg_scores)

            # L2 regularization on ego embeddings
            reg_loss = (
                p_ego.norm(2).pow(2) +
                pos_t_ego.norm(2).pow(2) +
                neg_t_ego.norm(2).pow(2)
            ) / float(batch_playlists.size(0)) / 2.0

            loss = loss_bpr + float(reg_weight) * reg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Epoch {ep}] loss={avg_loss:.4f}")

        # Evaluation
        if ep % eval_every == 0 or ep == epochs:
            metrics_dot = evaluate_retrieval(
                model, train_dict, test_dict, num_items, device, k_list=k_values, use_cosine=False
            )
            metrics_cos = evaluate_retrieval(
                model, train_dict, test_dict, num_items, device, k_list=k_values, use_cosine=True
            )
            print("Eval (dot):", {k: round(v, 4) for k, v in metrics_dot.items()})
            print("Eval (cos):", {k: round(v, 4) for k, v in metrics_cos.items()})

            record = {"epoch": ep, "loss": avg_loss}
            for k, v in metrics_dot.items():
                record[f"{k}_dot"] = v
            for k, v in metrics_cos.items():
                record[f"{k}_cos"] = v
            metrics_history.append(record)

            # Early stopping check (based on recall@10 dot)
            current_recall = metrics_dot.get("recall@10", 0.0)
            if current_recall > best_recall:
                best_recall = current_recall
                best_epoch = ep
                evals_without_improvement = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ✨ New best Recall@10: {best_recall:.4f}")
            else:
                evals_without_improvement += 1
                print(f"  No improvement for {evals_without_improvement} eval(s) (best: {best_recall:.4f} at epoch {best_epoch})")

            if evals_without_improvement >= patience_evals:
                print(f"\n⏹️ Early stopping! No improvement for {patience_epochs} epochs.")
                print(f"   Best Recall@10: {best_recall:.4f} at epoch {best_epoch}")
                break

    # Restore best model if early stopped
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n🔄 Restored best model from epoch {best_epoch}")

    # Save model, embeddings, and metrics
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{output_prefix}.pt")
    np.save(f"{output_prefix}_metrics.npy", metrics_history, allow_pickle=True)

    # Extract and save embeddings
    with torch.no_grad():
        playlist_emb, track_emb = model()
        playlist_embeddings = playlist_emb.cpu().numpy()
        track_embeddings = track_emb.cpu().numpy()

    np.save(f"{output_prefix}_track_embeddings.npy", track_embeddings)
    track_id_list = [idx_to_track[i] for i in range(num_items)]
    np.save(f"{output_prefix}_track_ids.npy", track_id_list)

    np.save(f"{output_prefix}_playlist_embeddings.npy", playlist_embeddings)
    playlist_id_list = [idx_to_playlist[i] for i in range(num_users)]
    np.save(f"{output_prefix}_playlist_ids.npy", playlist_id_list)

    print(f"\n✅ Saved model to {output_prefix}.pt")
    print(f"✅ Saved track embeddings to {output_prefix}_track_embeddings.npy")
    print(f"✅ Saved track IDs to {output_prefix}_track_ids.npy")
    print(f"✅ Saved playlist embeddings to {output_prefix}_playlist_embeddings.npy")
    print(f"✅ Saved playlist IDs to {output_prefix}_playlist_ids.npy")
    print(f"✅ Saved metrics to {output_prefix}_metrics.npy")


if __name__ == "__main__":
    main()
