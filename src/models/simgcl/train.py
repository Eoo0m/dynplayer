"""
Train SimGCL with LOO (Leave-One-Out) evaluation

Usage:
    python simgcl_loo/train_loo.py --dataset min5_win10
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import random


class BPRDataset(Dataset):
    """BPR sampling dataset"""
    def __init__(self, train_dict, num_items, samples_per_user=10, seed=42):
        self.train_dict = train_dict
        self.num_items = num_items
        self.samples_per_user = samples_per_user
        self.users = list(train_dict.keys())
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.users) * self.samples_per_user

    def __getitem__(self, idx):
        user = self.users[idx % len(self.users)]
        pos_list = self.train_dict[user]
        pos = self.rng.choice(pos_list)

        # Sample negative
        user_pos_set = set(pos_list)
        neg = self.rng.randint(0, self.num_items - 1)
        tries = 0
        while neg in user_pos_set and tries < 100:
            neg = self.rng.randint(0, self.num_items - 1)
            tries += 1

        return user, pos, neg


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_track_weights(train_dict, num_items):
    """
    Compute log-inverse probability weights for tracks.

    Rare tracks get higher weights, popular tracks get lower weights.
    Weight = -log(prob) where prob = count / total
    Weights are normalized to mean=1 for stable training.

    Args:
        train_dict: {user_id: [item_ids]}
        num_items: total number of items

    Returns:
        weights: torch.Tensor of shape [num_items]
    """
    # Count track occurrences
    track_counts = np.zeros(num_items, dtype=np.float32)
    for items in train_dict.values():
        for item in items:
            if item < num_items:
                track_counts[item] += 1

    # Laplace smoothing (avoid log(0))
    track_counts = track_counts + 1
    total_count = track_counts.sum()

    # Probability
    track_probs = track_counts / total_count

    # Log-inverse weight: -log(prob) = log(1/prob)
    weights = -np.log(track_probs)

    # Normalize to mean=1 for stable training
    weights = weights / weights.mean()

    return torch.from_numpy(weights)


def build_edge_index(train_dict, num_users, num_items):
    """
    Build edge index from training data (same as original SimGCL)

    Returns:
        edge_index: [2, num_edges] tensor
    """
    edges = []

    for user, items in train_dict.items():
        for item in items:
            # Bipartite: user <-> item
            user_node = user
            item_node = num_users + item

            edges.append([user_node, item_node])
            edges.append([item_node, user_node])

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index


def evaluate(model, train_dict, test_dict, num_users, device, k_list=[10, 20], use_cosine=False):
    """Evaluate model on LOO test set

    Args:
        use_cosine: If True, normalize embeddings (cosine similarity)
                   If False, use raw dot product (default SimGCL)
    """
    model.eval()

    with torch.no_grad():
        users_emb, items_emb = model()
        users_emb = users_emb.cpu().numpy()
        items_emb = items_emb.cpu().numpy()

    # Apply L2 normalization if using cosine similarity
    if use_cosine:
        users_emb = users_emb / (np.linalg.norm(users_emb, axis=1, keepdims=True) + 1e-10)
        items_emb = items_emb / (np.linalg.norm(items_emb, axis=1, keepdims=True) + 1e-10)

    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    for user, test_items in tqdm(test_dict.items(), desc="Evaluating", leave=False):
        if user not in train_dict or len(test_items) == 0 or len(train_dict[user]) == 0:
            continue

        train_items = set(train_dict[user])
        scores = items_emb @ users_emb[user]

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

            # NDCG
            dcg = (r / np.log2(np.arange(2, len(r) + 2))).sum()
            n_relevant = min(len(test_items), k)
            idcg = (np.ones(n_relevant) / np.log2(np.arange(2, n_relevant + 2))).sum()
            ndcg = dcg / idcg if idcg > 0 else 0.0

            recalls[k].append(recall)
            ndcgs[k].append(ndcg)

    return {
        f"recall@{k}": float(np.mean(v)) if v else 0.0 for k, v in recalls.items()
    } | {f"ndcg@{k}": float(np.mean(v)) if v else 0.0 for k, v in ndcgs.items()}


def compute_simgcl_loss(model, users, pos, neg, num_users, bpr_loss_fn, infonce_loss_fn, lambda_cl=0.1, tau=0.2, track_weights=None, num_random_neg=0):
    """
    SimGCL loss with graph propagation (original paper implementation)

    Propagates through graph for both BPR and contrastive loss.
    """
    # Get clean propagated embeddings for BPR
    playlist_emb, track_emb = model()

    user_emb = playlist_emb[users]
    pos_emb = track_emb[pos]
    neg_emb = track_emb[neg]

    # Normalize for BPR
    user_emb_norm = F.normalize(user_emb, dim=1)
    pos_emb_norm = F.normalize(pos_emb, dim=1)
    neg_emb_norm = F.normalize(neg_emb, dim=1)

    # BPR loss (with optional weighting)
    pos_scores = (user_emb_norm * pos_emb_norm).sum(dim=1)
    neg_scores = (user_emb_norm * neg_emb_norm).sum(dim=1)

    if track_weights is not None:
        batch_weights = track_weights[pos]
        loss_bpr = bpr_loss_fn(pos_scores, neg_scores, weights=batch_weights)
    else:
        loss_bpr = bpr_loss_fn(pos_scores, neg_scores)

    # Contrastive loss with propagation + noise
    if lambda_cl > 0:
        # Get two augmented views (propagation + noise)
        view1_playlist, view1_track = model.get_augmented_views()
        view2_playlist, view2_track = model.get_augmented_views()

        # Combine user and item embeddings from batch
        batch_indices = torch.cat([
            users,
            num_users + pos,
            num_users + neg
        ])
        batch_indices = torch.unique(batch_indices)

        # Extract batch views
        view1_batch = torch.cat([view1_playlist, view1_track], dim=0)[batch_indices]
        view2_batch = torch.cat([view2_playlist, view2_track], dim=0)[batch_indices]

        # Sample random negatives if requested
        random_neg_emb = None
        if num_random_neg > 0:
            total_nodes = model.num_playlists + model.num_tracks
            random_indices = torch.randint(0, total_nodes, (num_random_neg,), device=users.device)
            all_emb = torch.cat([view2_playlist, view2_track], dim=0)
            random_neg_emb = all_emb[random_indices]

        loss_cl = infonce_loss_fn(view1_batch, view2_batch, tau=tau, random_neg_emb=random_neg_emb)
    else:
        loss_cl = torch.tensor(0.0, device=users.device)

    total_loss = loss_bpr + lambda_cl * loss_cl

    return total_loss, loss_bpr, loss_cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["min2_win10", "min5_win10"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--normalize-embeddings", action="store_true",
                       help="Normalize embeddings to L2 norm=1 during training")
    parser.add_argument("--num-random-neg", type=int, default=256,
                       help="Number of random negative samples for contrastive loss (default: 256)")
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

    simgcl_config = config["simgcl"]
    n_layers = simgcl_config["n_layers"]
    batch_size = simgcl_config["batch_size"]
    lr = simgcl_config["lr"]
    samples_per_user = simgcl_config["samples_per_user"]
    noise_eps = simgcl_config["noise_eps"]
    lambda_cl = simgcl_config["lambda_cl"]
    tau = simgcl_config["tau"]

    # Use script location for relative paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "outputs" / args.dataset
    output_prefix = data_dir / "model_loo"

    print(f"=== SimGCL LOO Training ===")
    print(f"Dataset: {args.dataset} ({dataset_config['description']})")
    print(f"Data dir: {data_dir}")
    print(f"Embedding dim: {embedding_dim}, Layers: {n_layers}, Epochs: {epochs}")
    print(f"Loss mode: Full propagation (original SimGCL)")
    print(f"Normalize embeddings: {args.normalize_embeddings}")
    print(f"Random negatives for CL: {args.num_random_neg}")

    set_seed(seed)

    # Device setup (MPS doesn't support sparse tensors)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "mps":
        print("Warning: MPS does not support sparse tensors. Falling back to CPU.")
        device = torch.device("cpu")

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

    # Build edge index (same as original SimGCL)
    edge_index = build_edge_index(train_dict, num_users, num_items)
    edge_index = edge_index.to(device)

    print(f"Users: {num_users:,}, Items: {num_items:,}")
    print(f"Edges: {edge_index.size(1):,}")

    # Compute track weights for weighted BPR
    track_weights = compute_track_weights(train_dict, num_items).to(device)
    print(f"Track weights: min={track_weights.min().item():.4f}, max={track_weights.max().item():.4f}, mean={track_weights.mean().item():.4f}")

    # Import model and loss functions from local model.py
    import sys
    sys.path.insert(0, str(script_dir))
    from model import SimGCL, bpr_loss, infonce_loss

    # Build normalized adjacency matrix (required by SimGCL)
    import scipy.sparse as sp
    num_nodes = num_users + num_items
    edges_np = edge_index.cpu().numpy()

    # Create adjacency matrix
    data = np.ones(edges_np.shape[1], dtype=np.float32)
    adj = sp.coo_matrix((data, (edges_np[0], edges_np[1])), shape=(num_nodes, num_nodes))

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    norm_adj = norm_adj.tocoo()

    # Convert to torch sparse tensor
    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col])).to(device)
    values = torch.FloatTensor(norm_adj.data).to(device)
    shape = norm_adj.shape
    norm_adj_torch = torch.sparse_coo_tensor(indices, values, shape).to(device)

    # Build model
    model = SimGCL(
        num_playlists=num_users,
        num_tracks=num_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        noise_eps=noise_eps,
        normalize_embeddings=args.normalize_embeddings,
    ).to(device)

    # Setup graph in model
    model.setup_graph(norm_adj_torch)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset and optimizer
    dataset = BPRDataset(train_dict, num_items, samples_per_user, seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics_history = []

    # Training loop
    for ep in range(1, epochs + 1):
        model.train()

        total_bpr = 0.0
        total_cl = 0.0
        n_batches = 0

        for users, pos, neg in tqdm(loader, desc=f"Epoch {ep}/{epochs}", leave=False):
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # SimGCL loss with full propagation (original paper)
            loss, loss_bpr, loss_cl = compute_simgcl_loss(
                model, users, pos, neg, num_users, bpr_loss, infonce_loss, lambda_cl=lambda_cl, tau=tau,
                track_weights=track_weights, num_random_neg=args.num_random_neg
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_bpr += loss_bpr.item()
            total_cl += loss_cl.item()
            n_batches += 1

        avg_bpr = total_bpr / n_batches
        avg_cl = total_cl / n_batches
        print(f"[Epoch {ep}] bpr={avg_bpr:.4f}, cl={avg_cl:.4f}")

        # Evaluation
        if ep % eval_every == 0 or ep == epochs:
            # Evaluate with both dot product and cosine similarity
            metrics_dot = evaluate(model, train_dict, test_dict, num_users, device, k_list=k_values, use_cosine=False)
            metrics_cos = evaluate(model, train_dict, test_dict, num_users, device, k_list=k_values, use_cosine=True)

            print("Eval (dot):", {k: round(v, 4) for k, v in metrics_dot.items()})
            print("Eval (cos):", {k: round(v, 4) for k, v in metrics_cos.items()})

            record = {"epoch": ep, "bpr": avg_bpr, "cl": avg_cl}
            # Add both dot and cosine metrics
            for k, v in metrics_dot.items():
                record[f"{k}_dot"] = v
            for k, v in metrics_cos.items():
                record[f"{k}_cos"] = v
            metrics_history.append(record)

    # Save model, embeddings, and metrics
    torch.save(model.state_dict(), f"{output_prefix}.pt")
    np.save(f"{output_prefix}_metrics.npy", metrics_history, allow_pickle=True)

    # Extract and save embeddings (after propagation)
    with torch.no_grad():
        playlist_embeddings, track_embeddings = model()
        playlist_embeddings = playlist_embeddings.cpu().numpy()
        track_embeddings = track_embeddings.cpu().numpy()

    # Save track embeddings and IDs
    np.save(f"{output_prefix}_track_embeddings.npy", track_embeddings)
    track_id_list = [idx_to_track[i] for i in range(num_items)]
    np.save(f"{output_prefix}_track_ids.npy", track_id_list)

    # Save playlist embeddings and IDs
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
