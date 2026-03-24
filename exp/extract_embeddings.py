"""
Extract track embeddings from trained models

Usage:
    python extract_embeddings.py --model-type contrastive_learning_loo --dataset min5_win10
    python extract_embeddings.py --model-type simgcl_loo --dataset min5_win10
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_contrastive_learning_embeddings(model_path, data_dir, embedding_dim, device):
    """Extract embeddings from contrastive learning model"""
    import sys
    sys.path.insert(0, "contrastive_learning_loo")
    from model import NormalEmbedding

    # Load split to get num_tracks
    split = np.load(data_dir / "loo_split.npz", allow_pickle=True)
    num_items = int(split["num_tracks"])

    # Load model
    model = NormalEmbedding(num_items, embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract embeddings
    with torch.no_grad():
        all_indices = torch.arange(num_items, device=device)
        embeddings = model(all_indices).cpu().numpy()

    return embeddings


def extract_simgcl_embeddings(model_path, data_dir, embedding_dim, n_layers, noise_eps, device):
    """Extract embeddings from SimGCL model"""
    import sys
    sys.path.insert(0, "simgcl")
    from model import SimGCL

    # Load split
    split = np.load(data_dir / "loo_split.npz", allow_pickle=True)
    train_dict = split["train_dict"].item()
    num_users = int(split["num_users"])
    num_items = int(split["num_items"])

    # Build edge index
    edges = []
    for user, items in train_dict.items():
        for item in items:
            user_node = user
            item_node = num_users + item
            edges.append([user_node, item_node])
            edges.append([item_node, user_node])

    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)

    # Load model
    model = SimGCL(
        num_playlists=num_users,
        num_tracks=num_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        noise_eps=noise_eps,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Setup graph for model (SimGCL needs normalized adjacency)
    from scipy.sparse import coo_matrix
    import torch.sparse as sp_torch

    edges_np = edge_index.cpu().numpy()
    num_nodes = num_users + num_items

    # Create adjacency matrix
    data = np.ones(edges_np.shape[1], dtype=np.float32)
    adj = coo_matrix((data, (edges_np[0], edges_np[1])), shape=(num_nodes, num_nodes))

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    from scipy.sparse import diags
    d_mat_inv_sqrt = diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    norm_adj = norm_adj.tocoo()

    # Convert to torch sparse tensor
    indices = torch.LongTensor([norm_adj.row, norm_adj.col]).to(device)
    values = torch.FloatTensor(norm_adj.data).to(device)
    shape = norm_adj.shape
    norm_adj_torch = torch.sparse_coo_tensor(indices, values, shape).to(device)

    model.setup_graph(norm_adj_torch)

    # Extract track embeddings (after propagation)
    with torch.no_grad():
        _, track_embeddings = model()
        track_embeddings = track_embeddings.cpu().numpy()

    return track_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True,
                       choices=["contrastive_learning_loo", "simgcl_loo"])
    parser.add_argument("--dataset", required=True, choices=["min2_win10", "min5_win10"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    embedding_dim = config["training"]["embedding_dim"]

    # Device setup
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Model-specific paths and extraction
    if args.model_type == "contrastive_learning_loo":
        data_dir = Path("contrastive_learning_loo/outputs") / args.dataset
        model_path = data_dir / "model_loo.pt"
        output_path = data_dir / "track_embeddings.npy"

        print(f"Extracting embeddings from {model_path}...")
        embeddings = extract_contrastive_learning_embeddings(
            model_path, data_dir, embedding_dim, device
        )

    elif args.model_type == "simgcl_loo":
        simgcl_config = config["simgcl"]
        n_layers = simgcl_config["n_layers"]
        noise_eps = simgcl_config["noise_eps"]

        data_dir = Path("simgcl_loo/outputs") / args.dataset
        model_path = data_dir / "model_loo.pt"
        output_path = data_dir / "track_embeddings.npy"

        print(f"Extracting embeddings from {model_path}...")
        embeddings = extract_simgcl_embeddings(
            model_path, data_dir, embedding_dim, n_layers, noise_eps, device
        )

    # Save embeddings
    np.save(output_path, embeddings)
    print(f"✅ Saved embeddings: {embeddings.shape} to {output_path}")


if __name__ == "__main__":
    main()
