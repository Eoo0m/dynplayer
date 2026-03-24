"""
Prepare LOO (Leave-One-Out) split for LightGCN

For each playlist, leave one track out for testing, rest for training.
Builds bipartite graph from training data only.
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def prepare_loo_split(df, min_tracks_per_playlist=2, seed=42):
    """
    Create LOO split: for each playlist, hold out 1 track for test

    Args:
        df: DataFrame with columns ['playlist', 'track', 'count']
        min_tracks_per_playlist: minimum tracks to keep playlist
        seed: random seed

    Returns:
        train_dict: {playlist_idx: [track_idx]}
        test_dict: {playlist_idx: [track_idx]}
        playlist_to_idx: {playlist_id: idx}
        track_to_idx: {track_id: idx}
    """
    set_seed(seed)

    # Filter playlists with at least min_tracks
    playlist_counts = df.groupby('playlist_id').size()
    valid_playlists = playlist_counts[playlist_counts >= min_tracks_per_playlist].index
    df = df[df['playlist_id'].isin(valid_playlists)]

    print(f"Valid playlists: {len(valid_playlists):,}")
    print(f"Total interactions: {len(df):,}")

    # Create mappings
    playlists = sorted(df['playlist_id'].unique())
    tracks = sorted(df['track_id'].unique())

    playlist_to_idx = {p: i for i, p in enumerate(playlists)}
    track_to_idx = {t: i for i, t in enumerate(tracks)}

    # Group by playlist
    playlist_groups = df.groupby('playlist_id')['track_id'].apply(list).to_dict()

    # LOO split
    train_dict = {}
    test_dict = {}

    for playlist, tracks_list in playlist_groups.items():
        playlist_idx = playlist_to_idx[playlist]

        # Shuffle tracks for randomness
        tracks_list = list(tracks_list)
        random.shuffle(tracks_list)

        # Leave one out
        test_track = tracks_list[0]
        train_tracks = tracks_list[1:]

        # Convert to indices
        train_dict[playlist_idx] = [track_to_idx[t] for t in train_tracks]
        test_dict[playlist_idx] = [track_to_idx[test_track]]

    print(f"Train playlists: {len(train_dict):,}")
    print(f"Test playlists: {len(test_dict):,}")
    print(f"Total tracks: {len(track_to_idx):,}")

    # Statistics
    train_lengths = [len(v) for v in train_dict.values()]
    print(f"Train tracks per playlist: min={min(train_lengths)}, max={max(train_lengths)}, mean={np.mean(train_lengths):.1f}")

    return train_dict, test_dict, playlist_to_idx, track_to_idx


def build_bipartite_graph(train_dict, num_users, num_items):
    """
    Build normalized bipartite graph from training data only

    Args:
        train_dict: {playlist_idx: [track_idx]}
        num_users: number of playlists
        num_items: number of tracks

    Returns:
        graph: scipy sparse matrix [num_users + num_items, num_users + num_items]
    """
    # Build adjacency matrix (users x items) - more memory efficient
    rows = []
    cols = []

    for user, items in train_dict.items():
        for item in items:
            rows.append(user)
            cols.append(item)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)

    # User-Item matrix
    R = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(num_users, num_items),
        dtype=np.float32
    )

    print(f"Interactions in training graph: {R.nnz:,}")

    # Build bipartite adjacency directly in COO format (memory efficient)
    # A = [[0, R], [R.T, 0]]
    # Top block: user -> item edges
    row1 = rows
    col1 = cols + num_users

    # Bottom block: item -> user edges
    row2 = cols + num_users
    col2 = rows

    # Combine
    all_rows = np.concatenate([row1, row2])
    all_cols = np.concatenate([col1, col2])
    all_data = np.concatenate([data, data])

    adj_mat = sp.coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(num_users + num_items, num_users + num_items),
        dtype=np.float32
    )

    # Compute normalization D^{-1/2} A D^{-1/2}
    adj_mat = adj_mat.tocsr()  # Convert for efficient row sum
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # Apply normalization: D^{-1/2} @ A @ D^{-1/2}
    # This is equivalent to multiplying each edge by d_inv_sqrt[row] * d_inv_sqrt[col]
    adj_mat = adj_mat.tocoo()
    adj_mat.data = adj_mat.data * d_inv_sqrt[adj_mat.row] * d_inv_sqrt[adj_mat.col]

    print(f"Normalized graph: {adj_mat.shape}, nnz={adj_mat.nnz:,}")

    return adj_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["min2_win10", "min5_win10"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    # Load config for dataset info (find config.yaml relative to script)
    import yaml
    if args.config:
        config_path = args.config
    else:
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_config = config["data"]["datasets"][args.dataset]
    csv_dir = config["data"]["csv_dir"]
    csv_file = dataset_config["csv_file"]

    # Make CSV path absolute based on config.yaml location
    config_dir = Path(config_path).parent
    csv_path = config_dir / csv_dir / csv_file

    print(f"=== Preparing LOO Split for LightGCN ({args.dataset}) ===")
    print(f"CSV: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows (tracks)")

    # Expand playlist_id column (split by |)
    df['playlist_id'] = df['playlist_id'].str.split('|')
    df = df.explode('playlist_id')
    print(f"Expanded to {len(df):,} track-playlist pairs")

    # Create LOO split
    train_dict, test_dict, playlist_to_idx, track_to_idx = prepare_loo_split(
        df, min_tracks_per_playlist=2, seed=args.seed
    )

    # Build bipartite graph from training data
    graph = build_bipartite_graph(train_dict, len(playlist_to_idx), len(track_to_idx))

    # Save (use script location for relative paths)
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "loo_split.npz",
        train_dict=train_dict,
        test_dict=test_dict,
        playlist_to_idx=playlist_to_idx,
        track_to_idx=track_to_idx,
        num_users=len(playlist_to_idx),
        num_items=len(track_to_idx),
    )

    sp.save_npz(output_dir / "train_graph.npz", graph)

    print(f"\n✅ Saved to {output_dir}/")
    print(f"  - loo_split.npz (train/test split)")
    print(f"  - train_graph.npz (bipartite graph)")


if __name__ == "__main__":
    main()
