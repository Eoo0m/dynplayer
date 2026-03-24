"""
Compare Contrastive Learning (basic) vs LightGCN on two datasets
- Dataset 1: track_playlist_counts_min2_win10.csv
- Dataset 2: track_playlist_counts_min5_win10.csv
- Model 1: Contrastive Learning (uniformity_weight=0.0)
- Model 2: LightGCN

Total: 4 experiments (2 datasets × 2 models)
Results saved to JSON
"""

import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
DATASETS = [
    {
        "name": "min2_win10",
        "csv_path": "/Users/eomjoonseo/dynplayer_crawler/preprocessing/track_playlist_counts_min2_win10.csv",
    },
    {
        "name": "min5_win10",
        "csv_path": "/Users/eomjoonseo/dynplayer_crawler/preprocessing/track_playlist_counts_min5_win10.csv",
    },
]

MODELS = [
    {
        "name": "contrastive_basic",
        "type": "contrastive",
    },
    {
        "name": "lightgcn",
        "type": "lightgcn",
    },
]

# Hyperparameters
EPOCHS = 100
EMBEDDING_DIM = 256
EVAL_EVERY = 10
SEED = 42

# Output
OUTPUT_DIR = Path("/Users/eomjoonseo/dynplayer/comparison_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_contrastive_data(csv_path, output_dir):
    """
    Prepare train/test split for contrastive learning from CSV

    Mirrors the logic from make_train_test_split.py
    CSV format: track_id,playlist_id,count
    """
    print(f"\n📊 Preparing contrastive data from {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    # Preserve CSV order for track indices (matches original behaviour)
    all_track_ids = df["track_id"].drop_duplicates().tolist()
    tid2idx = {tid: i for i, tid in enumerate(all_track_ids)}

    # Build playlist -> tracks mapping
    from collections import defaultdict
    playlist_tracks = defaultdict(set)
    for _, row in df.iterrows():
        playlist_tracks[row["playlist_id"]].add(row["track_id"])

    print(f"  Playlists: {len(playlist_tracks):,}")
    print(f"  Tracks: {len(all_track_ids):,}")

    # Split playlists (matching original logic)
    import random
    train_dict = {}
    test_dict = {}
    playlist_ids = sorted(playlist_tracks.keys())

    for pid_idx, pid in enumerate(playlist_ids):
        tracks = list(playlist_tracks[pid])

        if len(tracks) < 2:
            continue

        # Use deterministic per-playlist shuffling
        rng = random.Random(SEED + pid_idx)
        rng.shuffle(tracks)

        # 20% test, 80% train (matching original test_ratio=0.2)
        split_idx = int(len(tracks) * 0.2)
        test_tracks = tracks[:split_idx]
        train_tracks = tracks[split_idx:]

        if len(train_tracks) > 0 and len(test_tracks) > 0:
            train_dict[pid_idx] = [tid2idx[t] for t in train_tracks]
            test_dict[pid_idx] = [tid2idx[t] for t in test_tracks]

    # Save split (matching original format)
    split_file = output_dir / "train_test_split_unique.npz"
    np.savez(
        split_file,
        train_dict=train_dict,
        test_dict=test_dict,
        all_track_ids=np.array(all_track_ids),
        playlist_ids=np.array(playlist_ids),
    )

    print(f"  ✅ Saved to {split_file}")
    print(f"  Train playlists: {len(train_dict):,}")
    print(f"  Test playlists: {len(test_dict):,}")

    return split_file


def prepare_lightgcn_data(csv_path, output_dir):
    """
    Prepare bipartite graph for LightGCN from CSV

    Mirrors the logic from prepare_graph.py
    CSV format: track_id,playlist_id,count
    """
    print(f"\n📊 Preparing LightGCN data from {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    # Build playlist -> tracks mapping
    from collections import defaultdict
    playlist_tracks = defaultdict(list)
    for _, row in df.iterrows():
        playlist_tracks[row["playlist_id"]].append(row["track_id"])

    print(f"  Unique playlists: {len(playlist_tracks):,}")

    # Create train/test split (matching original: 80/20 with seed=42 per playlist)
    train_pairs = []
    test_pairs = []
    all_track_ids_set = set()

    for playlist_id, track_ids in playlist_tracks.items():
        # Original uses np.random.seed(42) INSIDE the loop (same shuffle for all playlists!)
        np.random.seed(42)
        track_ids = list(track_ids)
        np.random.shuffle(track_ids)

        # 80/20 split
        split_idx = int(len(track_ids) * 0.8)
        train_tracks = track_ids[:split_idx]
        test_tracks = track_ids[split_idx:]

        for track_id in train_tracks:
            train_pairs.append((playlist_id, track_id))
            all_track_ids_set.add(track_id)

        for track_id in test_tracks:
            test_pairs.append((playlist_id, track_id))
            all_track_ids_set.add(track_id)

    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Test pairs: {len(test_pairs):,}")
    print(f"  Total unique tracks: {len(all_track_ids_set):,}")

    # Build mappings
    unique_tracks = sorted(list(all_track_ids_set))
    unique_playlists = sorted(list(playlist_tracks.keys()))

    track_to_idx = {tid: i for i, tid in enumerate(unique_tracks)}
    idx_to_track = {i: tid for tid, i in track_to_idx.items()}

    playlist_to_idx = {pid: i for i, pid in enumerate(unique_playlists)}
    idx_to_playlist = {i: pid for pid, i in playlist_to_idx.items()}

    num_playlists = len(unique_playlists)
    num_tracks = len(unique_tracks)

    print(f"  Number of playlists: {num_playlists:,}")
    print(f"  Number of tracks: {num_tracks:,}")

    # Build bipartite edges from train pairs only
    edges = []
    for playlist_id, track_id in train_pairs:
        playlist_idx = playlist_to_idx[playlist_id]
        track_idx = track_to_idx[track_id]

        # Node index space: [0, num_playlists) for playlists, [num_playlists, num_playlists+num_tracks) for tracks
        playlist_node = playlist_idx
        track_node = num_playlists + track_idx

        edges.append((playlist_node, track_node))
        edges.append((track_node, playlist_node))

    edge_index = np.array(edges, dtype=np.int64).T  # (2, num_edges)

    print(f"  Total edges (bidirectional): {edge_index.shape[1]:,}")

    # Build train/test dictionaries with new indices
    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for playlist_id, track_id in train_pairs:
        playlist_idx = playlist_to_idx[playlist_id]
        track_idx = track_to_idx[track_id]
        train_dict[playlist_idx].append(track_idx)

    for playlist_id, track_id in test_pairs:
        playlist_idx = playlist_to_idx[playlist_id]
        track_idx = track_to_idx[track_id]
        test_dict[playlist_idx].append(track_idx)

    train_dict = dict(train_dict)
    test_dict = dict(test_dict)

    # Save all files
    import pickle
    np.save(output_dir / "edge_index.npy", edge_index)

    with open(output_dir / "track_to_idx.pkl", "wb") as f:
        pickle.dump(track_to_idx, f)
    with open(output_dir / "idx_to_track.pkl", "wb") as f:
        pickle.dump(idx_to_track, f)
    with open(output_dir / "playlist_to_idx.pkl", "wb") as f:
        pickle.dump(playlist_to_idx, f)
    with open(output_dir / "idx_to_playlist.pkl", "wb") as f:
        pickle.dump(idx_to_playlist, f)

    np.savez(
        output_dir / "train_test_split.npz",
        train_dict=train_dict,
        test_dict=test_dict,
        num_playlists=num_playlists,
        num_tracks=num_tracks,
    )

    print(f"  ✅ Saved to {output_dir}")
    print(f"  Train playlists: {len(train_dict):,}")
    print(f"  Test playlists: {len(test_dict):,}")

    return output_dir


def run_contrastive_experiment(dataset_name, data_dir):
    """Run contrastive learning experiment"""
    print(f"\n🚀 Running Contrastive Learning on {dataset_name}")

    save_prefix = OUTPUT_DIR / f"{dataset_name}_contrastive"
    split_file = data_dir / "train_test_split_unique.npz"

    cmd = [
        "python",
        "contrastive_learning new/train_with_split.py",
        "--split-file", str(split_file),
        "--dim", str(EMBEDDING_DIM),
        "--epochs", str(EPOCHS),
        "--eval-every", str(EVAL_EVERY),
        "--seed", str(SEED),
        "--save-prefix", str(save_prefix),
        "--uniformity-weight", "0.0",  # Basic version (no uniformity loss)
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  ❌ Failed")
        return None

    print(f"  ✅ Completed")

    # Load metrics
    metrics_file = f"{save_prefix}_metrics.npy"
    metrics = np.load(metrics_file, allow_pickle=True)

    # Find best epoch (by NDCG@10)
    best_epoch = max(metrics, key=lambda x: x.get("ndcg@10", 0))

    return {
        "dataset": dataset_name,
        "model": "contrastive_basic",
        "best_epoch": best_epoch["epoch"],
        "best_metrics": {
            "ndcg@10": float(best_epoch.get("ndcg@10", 0)),
            "ndcg@20": float(best_epoch.get("ndcg@20", 0)),
            "recall@10": float(best_epoch.get("recall@10", 0)),
            "recall@20": float(best_epoch.get("recall@20", 0)),
        },
        "all_epochs": [
            {
                "epoch": int(m["epoch"]),
                "ndcg@10": float(m.get("ndcg@10", 0)),
                "ndcg@20": float(m.get("ndcg@20", 0)),
                "recall@10": float(m.get("recall@10", 0)),
                "recall@20": float(m.get("recall@20", 0)),
            }
            for m in metrics
        ],
    }


def run_lightgcn_experiment(dataset_name, data_dir):
    """Run LightGCN experiment"""
    print(f"\n🚀 Running LightGCN on {dataset_name}")

    output_prefix = OUTPUT_DIR / f"{dataset_name}_lightgcn"

    cmd = [
        "python",
        "lightgcn/train.py",
        "--graph-dir", str(data_dir),
        "--embedding-dim", str(EMBEDDING_DIM),
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
        "--output-prefix", str(output_prefix),
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  ❌ Failed")
        return None

    print(f"  ✅ Completed")

    # Load metrics
    metrics_file = f"{output_prefix}_metrics.npy"
    metrics = np.load(metrics_file, allow_pickle=True)

    # Find best epoch (by NDCG@10)
    best_epoch = max(metrics, key=lambda x: x.get("ndcg@10", 0))

    return {
        "dataset": dataset_name,
        "model": "lightgcn",
        "best_epoch": best_epoch["epoch"],
        "best_metrics": {
            "ndcg@10": float(best_epoch.get("ndcg@10", 0)),
            "ndcg@20": float(best_epoch.get("ndcg@20", 0)),
            "recall@10": float(best_epoch.get("recall@10", 0)),
            "recall@20": float(best_epoch.get("recall@20", 0)),
        },
        "all_epochs": [
            {
                "epoch": int(m["epoch"]),
                "ndcg@10": float(m.get("ndcg@10", 0)),
                "ndcg@20": float(m.get("ndcg@20", 0)),
                "recall@10": float(m.get("recall@10", 0)),
                "recall@20": float(m.get("recall@20", 0)),
            }
            for m in metrics
        ],
    }


def main():
    print("=" * 80)
    print("🎯 Model Comparison: Contrastive (basic) vs LightGCN")
    print("=" * 80)

    results = []

    for dataset in DATASETS:
        dataset_name = dataset["name"]
        csv_path = dataset["csv_path"]

        print(f"\n{'=' * 80}")
        print(f"📦 Dataset: {dataset_name}")
        print(f"{'=' * 80}")

        # Prepare data directories
        contrastive_data_dir = OUTPUT_DIR / f"{dataset_name}_contrastive_data"
        lightgcn_data_dir = OUTPUT_DIR / f"{dataset_name}_lightgcn_data"

        contrastive_data_dir.mkdir(exist_ok=True)
        lightgcn_data_dir.mkdir(exist_ok=True)

        # Prepare data
        prepare_contrastive_data(csv_path, contrastive_data_dir)
        prepare_lightgcn_data(csv_path, lightgcn_data_dir)

        # Run experiments
        contrastive_result = run_contrastive_experiment(dataset_name, contrastive_data_dir)
        if contrastive_result:
            results.append(contrastive_result)

        lightgcn_result = run_lightgcn_experiment(dataset_name, lightgcn_data_dir)
        if lightgcn_result:
            results.append(lightgcn_result)

    # Save results
    output_file = OUTPUT_DIR / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ Results saved to {output_file}")
    print(f"{'=' * 80}")

    # Print summary
    print("\n📊 Summary:")
    for result in results:
        print(f"\n{result['dataset']} - {result['model']}:")
        if "best_metrics" in result:
            print(f"  Best epoch: {result['best_epoch']}")
            for k, v in result["best_metrics"].items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
