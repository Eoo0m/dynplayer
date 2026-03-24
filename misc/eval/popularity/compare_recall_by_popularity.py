"""
Compare recall between simgcl_loo and simgcl_weighted
by track popularity (top 10% vs bottom 10%)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_model_data(model_dir):
    """Load embeddings and IDs from model directory"""
    track_embs = np.load(f"{model_dir}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{model_dir}/model_loo_track_ids.npy", allow_pickle=True)
    playlist_embs = np.load(f"{model_dir}/model_loo_playlist_embeddings.npy")
    playlist_ids = np.load(f"{model_dir}/model_loo_playlist_ids.npy", allow_pickle=True)

    # Load split data
    split = np.load(f"{model_dir}/loo_split.npz", allow_pickle=True)
    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()

    return {
        "track_embs": track_embs,
        "track_ids": track_ids,
        "playlist_embs": playlist_embs,
        "playlist_ids": playlist_ids,
        "train_dict": train_dict,
        "test_dict": test_dict,
    }


def compute_recall_by_popularity(model_data, track_popularity, k=10, normalize=True):
    """
    Compute recall@k grouped by test track popularity

    Returns:
        recalls_by_group: dict with 'top10', 'bottom10', 'all' keys
    """
    track_embs = model_data["track_embs"]
    playlist_embs = model_data["playlist_embs"]
    train_dict = model_data["train_dict"]
    test_dict = model_data["test_dict"]
    track_ids = model_data["track_ids"]

    # Normalize embeddings
    if normalize:
        track_embs = track_embs / (np.linalg.norm(track_embs, axis=1, keepdims=True) + 1e-10)
        playlist_embs = playlist_embs / (np.linalg.norm(playlist_embs, axis=1, keepdims=True) + 1e-10)

    # Create track_id -> idx mapping
    track_id_to_idx = {str(tid): i for i, tid in enumerate(track_ids)}

    # Collect test tracks and their popularity
    test_track_info = []  # (playlist_idx, track_idx, popularity)

    for playlist_idx, test_items in test_dict.items():
        if playlist_idx not in train_dict or len(test_items) == 0:
            continue

        for track_idx in test_items:
            track_id = str(track_ids[track_idx])
            if track_id in track_popularity:
                pop = track_popularity[track_id]
                test_track_info.append((playlist_idx, track_idx, pop))

    # Sort by popularity
    test_track_info.sort(key=lambda x: x[2], reverse=True)

    n_total = len(test_track_info)
    n_10pct = n_total // 10

    top10_tracks = set((p, t) for p, t, _ in test_track_info[:n_10pct])
    bottom10_tracks = set((p, t) for p, t, _ in test_track_info[-n_10pct:])

    print(f"  Total test tracks: {n_total}")
    print(f"  Top 10% count: {len(top10_tracks)} (popularity >= {test_track_info[n_10pct-1][2]})")
    print(f"  Bottom 10% count: {len(bottom10_tracks)} (popularity <= {test_track_info[-n_10pct][2]})")

    # Compute recall for each group
    recalls = {"top10": [], "bottom10": [], "all": []}

    for playlist_idx, test_items in tqdm(test_dict.items(), desc="Computing recall"):
        if playlist_idx not in train_dict or len(test_items) == 0:
            continue

        train_items = set(train_dict[playlist_idx])

        # Compute scores
        scores = track_embs @ playlist_embs[playlist_idx]

        # Mask train items
        for ti in train_items:
            scores[ti] = -np.inf

        # Get top-k
        topk = np.argpartition(scores, -k)[-k:]
        topk_set = set(topk)

        # Check each test item
        for track_idx in test_items:
            hit = 1 if track_idx in topk_set else 0
            recalls["all"].append(hit)

            if (playlist_idx, track_idx) in top10_tracks:
                recalls["top10"].append(hit)
            elif (playlist_idx, track_idx) in bottom10_tracks:
                recalls["bottom10"].append(hit)

    return {
        "top10": np.mean(recalls["top10"]) if recalls["top10"] else 0,
        "bottom10": np.mean(recalls["bottom10"]) if recalls["bottom10"] else 0,
        "all": np.mean(recalls["all"]) if recalls["all"] else 0,
        "n_top10": len(recalls["top10"]),
        "n_bottom10": len(recalls["bottom10"]),
        "n_all": len(recalls["all"]),
    }


def main():
    print("=== Loading track popularity ===")
    track_meta_df = pd.read_csv("data/csvs/track_playlist_counts_min5_win10.csv")
    track_meta_df["track_id"] = track_meta_df["track_id"].astype(str)
    track_popularity = dict(zip(track_meta_df["track_id"], track_meta_df["count"]))
    print(f"Tracks with popularity: {len(track_popularity)}")

    models = {
        "simgcl_loo": "simgcl_loo/outputs/min5_win10",
        "simgcl_weighted": "simgcl_weighted/outputs/min5_win10",
        "simgcl_randneg": "simgcl_randneg/outputs/min5_win10",
    }

    results = {}

    for name, path in models.items():
        print(f"\n=== {name} ===")
        model_data = load_model_data(path)
        print(f"Track embeddings: {model_data['track_embs'].shape}")
        print(f"Playlist embeddings: {model_data['playlist_embs'].shape}")

        recalls = compute_recall_by_popularity(model_data, track_popularity, k=10)
        results[name] = recalls

    # Print comparison
    print("\n" + "=" * 80)
    print("=== Recall@10 Comparison (Cosine Similarity) ===")
    print("=" * 80)
    print(f"{'Group':<15} {'simgcl_loo':>15} {'simgcl_weighted':>15} {'simgcl_randneg':>15}")
    print("-" * 80)

    for group in ["all", "top10", "bottom10"]:
        loo = results["simgcl_loo"][group]
        weighted = results["simgcl_weighted"][group]
        randneg = results["simgcl_randneg"][group]
        print(f"{group:<15} {loo:>15.4f} {weighted:>15.4f} {randneg:>15.4f}")

    print("-" * 80)
    print(f"{'Sample sizes':<15} {results['simgcl_loo']['n_all']:>15} {results['simgcl_weighted']['n_all']:>15} {results['simgcl_randneg']['n_all']:>15}")


if __name__ == "__main__":
    main()
