"""
Evaluate models by track popularity.

Split test items into popular (top 10%) and unpopular (bottom 10%)
and calculate Recall@K, NDCG@K separately.
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def load_embeddings(model_dir):
    """Load embeddings from model directory."""
    track_embs = np.load(f"{model_dir}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{model_dir}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = [str(t) for t in track_ids]
    track_id_to_idx = {t: i for i, t in enumerate(track_ids)}

    playlist_embs = np.load(f"{model_dir}/model_loo_playlist_embeddings.npy")
    playlist_ids = np.load(f"{model_dir}/model_loo_playlist_ids.npy", allow_pickle=True)
    playlist_ids = [str(p) for p in playlist_ids]
    playlist_id_to_idx = {p: i for i, p in enumerate(playlist_ids)}

    return track_embs, track_id_to_idx, playlist_embs, playlist_id_to_idx


def load_loo_split(data_dir):
    """Load LOO split data."""
    split = np.load(f"{data_dir}/loo_split.npz", allow_pickle=True)
    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()
    idx_to_track = {v: k for k, v in split["track_to_idx"].item().items()}
    idx_to_playlist = {v: k for k, v in split["playlist_to_idx"].item().items()}
    return train_dict, test_dict, idx_to_track, idx_to_playlist


def evaluate_retrieval(
    playlist_embs, track_embs,
    playlist_id_to_idx, track_id_to_idx,
    train_dict, test_dict,
    idx_to_track, idx_to_playlist,
    track_popularity,
    popular_threshold, unpopular_threshold,
    k_list=[10, 20, 50],
    use_cosine=True
):
    """Evaluate retrieval metrics split by popularity."""

    if use_cosine:
        # Normalize embeddings for cosine similarity
        playlist_embs_norm = playlist_embs / (np.linalg.norm(playlist_embs, axis=1, keepdims=True) + 1e-10)
        track_embs_norm = track_embs / (np.linalg.norm(track_embs, axis=1, keepdims=True) + 1e-10)
    else:
        # Use raw embeddings for dot product
        playlist_embs_norm = playlist_embs
        track_embs_norm = track_embs

    results = {
        "all": {"recall": {k: [] for k in k_list}, "ndcg": {k: [] for k in k_list}},
        "popular": {"recall": {k: [] for k in k_list}, "ndcg": {k: [] for k in k_list}},
        "unpopular": {"recall": {k: [] for k in k_list}, "ndcg": {k: [] for k in k_list}},
    }

    n_popular = 0
    n_unpopular = 0
    n_all = 0

    for playlist_idx, test_items in tqdm(test_dict.items(), desc="Evaluating"):
        if playlist_idx not in train_dict or len(test_items) == 0:
            continue

        train_items = train_dict[playlist_idx]
        playlist_id = idx_to_playlist[playlist_idx]

        if playlist_id not in playlist_id_to_idx:
            continue

        # Get playlist embedding
        p_emb = playlist_embs_norm[playlist_id_to_idx[playlist_id]]

        # Compute scores with all tracks
        scores = track_embs_norm @ p_emb

        # Mask train items
        for ti in train_items:
            track_id = idx_to_track[ti]
            if track_id in track_id_to_idx:
                scores[track_id_to_idx[track_id]] = -np.inf

        # Get top-k
        max_k = max(k_list)
        topk_indices = np.argpartition(scores, -max_k)[-max_k:]
        topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]

        # For each test item
        for test_item_idx in test_items:
            test_track_id = idx_to_track[test_item_idx]

            if test_track_id not in track_id_to_idx:
                continue

            # Get popularity of test item
            pop = track_popularity.get(test_track_id, 0)

            # Determine category
            if pop >= popular_threshold:
                category = "popular"
                n_popular += 1
            elif pop <= unpopular_threshold:
                category = "unpopular"
                n_unpopular += 1
            else:
                category = None  # middle, skip for pop/unpop but include in all

            n_all += 1

            # Find rank of test item
            test_emb_idx = track_id_to_idx[test_track_id]

            # Check if in top-k
            for k in k_list:
                topk_k = topk_indices[:k]
                hit = 1 if test_emb_idx in topk_k else 0

                # Recall
                results["all"]["recall"][k].append(hit)
                if category:
                    results[category]["recall"][k].append(hit)

                # NDCG
                if hit:
                    rank = np.where(topk_k == test_emb_idx)[0][0] + 1
                    ndcg = 1 / np.log2(rank + 1)
                else:
                    ndcg = 0

                results["all"]["ndcg"][k].append(ndcg)
                if category:
                    results[category]["ndcg"][k].append(ndcg)

    # Calculate averages
    final_results = {}
    for cat in ["all", "popular", "unpopular"]:
        final_results[cat] = {}
        for metric in ["recall", "ndcg"]:
            final_results[cat][metric] = {}
            for k in k_list:
                vals = results[cat][metric][k]
                final_results[cat][metric][k] = np.mean(vals) if vals else 0.0

    return final_results, n_all, n_popular, n_unpopular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgcn", "simgcl_weighted", "simgcl_randneg", "all"], default="all")
    args = parser.parse_args()

    print("=== Loading Track Popularity ===")
    track_meta_df = pd.read_csv("data/csvs/track_playlist_counts_min5_win10.csv")
    track_meta_df["track_id"] = track_meta_df["track_id"].astype(str)
    track_popularity = dict(zip(track_meta_df["track_id"], track_meta_df["count"]))
    print(f"Tracks: {len(track_popularity):,}")

    # Calculate thresholds (top/bottom 10%)
    pop_values = sorted(track_popularity.values(), reverse=True)
    n = len(pop_values)
    popular_threshold = pop_values[int(n * 0.1)]  # top 10%
    unpopular_threshold = pop_values[int(n * 0.9)]  # bottom 10%
    print(f"Popular threshold (top 10%): >= {popular_threshold}")
    print(f"Unpopular threshold (bottom 10%): <= {unpopular_threshold}")

    models = {
        "lightgcn": "models/lightgcn_loo/outputs/min5_win10",
        "simgcl_weighted": "models/simgcl_weighted/outputs/min5_win10",
        "simgcl_randneg": "models/simgcl_randneg/outputs/min5_win10",
    }

    if args.model == "all":
        model_list = list(models.keys())
    else:
        model_list = [args.model]

    k_list = [10, 20, 50]
    all_results = {}

    for model_name in model_list:
        model_dir = models[model_name]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load embeddings
        track_embs, track_id_to_idx, playlist_embs, playlist_id_to_idx = load_embeddings(model_dir)
        print(f"Track embeddings: {track_embs.shape}")
        print(f"Playlist embeddings: {playlist_embs.shape}")

        # Load LOO split
        train_dict, test_dict, idx_to_track, idx_to_playlist = load_loo_split(model_dir)
        print(f"Test playlists: {len(test_dict):,}")

        # Evaluate with both cosine and dot product
        for sim_type, use_cosine in [("cosine", True), ("dot", False)]:
            results, n_all, n_popular, n_unpopular = evaluate_retrieval(
                playlist_embs, track_embs,
                playlist_id_to_idx, track_id_to_idx,
                train_dict, test_dict,
                idx_to_track, idx_to_playlist,
                track_popularity,
                popular_threshold, unpopular_threshold,
                k_list=k_list,
                use_cosine=use_cosine
            )

            all_results[f"{model_name}_{sim_type}"] = results

            print(f"\n[{sim_type.upper()}] Test items: {n_all:,} (popular: {n_popular:,}, unpopular: {n_unpopular:,})")

            for cat in ["all", "popular", "unpopular"]:
                print(f"  {cat.upper()}:")
                for k in k_list:
                    r = results[cat]["recall"][k]
                    n = results[cat]["ndcg"][k]
                    print(f"    R@{k}: {r:.4f}  NDCG@{k}: {n:.4f}")

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)

    header = f"{'Model':<25} | {'Sim':<6} | {'Category':<10} | " + " | ".join([f"R@{k:<3} NDCG@{k:<3}" for k in k_list])
    print(header)
    print("-"*100)

    for model_name in model_list:
        for sim_type in ["cosine", "dot"]:
            key = f"{model_name}_{sim_type}"
            if key not in all_results:
                continue
            results = all_results[key]
            for cat in ["all", "popular", "unpopular"]:
                row = f"{model_name:<25} | {sim_type:<6} | {cat:<10} | "
                metrics = []
                for k in k_list:
                    r = results[cat]["recall"][k]
                    n = results[cat]["ndcg"][k]
                    metrics.append(f"{r:.3f}  {n:.3f}")
                row += " | ".join(metrics)
                print(row)
        print("-"*100)


if __name__ == "__main__":
    main()
