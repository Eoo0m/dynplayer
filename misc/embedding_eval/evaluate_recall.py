"""
Embedding-based Recall Evaluation with Popularity Analysis

Evaluates recall on LOO test set, separating by track popularity:
- Top 10% popular tracks
- Bottom 10% unpopular (niche) tracks
- Overall

Usage:
    python embedding_eval/evaluate_recall.py \
        --embeddings path/to/track_embeddings.npy \
        --track-ids path/to/track_ids.npy \
        --playlist-embeddings path/to/playlist_embeddings.npy \
        --loo-split path/to/loo_split.npz \
        --k 10 20 50
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def load_embeddings(embeddings_path, track_ids_path, playlist_embeddings_path=None):
    """Load track embeddings, IDs, and optionally playlist embeddings"""
    embeddings = np.load(embeddings_path)
    track_ids = np.load(track_ids_path, allow_pickle=True)

    print(f"Loaded track embeddings: {embeddings.shape}")
    print(f"Loaded track IDs: {len(track_ids)}")

    playlist_embeddings = None
    if playlist_embeddings_path:
        playlist_embeddings = np.load(playlist_embeddings_path)
        print(f"Loaded playlist embeddings: {playlist_embeddings.shape}")

    return embeddings, track_ids, playlist_embeddings


def load_loo_split(split_path):
    """Load LOO split data"""
    split = np.load(split_path, allow_pickle=True)

    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()
    num_users = int(split["num_users"])
    num_items = int(split["num_items"])
    track_to_idx = split["track_to_idx"].item()

    print(f"Users: {num_users:,}, Items: {num_items:,}")
    print(f"Train playlists: {len(train_dict):,}")
    print(f"Test playlists: {len(test_dict):,}")

    return train_dict, test_dict, num_users, num_items, track_to_idx


def compute_track_popularity(train_dict, num_items):
    """
    Compute track popularity from training data

    Returns:
        track_counts: np.array of shape [num_items]
    """
    track_counts = np.zeros(num_items, dtype=np.int32)

    for items in train_dict.values():
        for item in items:
            if item < num_items:
                track_counts[item] += 1

    return track_counts


def get_popularity_groups(test_dict, track_counts, top_percent=10, bottom_percent=10):
    """
    Split test items by popularity

    Args:
        test_dict: {user_idx: [test_item_idx]}
        track_counts: popularity counts for all tracks
        top_percent: top X% popular
        bottom_percent: bottom X% unpopular

    Returns:
        popular_users: set of user indices with popular test items
        unpopular_users: set of user indices with unpopular test items
        all_users: set of all user indices
    """
    # Get test track indices and their popularity
    test_track_indices = []
    test_user_indices = []

    for user, test_items in test_dict.items():
        for item in test_items:
            test_track_indices.append(item)
            test_user_indices.append(user)

    test_track_indices = np.array(test_track_indices)
    test_user_indices = np.array(test_user_indices)
    test_popularities = track_counts[test_track_indices]

    # Sort by popularity
    sorted_indices = np.argsort(test_popularities)
    n_test = len(sorted_indices)

    # Top and bottom percentiles
    top_threshold_idx = int(n_test * (1 - top_percent / 100))
    bottom_threshold_idx = int(n_test * (bottom_percent / 100))

    popular_indices = sorted_indices[top_threshold_idx:]
    unpopular_indices = sorted_indices[:bottom_threshold_idx]

    popular_users = set(test_user_indices[popular_indices])
    unpopular_users = set(test_user_indices[unpopular_indices])
    all_users = set(test_dict.keys())

    # Stats
    popular_counts = test_popularities[popular_indices]
    unpopular_counts = test_popularities[unpopular_indices]

    print(f"\n=== Popularity Groups ===")
    print(f"Top {top_percent}% popular: {len(popular_users):,} users")
    print(f"  Popularity range: {popular_counts.min()} - {popular_counts.max()}, mean={popular_counts.mean():.1f}")
    print(f"Bottom {bottom_percent}% unpopular: {len(unpopular_users):,} users")
    print(f"  Popularity range: {unpopular_counts.min()} - {unpopular_counts.max()}, mean={unpopular_counts.mean():.1f}")
    print(f"All users: {len(all_users):,}")

    return popular_users, unpopular_users, all_users


def evaluate_recall(
    train_dict,
    test_dict,
    track_embeddings,
    user_subset,
    playlist_embeddings=None,
    k_list=[10, 20, 50],
    use_cosine=False,
    desc="Evaluating"
):
    """
    Evaluate recall on a subset of users

    Args:
        train_dict: {user_idx: [train_item_idx]}
        test_dict: {user_idx: [test_item_idx]}
        track_embeddings: [num_items, dim]
        user_subset: set of user indices to evaluate
        playlist_embeddings: [num_users, dim] optional, uses model's playlist embeddings if provided
        k_list: list of K values
        use_cosine: whether to use cosine similarity

    Returns:
        metrics: {recall@k: value, ndcg@k: value}
    """
    # Normalize track embeddings for cosine similarity
    if use_cosine:
        track_norms = np.linalg.norm(track_embeddings, axis=1, keepdims=True) + 1e-10
        track_embeddings_norm = track_embeddings / track_norms
        if playlist_embeddings is not None:
            playlist_norms = np.linalg.norm(playlist_embeddings, axis=1, keepdims=True) + 1e-10
            playlist_embeddings_norm = playlist_embeddings / playlist_norms
        else:
            playlist_embeddings_norm = None
    else:
        track_embeddings_norm = track_embeddings
        playlist_embeddings_norm = playlist_embeddings

    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    for user in tqdm(user_subset, desc=desc, leave=False):
        if user not in train_dict or user not in test_dict:
            continue

        train_items = train_dict[user]
        test_items = test_dict[user]

        if len(train_items) == 0 or len(test_items) == 0:
            continue

        # Get user embedding: use playlist embedding if available, else average of track embeddings
        if playlist_embeddings_norm is not None:
            user_emb = playlist_embeddings_norm[user]
        else:
            user_emb = track_embeddings_norm[train_items].mean(axis=0)
            if use_cosine:
                user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-10)

        # Compute scores
        scores = track_embeddings_norm @ user_emb

        # Mask train items
        train_set = set(train_items)
        for ti in train_set:
            scores[ti] = -np.inf

        # Get top-k
        max_k = max(k_list)
        topk = np.argpartition(scores, -max_k)[-max_k:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        # Compute metrics
        test_set = set(test_items)
        rel = np.array([1 if i in test_set else 0 for i in topk])

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

    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to track embeddings .npy")
    parser.add_argument("--track-ids", required=True, help="Path to track IDs .npy")
    parser.add_argument("--playlist-embeddings", default=None, help="Path to playlist embeddings .npy (optional)")
    parser.add_argument("--loo-split", required=True, help="Path to LOO split .npz")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 50], help="K values for recall")
    parser.add_argument("--top-percent", type=float, default=10, help="Top X% popular")
    parser.add_argument("--bottom-percent", type=float, default=10, help="Bottom X% unpopular")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity instead of dot product")
    args = parser.parse_args()

    print("=== Embedding-based Recall Evaluation ===")
    print(f"Track Embeddings: {args.embeddings}")
    print(f"Playlist Embeddings: {args.playlist_embeddings or 'None (will use track mean)'}")
    print(f"LOO Split: {args.loo_split}")
    print(f"K values: {args.k}")
    print(f"Similarity: {'Cosine' if args.use_cosine else 'Dot product'}")

    # Load data
    embeddings, track_ids, playlist_embeddings = load_embeddings(
        args.embeddings, args.track_ids, args.playlist_embeddings
    )
    train_dict, test_dict, num_users, num_items, track_to_idx = load_loo_split(args.loo_split)

    # Verify alignment
    if len(embeddings) != num_items:
        print(f"Warning: Embedding count ({len(embeddings)}) != num_items ({num_items})")
        print("Attempting to align by track IDs...")

        # Create idx mapping from track_ids
        idx_to_track = {i: t for t, i in track_to_idx.items()}
        track_id_to_emb_idx = {tid: i for i, tid in enumerate(track_ids)}

        # Reindex embeddings
        aligned_embeddings = np.zeros((num_items, embeddings.shape[1]), dtype=embeddings.dtype)
        for idx in range(num_items):
            track_id = idx_to_track.get(idx)
            if track_id and track_id in track_id_to_emb_idx:
                emb_idx = track_id_to_emb_idx[track_id]
                aligned_embeddings[idx] = embeddings[emb_idx]

        embeddings = aligned_embeddings
        print(f"Aligned embeddings: {embeddings.shape}")

    # Compute track popularity
    track_counts = compute_track_popularity(train_dict, num_items)

    # Get popularity groups
    popular_users, unpopular_users, all_users = get_popularity_groups(
        test_dict, track_counts,
        top_percent=args.top_percent,
        bottom_percent=args.bottom_percent
    )

    use_cosine = args.use_cosine

    # Evaluate on all users
    print("\n=== Evaluating All Users ===")
    all_metrics = evaluate_recall(
        train_dict, test_dict, embeddings, all_users,
        playlist_embeddings=playlist_embeddings,
        k_list=args.k, use_cosine=use_cosine, desc="All"
    )

    # Evaluate on popular users
    print("\n=== Evaluating Popular Track Users (Top 10%) ===")
    popular_metrics = evaluate_recall(
        train_dict, test_dict, embeddings, popular_users,
        playlist_embeddings=playlist_embeddings,
        k_list=args.k, use_cosine=use_cosine, desc="Popular"
    )

    # Evaluate on unpopular users
    print("\n=== Evaluating Unpopular Track Users (Bottom 10%) ===")
    unpopular_metrics = evaluate_recall(
        train_dict, test_dict, embeddings, unpopular_users,
        playlist_embeddings=playlist_embeddings,
        k_list=args.k, use_cosine=use_cosine, desc="Unpopular"
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n[All Users]")
    for k in args.k:
        print(f"  Recall@{k}: {all_metrics[f'recall@{k}']:.4f}, NDCG@{k}: {all_metrics[f'ndcg@{k}']:.4f}")

    print(f"\n[Popular Tracks - Top {args.top_percent}%]")
    for k in args.k:
        print(f"  Recall@{k}: {popular_metrics[f'recall@{k}']:.4f}, NDCG@{k}: {popular_metrics[f'ndcg@{k}']:.4f}")

    print(f"\n[Unpopular Tracks - Bottom {args.bottom_percent}%]")
    for k in args.k:
        print(f"  Recall@{k}: {unpopular_metrics[f'recall@{k}']:.4f}, NDCG@{k}: {unpopular_metrics[f'ndcg@{k}']:.4f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY (Recall)")
    print("=" * 60)
    print(f"{'K':<6} {'All':<12} {'Popular':<12} {'Unpopular':<12} {'Gap (P-U)':<12}")
    print("-" * 60)
    for k in args.k:
        all_r = all_metrics[f'recall@{k}']
        pop_r = popular_metrics[f'recall@{k}']
        unpop_r = unpopular_metrics[f'recall@{k}']
        gap = pop_r - unpop_r
        print(f"{k:<6} {all_r:<12.4f} {pop_r:<12.4f} {unpop_r:<12.4f} {gap:<+12.4f}")


if __name__ == "__main__":
    main()
