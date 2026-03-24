"""
Analyze recall by track popularity for Dot Product models (DOT ONLY).

Computes:
- Overall recall (using DOT product)
- Recall for popular tracks (top 20% most frequent)
- Recall for unpopular tracks (bottom 20% least frequent)

Usage:
    python scripts/analyze_popularity_recall_dot.py --dataset min5_win10
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def load_model_embeddings(model_dir):
    """Load track and playlist embeddings from model directory"""
    track_emb = np.load(model_dir / "model_loo_track_embeddings.npy")
    playlist_emb = np.load(model_dir / "model_loo_playlist_embeddings.npy")
    track_ids = np.load(model_dir / "model_loo_track_ids.npy", allow_pickle=True)
    playlist_ids = np.load(model_dir / "model_loo_playlist_ids.npy", allow_pickle=True)

    return track_emb, playlist_emb, track_ids, playlist_ids


def compute_track_popularity(train_dict, num_items):
    """
    Compute track popularity (occurrence count) from training data.

    Returns:
        popularity: np.array of shape [num_items] with occurrence counts
    """
    popularity = np.zeros(num_items, dtype=np.int32)
    for items in train_dict.values():
        for item in items:
            if item < num_items:
                popularity[item] += 1
    return popularity


def evaluate_by_popularity_dot(track_emb, playlist_emb, train_dict, test_dict,
                                track_popularity, k=10):
    """
    Evaluate recall separately for popular and unpopular tracks.
    Uses DOT PRODUCT similarity only.

    Args:
        track_emb: [num_items, dim] track embeddings
        playlist_emb: [num_users, dim] playlist embeddings
        train_dict: {user_id: [item_ids]} training data
        test_dict: {user_id: [item_ids]} test data (LOO)
        track_popularity: [num_items] popularity counts
        k: top-k for recall

    Returns:
        dict with overall, popular, and unpopular recall metrics
    """
    # Define popularity thresholds (top 20% and bottom 20%)
    popularity_percentiles = np.percentile(track_popularity[track_popularity > 0], [20, 80])
    unpopular_threshold = popularity_percentiles[0]
    popular_threshold = popularity_percentiles[1]

    print(f"  Popularity thresholds: unpopular <= {unpopular_threshold:.1f}, popular >= {popular_threshold:.1f}")

    # Track metrics
    recalls_overall = []
    recalls_popular = []
    recalls_unpopular = []

    n_popular = 0
    n_unpopular = 0
    n_total = 0

    for user, test_items in tqdm(test_dict.items(), desc="  Evaluating", leave=False):
        if user not in train_dict or len(test_items) == 0 or len(train_dict[user]) == 0:
            continue

        train_items = set(train_dict[user])

        # DOT PRODUCT similarity (no normalization)
        scores = track_emb @ playlist_emb[user]

        # Mask train items
        for ti in train_items:
            scores[ti] = -np.inf

        # Get top-k
        topk = np.argpartition(scores, -k)[-k:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        # Compute recall
        hits = len(set(topk) & set(test_items))
        recall = hits / len(test_items)
        recalls_overall.append(recall)
        n_total += 1

        # Check if test items are popular or unpopular
        test_item = test_items[0]  # LOO: single test item
        test_popularity = track_popularity[test_item]

        if test_popularity >= popular_threshold:
            recalls_popular.append(recall)
            n_popular += 1
        elif test_popularity <= unpopular_threshold:
            recalls_unpopular.append(recall)
            n_unpopular += 1

    return {
        "recall_overall": float(np.mean(recalls_overall)) if recalls_overall else 0.0,
        "recall_popular": float(np.mean(recalls_popular)) if recalls_popular else 0.0,
        "recall_unpopular": float(np.mean(recalls_unpopular)) if recalls_unpopular else 0.0,
        "n_total": n_total,
        "n_popular": n_popular,
        "n_unpopular": n_unpopular,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="min5_win10", choices=["min2_win10", "min5_win10"])
    parser.add_argument("--k", type=int, default=10, help="Top-k for recall")
    args = parser.parse_args()

    # Model directories (DOT product models only)
    base_dir = Path(__file__).parent.parent / "outputs" / "trained_models"
    models = {
        "lightgcn_dot": base_dir / "lightgcn_dot" / "outputs" / args.dataset,
        "simgcl_dot": base_dir / "simgcl_dot" / "outputs" / args.dataset,
    }

    print(f"\n=== Popularity-based Recall Analysis (DOT PRODUCT ONLY) ===")
    print(f"Dataset: {args.dataset}, k={args.k}")

    # Load LOO split (from any model dir, they all use the same split)
    first_model_dir = list(models.values())[0]
    split = np.load(first_model_dir / "loo_split.npz", allow_pickle=True)
    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()
    num_users = int(split["num_users"])
    num_items = int(split["num_items"])

    print(f"Users: {num_users:,}, Items: {num_items:,}")

    # Compute track popularity from training data
    print("\nComputing track popularity...")
    track_popularity = compute_track_popularity(train_dict, num_items)
    print(f"Popularity stats: min={track_popularity.min()}, max={track_popularity.max()}, "
          f"mean={track_popularity.mean():.1f}, median={np.median(track_popularity):.1f}")

    # Analyze each model
    results = {}

    for model_name, model_dir in models.items():
        if not model_dir.exists():
            print(f"\nSkipping {model_name} (directory not found)")
            continue

        print(f"\n{model_name.upper()}")
        print("=" * 60)

        # Load embeddings
        track_emb, playlist_emb, track_ids, playlist_ids = load_model_embeddings(model_dir)
        print(f"Loaded embeddings: tracks={track_emb.shape}, playlists={playlist_emb.shape}")

        # Check embedding norms
        track_norms = np.linalg.norm(track_emb, axis=1)
        print(f"Track embedding norms: min={track_norms.min():.3f}, max={track_norms.max():.3f}, "
              f"mean={track_norms.mean():.3f}")

        # Evaluate with DOT product only
        print("\nDot product similarity:")
        metrics = evaluate_by_popularity_dot(
            track_emb, playlist_emb, train_dict, test_dict, track_popularity, k=args.k
        )

        # Store results
        results[model_name] = metrics

        # Print results
        print(f"\nResults for {model_name} (DOT product):")
        print("-" * 60)
        print(f"Overall Recall@{args.k}:   {metrics['recall_overall']:.4f}")
        print(f"Popular Recall@{args.k}:   {metrics['recall_popular']:.4f}")
        print(f"Unpopular Recall@{args.k}: {metrics['recall_unpopular']:.4f}")
        print(f"# Popular:     {metrics['n_popular']:,}")
        print(f"# Unpopular:   {metrics['n_unpopular']:,}")
        print(f"# Total:       {metrics['n_total']:,}")

        if metrics['recall_unpopular'] > 0:
            ratio = metrics['recall_popular'] / metrics['recall_unpopular']
            print(f"Popular/Unpopular Ratio: {ratio:.2f}x")

    # Summary comparison
    print("\n\n" + "=" * 80)
    print("SUMMARY: Dot Product Results")
    print("=" * 80)
    print(f"{'Model':<20} {'Overall':>12} {'Popular':>12} {'Unpopular':>12} {'Pop/Unpop Ratio':>18}")
    print("-" * 80)

    for model_name in sorted(results.keys()):
        metrics = results[model_name]
        ratio = metrics['recall_popular'] / metrics['recall_unpopular'] if metrics['recall_unpopular'] > 0 else float('inf')
        print(f"{model_name:<20} {metrics['recall_overall']:>12.4f} {metrics['recall_popular']:>12.4f} "
              f"{metrics['recall_unpopular']:>12.4f} {ratio:>18.2f}x")

    print("\n")


if __name__ == "__main__":
    main()
