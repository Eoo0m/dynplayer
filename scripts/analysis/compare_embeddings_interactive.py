"""
Interactive embedding comparison tool

Compare two models by searching for similar tracks interactively

Usage:
    python compare_embeddings_interactive.py \
        --model1-path contrastive_learning_loo/outputs/min5_win10/track_embeddings.npy \
        --model1-name "Contrastive Learning LOO" \
        --model2-path simgcl_loo/outputs/min5_win10/track_embeddings.npy \
        --model2-name "SimGCL LOO" \
        --dataset min5_win10
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml


def load_config(config_path=None):
    """Load config file with relative path support"""
    if config_path is None:
        # Find config.yaml relative to this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_csv_cooccurrence(csv_path, top_k=100):
    """Load CSV and compute track co-occurrence graph"""
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Expand playlist_id if pipe-separated
    if df['playlist_id'].dtype == object and '|' in str(df['playlist_id'].iloc[0]):
        df['playlist_id'] = df['playlist_id'].str.split('|')
        df = df.explode('playlist_id')

    print(f"Total interactions: {len(df):,}")

    # Create track name mapping
    track_info = df[['track_id', 'track', 'artist']].drop_duplicates('track_id')
    track_names = {row['track_id']: f"{row['track']} - {row['artist']}"
                   for _, row in track_info.iterrows()}

    # Group by playlist
    playlist_tracks = df.groupby('playlist_id')['track_id'].apply(list).to_dict()

    # Build co-occurrence matrix
    cooccur = defaultdict(lambda: defaultdict(int))

    for tracks in playlist_tracks.values():
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i+1:]:
                cooccur[t1][t2] += 1
                cooccur[t2][t1] += 1

    # Get top-k neighbors for each track
    track_neighbors = {}
    for track, neighbors in cooccur.items():
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        track_neighbors[track] = [(t, cnt) for t, cnt in sorted_neighbors[:top_k]]

    # Create mappings
    all_tracks = sorted(df['track_id'].unique())
    track_to_idx = {t: i for i, t in enumerate(all_tracks)}
    idx_to_track = {i: t for i, t in enumerate(all_tracks)}

    print(f"Unique tracks: {len(all_tracks):,}")
    print(f"Tracks with neighbors: {len(track_neighbors):,}")

    return track_neighbors, track_to_idx, idx_to_track, track_names


def load_embeddings(model_path):
    """Load track embeddings and track IDs"""
    embeddings = np.load(model_path)

    # Try to load corresponding track_ids file
    track_ids_path = model_path.replace('_track_embeddings.npy', '_track_ids.npy')
    track_ids_path = track_ids_path.replace('_embeddings.npy', '_track_ids.npy')

    if Path(track_ids_path).exists():
        track_ids = np.load(track_ids_path, allow_pickle=True)
        return embeddings, track_ids
    else:
        return embeddings, None


def find_similar_tracks(track_idx, embeddings, k=10, use_cosine=True):
    """Find top-k similar tracks"""
    query_emb = embeddings[track_idx]

    if use_cosine:
        query_emb_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        scores = embeddings_norm @ query_emb_norm
    else:
        scores = embeddings @ query_emb

    scores[track_idx] = -np.inf

    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    top_k_scores = scores[top_k_indices]

    return top_k_indices, top_k_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1-path", required=True)
    parser.add_argument("--model1-name", default="Model 1")
    parser.add_argument("--model2-path", required=True)
    parser.add_argument("--model2-name", default="Model 2")
    parser.add_argument("--dataset", required=True, choices=["min2_win10", "min5_win10"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--use-cosine", action="store_true")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_config = config["data"]["datasets"][args.dataset]
    csv_dir = config["data"]["csv_dir"]
    csv_file = dataset_config["csv_file"]

    # Make CSV path relative to config.yaml location
    if args.config:
        config_dir = Path(args.config).parent
    else:
        config_dir = Path(__file__).parent
    csv_path = config_dir / csv_dir / csv_file

    print(f"\n=== Embedding Comparison (Interactive) ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model 1: {args.model1_name}")
    print(f"Model 2: {args.model2_name}")
    print(f"Similarity: {'Cosine' if args.use_cosine else 'Dot Product'}")
    print(f"k={args.k}")

    # Load embeddings
    print(f"\n=== Loading Embeddings ===")
    emb1, track_ids1 = load_embeddings(args.model1_path)
    emb2, track_ids2 = load_embeddings(args.model2_path)

    print(f"{args.model1_name} shape: {emb1.shape}")
    print(f"{args.model2_name} shape: {emb2.shape}")

    # Use track_ids from embeddings if available, otherwise fall back to CSV
    if track_ids1 is not None and track_ids2 is not None:
        print(f"\nUsing track IDs from embedding files")

        # Verify both models have the same track order
        if not np.array_equal(track_ids1, track_ids2):
            print("WARNING: Models have different track orderings!")
            print(f"Model 1: {len(track_ids1)} tracks")
            print(f"Model 2: {len(track_ids2)} tracks")

        # Use model1's track_ids as reference
        track_to_idx = {tid: i for i, tid in enumerate(track_ids1)}
        idx_to_track = {i: tid for i, tid in enumerate(track_ids1)}

        print(f"Loaded {len(track_ids1)} tracks from embedding files")
    else:
        print(f"\nTrack IDs not found in embedding files, using CSV mapping")
        # Fallback to CSV-based mapping (old behavior)
        all_tracks = sorted(pd.read_csv(csv_path)['track_id'].unique())
        track_to_idx = {t: i for i, t in enumerate(all_tracks)}
        idx_to_track = {i: t for i, t in enumerate(all_tracks)}
        print(f"WARNING: Using CSV-based mapping may not match training order!")

    # Load ground truth (for track names and co-occurrence neighbors)
    print(f"\n=== Loading Ground Truth ===")
    track_neighbors, _, _, track_names = load_csv_cooccurrence(csv_path, top_k=100)

    # Interactive loop
    print(f"\n=== Interactive Search ===")
    print("Enter a track ID to search (or 'quit' to exit, 'random' for random track)")

    while True:
        user_input = input("\nTrack ID: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input.lower() == 'random':
            # Pick random track that has neighbors
            tracks_with_neighbors = [idx for idx in range(len(idx_to_track))
                                    if idx_to_track[idx] in track_neighbors]
            if not tracks_with_neighbors:
                print("No tracks with neighbors found!")
                continue
            track_idx = np.random.choice(tracks_with_neighbors)
            track_id = idx_to_track[track_idx]
            print(f"Random track: {track_id}")
        else:
            track_id = user_input

            if track_id not in track_to_idx:
                print(f"Track ID '{track_id}' not found!")
                continue

            track_idx = track_to_idx[track_id]

        # Get ground truth neighbors
        if track_id in track_neighbors:
            gt_neighbors = track_neighbors[track_id][:args.k]
            gt_ids = [t for t, _ in gt_neighbors]
            gt_counts = [c for _, c in gt_neighbors]
        else:
            gt_ids = []
            gt_counts = []

        # Get predictions from both models
        pred1_indices, pred1_scores = find_similar_tracks(track_idx, emb1, k=args.k, use_cosine=args.use_cosine)
        pred2_indices, pred2_scores = find_similar_tracks(track_idx, emb2, k=args.k, use_cosine=args.use_cosine)

        pred1_ids = [idx_to_track[idx] for idx in pred1_indices]
        pred2_ids = [idx_to_track[idx] for idx in pred2_indices]

        # Compute overlaps
        overlap1 = len(set(pred1_ids) & set(gt_ids))
        overlap2 = len(set(pred2_ids) & set(gt_ids))
        model_agreement = len(set(pred1_ids) & set(pred2_ids))

        # Display results
        print(f"\n{'='*80}")
        print(f"Query Track: {track_id}")
        print(f"{'='*80}")

        print(f"\nGround Truth (Co-occurrence neighbors, top-{args.k}):")
        if gt_ids:
            for i, (tid, cnt) in enumerate(zip(gt_ids, gt_counts), 1):
                track_name = track_names.get(tid, "Unknown")[:60]
                print(f"  {i:2d}. {track_name:60s} (co-occur: {cnt:3d})")
        else:
            print("  (No co-occurrence neighbors found)")

        print(f"\n{args.model1_name}:")
        print(f"  Overlap with GT: {overlap1}/{args.k}")
        for i, (tid, score) in enumerate(zip(pred1_ids, pred1_scores), 1):
            in_gt = "✓" if tid in gt_ids else " "
            in_m2 = "★" if tid in pred2_ids else " "
            track_name = track_names.get(tid, "Unknown")[:60]
            print(f"  {i:2d}. {track_name:60s} (score: {score:.4f}) [{in_gt}] [{in_m2}]")

        print(f"\n{args.model2_name}:")
        print(f"  Overlap with GT: {overlap2}/{args.k}")
        for i, (tid, score) in enumerate(zip(pred2_ids, pred2_scores), 1):
            in_gt = "✓" if tid in gt_ids else " "
            in_m1 = "★" if tid in pred1_ids else " "
            track_name = track_names.get(tid, "Unknown")[:60]
            print(f"  {i:2d}. {track_name:60s} (score: {score:.4f}) [{in_gt}] [{in_m1}]")

        print(f"\nModel Agreement: {model_agreement}/{args.k} tracks in common")
        print(f"Legend: [✓] in ground truth, [★] in other model")


if __name__ == "__main__":
    main()
