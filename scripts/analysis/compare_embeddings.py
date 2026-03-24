"""
Compare embeddings from two models by searching for similar tracks

For each test track:
1. Find top-10 similar tracks from model 1
2. Find top-10 similar tracks from model 2
3. Find co-occurrence neighbors from original CSV
4. Calculate overlap between model predictions and ground truth neighbors
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_csv_cooccurrence(csv_path, top_k=50):
    """
    Load CSV and compute track co-occurrence graph (ground truth neighbors)

    Returns:
        track_neighbors: {track_id: [neighbor_track_ids sorted by co-occurrence count]}
        track_to_idx: {track_id: index}
        idx_to_track: {index: track_id}
    """
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Expand playlist_id if pipe-separated
    if df['playlist_id'].dtype == object and '|' in str(df['playlist_id'].iloc[0]):
        df['playlist_id'] = df['playlist_id'].str.split('|')
        df = df.explode('playlist_id')

    print(f"Total interactions: {len(df):,}")

    # Group by playlist
    playlist_tracks = df.groupby('playlist_id')['track_id'].apply(list).to_dict()

    # Build co-occurrence matrix
    cooccur = defaultdict(lambda: defaultdict(int))

    for tracks in playlist_tracks.values():
        # For each pair of tracks in the same playlist
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i+1:]:
                cooccur[t1][t2] += 1
                cooccur[t2][t1] += 1

    # Get top-k neighbors for each track
    track_neighbors = {}
    for track, neighbors in cooccur.items():
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        track_neighbors[track] = [t for t, _ in sorted_neighbors[:top_k]]

    # Create mappings
    all_tracks = sorted(df['track_id'].unique())
    track_to_idx = {t: i for i, t in enumerate(all_tracks)}
    idx_to_track = {i: t for i, t in enumerate(all_tracks)}

    print(f"Unique tracks: {len(all_tracks):,}")
    print(f"Tracks with neighbors: {len(track_neighbors):,}")

    return track_neighbors, track_to_idx, idx_to_track


def load_embeddings(model_path):
    """Load track embeddings from model"""
    if model_path.endswith('.npy'):
        return np.load(model_path)
    else:
        raise ValueError(f"Unsupported file format: {model_path}")


def find_similar_tracks(track_idx, embeddings, k=10, use_cosine=True):
    """
    Find top-k similar tracks using embedding similarity

    Args:
        track_idx: index of query track
        embeddings: [num_tracks, dim] track embeddings
        k: number of neighbors to return
        use_cosine: if True, use cosine similarity; otherwise use dot product

    Returns:
        top_k_indices: [k] indices of most similar tracks (excluding query)
    """
    query_emb = embeddings[track_idx]

    if use_cosine:
        # Normalize for cosine similarity
        query_emb_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        scores = embeddings_norm @ query_emb_norm
    else:
        # Dot product
        scores = embeddings @ query_emb

    # Mask out the query track itself
    scores[track_idx] = -np.inf

    # Get top-k
    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

    return top_k_indices


def compute_overlap(pred_tracks, true_neighbors):
    """
    Compute overlap between predicted tracks and ground truth neighbors

    Returns:
        overlap_count: number of tracks in both lists
        overlap_ratio: overlap_count / len(pred_tracks)
    """
    pred_set = set(pred_tracks)
    true_set = set(true_neighbors)

    overlap = pred_set & true_set
    overlap_count = len(overlap)
    overlap_ratio = overlap_count / len(pred_tracks) if pred_tracks else 0.0

    return overlap_count, overlap_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1-path", required=True, help="Path to model 1 track embeddings (.npy)")
    parser.add_argument("--model1-name", default="Model 1", help="Name of model 1")
    parser.add_argument("--model2-path", required=True, help="Path to model 2 track embeddings (.npy)")
    parser.add_argument("--model2-name", default="Model 2", help="Name of model 2")
    parser.add_argument("--dataset", required=True, choices=["min2_win10", "min5_win10"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--k", type=int, default=10, help="Number of similar tracks to retrieve")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of tracks to sample for comparison")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity (default: dot product)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load config
    config = load_config(args.config)
    dataset_config = config["data"]["datasets"][args.dataset]
    csv_dir = config["data"]["csv_dir"]
    csv_file = dataset_config["csv_file"]
    csv_path = f"{csv_dir}/{csv_file}"

    print(f"\n=== Embedding Comparison ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model 1: {args.model1_name} ({args.model1_path})")
    print(f"Model 2: {args.model2_name} ({args.model2_path})")
    print(f"Similarity: {'Cosine' if args.use_cosine else 'Dot Product'}")
    print(f"k={args.k}, samples={args.num_samples}")

    # Load CSV co-occurrence (ground truth)
    print(f"\n=== Loading Ground Truth ===")
    track_neighbors, track_to_idx, idx_to_track = load_csv_cooccurrence(csv_path, top_k=100)

    # Load embeddings
    print(f"\n=== Loading Embeddings ===")
    emb1 = load_embeddings(args.model1_path)
    emb2 = load_embeddings(args.model2_path)

    print(f"{args.model1_name} shape: {emb1.shape}")
    print(f"{args.model2_name} shape: {emb2.shape}")

    if emb1.shape != emb2.shape:
        raise ValueError(f"Embedding shapes don't match: {emb1.shape} vs {emb2.shape}")

    num_tracks = emb1.shape[0]

    # Sample tracks that have ground truth neighbors
    tracks_with_neighbors = [idx for idx in range(num_tracks) if idx_to_track[idx] in track_neighbors]

    if len(tracks_with_neighbors) < args.num_samples:
        print(f"Warning: Only {len(tracks_with_neighbors)} tracks have neighbors, using all")
        sample_indices = tracks_with_neighbors
    else:
        sample_indices = np.random.choice(tracks_with_neighbors, args.num_samples, replace=False)

    # Compare models
    print(f"\n=== Comparing {len(sample_indices)} tracks ===")

    model1_overlaps = []
    model2_overlaps = []
    both_overlaps = []

    for track_idx in sample_indices:
        track_id = idx_to_track[track_idx]
        true_neighbors_ids = track_neighbors[track_id]

        # Convert to indices
        true_neighbors_indices = [track_to_idx[tid] for tid in true_neighbors_ids if tid in track_to_idx]

        if len(true_neighbors_indices) == 0:
            continue

        # Get predictions from both models
        pred1_indices = find_similar_tracks(track_idx, emb1, k=args.k, use_cosine=args.use_cosine)
        pred2_indices = find_similar_tracks(track_idx, emb2, k=args.k, use_cosine=args.use_cosine)

        # Convert to track IDs
        pred1_ids = [idx_to_track[idx] for idx in pred1_indices]
        pred2_ids = [idx_to_track[idx] for idx in pred2_indices]

        # Compute overlaps with ground truth
        overlap1_count, overlap1_ratio = compute_overlap(pred1_ids, true_neighbors_ids)
        overlap2_count, overlap2_ratio = compute_overlap(pred2_ids, true_neighbors_ids)

        # Compute overlap between two models
        both_overlap = len(set(pred1_ids) & set(pred2_ids))

        model1_overlaps.append(overlap1_count)
        model2_overlaps.append(overlap2_count)
        both_overlaps.append(both_overlap)

    # Print results
    print(f"\n=== Results ===")
    print(f"Evaluated {len(model1_overlaps)} tracks")
    print(f"\n{args.model1_name}:")
    print(f"  Avg overlap with ground truth: {np.mean(model1_overlaps):.2f}/{args.k} ({np.mean(model1_overlaps)/args.k*100:.1f}%)")
    print(f"  Median: {np.median(model1_overlaps):.0f}, Min: {np.min(model1_overlaps):.0f}, Max: {np.max(model1_overlaps):.0f}")

    print(f"\n{args.model2_name}:")
    print(f"  Avg overlap with ground truth: {np.mean(model2_overlaps):.2f}/{args.k} ({np.mean(model2_overlaps)/args.k*100:.1f}%)")
    print(f"  Median: {np.median(model2_overlaps):.0f}, Min: {np.min(model2_overlaps):.0f}, Max: {np.max(model2_overlaps):.0f}")

    print(f"\nModel Agreement:")
    print(f"  Avg overlap between models: {np.mean(both_overlaps):.2f}/{args.k} ({np.mean(both_overlaps)/args.k*100:.1f}%)")

    # Example: show a few sample comparisons
    print(f"\n=== Sample Comparisons (first 3) ===")
    for i, track_idx in enumerate(sample_indices[:3]):
        track_id = idx_to_track[track_idx]
        true_neighbors_ids = track_neighbors[track_id][:args.k]

        pred1_indices = find_similar_tracks(track_idx, emb1, k=args.k, use_cosine=args.use_cosine)
        pred2_indices = find_similar_tracks(track_idx, emb2, k=args.k, use_cosine=args.use_cosine)

        pred1_ids = [idx_to_track[idx] for idx in pred1_indices]
        pred2_ids = [idx_to_track[idx] for idx in pred2_indices]

        overlap1_count = len(set(pred1_ids) & set(true_neighbors_ids))
        overlap2_count = len(set(pred2_ids) & set(true_neighbors_ids))

        print(f"\nTrack {track_id}:")
        print(f"  Ground truth (top-{args.k}): {true_neighbors_ids[:5]}...")
        print(f"  {args.model1_name}: {pred1_ids[:5]}... (overlap: {overlap1_count}/{args.k})")
        print(f"  {args.model2_name}: {pred2_ids[:5]}... (overlap: {overlap2_count}/{args.k})")


if __name__ == "__main__":
    main()
