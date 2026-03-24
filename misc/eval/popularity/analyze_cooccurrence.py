"""
Analyze co-occurrence patterns between popular and unpopular tracks.

Compare how often track pairs appear together in playlists:
- Popular tracks (top 10%)
- Unpopular tracks (bottom 10%)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
import random


def main():
    print("=== Loading Data ===")

    # Load playlist-track mapping
    playlist_tracks_df = pd.read_csv("clip/filtered_playlist_tracks.csv")
    print(f"Playlists: {len(playlist_tracks_df):,}")

    # Load track metadata with popularity (count)
    track_meta_df = pd.read_csv("data/csvs/track_playlist_counts_min5_win10.csv")
    track_meta_df["track_id"] = track_meta_df["track_id"].astype(str)
    print(f"Tracks: {len(track_meta_df):,}")

    # Create track_id -> count mapping
    track_popularity = dict(zip(track_meta_df["track_id"], track_meta_df["count"]))

    # Build track -> playlists mapping
    print("\n=== Building Track-Playlist Index ===")
    track_to_playlists = defaultdict(set)

    for _, row in tqdm(playlist_tracks_df.iterrows(), total=len(playlist_tracks_df)):
        pid = str(row["playlist_id"])
        track_ids_str = row["track_ids"]
        if pd.isna(track_ids_str):
            continue

        for tid in track_ids_str.split("|"):
            if tid in track_popularity:
                track_to_playlists[tid].add(pid)

    print(f"Tracks with playlist info: {len(track_to_playlists):,}")

    # Sort tracks by popularity
    sorted_tracks = sorted(track_popularity.items(), key=lambda x: x[1], reverse=True)
    n_tracks = len(sorted_tracks)
    top_10_pct = int(n_tracks * 0.1)

    popular_tracks = [t[0] for t in sorted_tracks[:top_10_pct] if t[0] in track_to_playlists]
    unpopular_tracks = [t[0] for t in sorted_tracks[-top_10_pct:] if t[0] in track_to_playlists]

    print(f"\n=== Track Selection ===")
    print(f"Top 10% (popular): {len(popular_tracks):,} tracks")
    print(f"  Popularity range: {sorted_tracks[top_10_pct-1][1]} - {sorted_tracks[0][1]}")
    print(f"Bottom 10% (unpopular): {len(unpopular_tracks):,} tracks")
    print(f"  Popularity range: {sorted_tracks[-1][1]} - {sorted_tracks[-top_10_pct][1]}")

    # Sample pairs and calculate co-occurrence
    print("\n=== Calculating Co-occurrence ===")
    n_samples = 10000
    random.seed(42)

    def calculate_cooccurrence(tracks, n_samples):
        """Calculate average number of shared playlists for random pairs."""
        cooccurrences = []

        for _ in tqdm(range(n_samples), desc="Sampling pairs"):
            t1, t2 = random.sample(tracks, 2)
            playlists_t1 = track_to_playlists[t1]
            playlists_t2 = track_to_playlists[t2]
            shared = len(playlists_t1 & playlists_t2)
            cooccurrences.append(shared)

        return cooccurrences

    print("\nPopular tracks:")
    popular_cooc = calculate_cooccurrence(popular_tracks, n_samples)

    print("\nUnpopular tracks:")
    unpopular_cooc = calculate_cooccurrence(unpopular_tracks, n_samples)

    # Statistics
    print("\n" + "="*60)
    print("RESULTS: Average Shared Playlists per Track Pair")
    print("="*60)

    print(f"\nPopular tracks (top 10%):")
    print(f"  Mean: {np.mean(popular_cooc):.2f}")
    print(f"  Median: {np.median(popular_cooc):.2f}")
    print(f"  Std: {np.std(popular_cooc):.2f}")
    print(f"  Max: {np.max(popular_cooc)}")
    print(f"  % pairs with 0 shared: {100 * sum(1 for x in popular_cooc if x == 0) / len(popular_cooc):.1f}%")
    print(f"  % pairs with 1+ shared: {100 * sum(1 for x in popular_cooc if x >= 1) / len(popular_cooc):.1f}%")
    print(f"  % pairs with 5+ shared: {100 * sum(1 for x in popular_cooc if x >= 5) / len(popular_cooc):.1f}%")

    print(f"\nUnpopular tracks (bottom 10%):")
    print(f"  Mean: {np.mean(unpopular_cooc):.2f}")
    print(f"  Median: {np.median(unpopular_cooc):.2f}")
    print(f"  Std: {np.std(unpopular_cooc):.2f}")
    print(f"  Max: {np.max(unpopular_cooc)}")
    print(f"  % pairs with 0 shared: {100 * sum(1 for x in unpopular_cooc if x == 0) / len(unpopular_cooc):.1f}%")
    print(f"  % pairs with 1+ shared: {100 * sum(1 for x in unpopular_cooc if x >= 1) / len(unpopular_cooc):.1f}%")
    print(f"  % pairs with 5+ shared: {100 * sum(1 for x in unpopular_cooc if x >= 5) / len(unpopular_cooc):.1f}%")

    # Ratio
    if np.mean(unpopular_cooc) > 0:
        ratio = np.mean(popular_cooc) / np.mean(unpopular_cooc)
        print(f"\nRatio (popular/unpopular): {ratio:.2f}x")


if __name__ == "__main__":
    main()
