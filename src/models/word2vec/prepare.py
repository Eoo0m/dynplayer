"""
Prepare playlist sequences for Word2Vec training with LOO (Leave-One-Out) split.

1. Load playlist-track data
2. Split playlists into train/test (80/20)
3. For test playlists, use LOO: keep one track for testing
4. Create sequences from train playlists only

Usage:
    python word2vec/prepare.py --csv data/csvs/track_playlist_counts_min5_win10.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import argparse
from tqdm import tqdm
import random


def load_playlist_data(csv_path):
    """
    Load track-playlist data from CSV.
    Each row contains a track and its associated playlists (pipe-separated).
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} tracks")

    # Build playlist -> tracks mapping
    playlist_to_tracks = defaultdict(list)
    track_to_info = {}
    all_track_ids = set()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building playlist-track mapping"):
        track_id = row['track_id']
        all_track_ids.add(track_id)
        track_to_info[track_id] = {
            'track': row['track'],
            'artist': row['artist'],
            'album': row['album']
        }

        # Parse playlist IDs
        playlist_ids = str(row['playlist_id']).split('|')
        for playlist_id in playlist_ids:
            if playlist_id and playlist_id != 'nan':
                playlist_to_tracks[playlist_id].append(track_id)

    print(f"Found {len(playlist_to_tracks):,} playlists")
    print(f"Found {len(all_track_ids):,} unique tracks")

    return playlist_to_tracks, track_to_info, list(all_track_ids)


def create_loo_split(playlist_to_tracks, all_track_ids, train_ratio=0.8, min_playlist_length=5, seed=42):
    """
    Create LOO (Leave-One-Out) split.

    - Split playlists into train/test (80/20)
    - For test playlists: randomly hold out one track for evaluation
    - Build track_to_idx mapping

    Returns:
        train_playlist_tracks: {playlist_id: [track_ids]} for training
        test_playlist_observed: {playlist_id: [track_ids]} observed tracks
        test_playlist_masked: {playlist_id: track_id} held-out track
        track_to_idx: {track_id: idx}
    """
    random.seed(seed)
    np.random.seed(seed)

    # Filter playlists by length
    valid_playlists = {
        pid: tracks for pid, tracks in playlist_to_tracks.items()
        if len(tracks) >= min_playlist_length
    }
    print(f"Valid playlists (>= {min_playlist_length} tracks): {len(valid_playlists):,}")

    # Create track_to_idx mapping
    track_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}

    # Split playlists
    playlist_ids = list(valid_playlists.keys())
    random.shuffle(playlist_ids)

    n_train = int(len(playlist_ids) * train_ratio)
    train_pids = set(playlist_ids[:n_train])
    test_pids = set(playlist_ids[n_train:])

    print(f"Train playlists: {len(train_pids):,}")
    print(f"Test playlists: {len(test_pids):,}")

    # Train: all tracks in train playlists
    train_playlist_tracks = {}
    for pid in train_pids:
        tracks = valid_playlists[pid]
        train_playlist_tracks[pid] = [track_to_idx[t] for t in tracks if t in track_to_idx]

    # Test: LOO split (hold out one random track)
    test_playlist_observed = {}
    test_playlist_masked = {}

    for pid in test_pids:
        tracks = valid_playlists[pid]
        track_indices = [track_to_idx[t] for t in tracks if t in track_to_idx]

        if len(track_indices) < 2:
            continue

        # Randomly select one track to mask
        mask_idx = random.randint(0, len(track_indices) - 1)
        masked_track = track_indices[mask_idx]
        observed_tracks = track_indices[:mask_idx] + track_indices[mask_idx + 1:]

        test_playlist_observed[pid] = observed_tracks
        test_playlist_masked[pid] = masked_track

    print(f"Test playlists with LOO: {len(test_playlist_observed):,}")

    return train_playlist_tracks, test_playlist_observed, test_playlist_masked, track_to_idx


def create_sequences_from_train(train_playlist_tracks, idx_to_track, max_playlist_length=500):
    """
    Create track sequences from train playlists only.
    Convert indices back to track_ids for Gensim Word2Vec.
    """
    sequences = []

    for pid, track_indices in tqdm(train_playlist_tracks.items(), desc="Creating sequences"):
        # Truncate very long playlists
        if len(track_indices) > max_playlist_length:
            track_indices = track_indices[:max_playlist_length]

        # Convert indices to track_ids
        track_ids = [idx_to_track[idx] for idx in track_indices]
        sequences.append(track_ids)

    print(f"Created {len(sequences):,} sequences")

    # Statistics
    lengths = [len(s) for s in sequences]
    print(f"Sequence length stats:")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths):.1f}")

    return sequences


def save_data(sequences, track_to_info, train_playlist_tracks, test_playlist_observed,
              test_playlist_masked, track_to_idx, all_track_ids, output_dir):
    """Save all data for training and evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sequences for Word2Vec
    sequences_path = output_dir / "sequences.pkl"
    with open(sequences_path, 'wb') as f:
        pickle.dump(sequences, f)
    print(f"Saved sequences to {sequences_path}")

    # Save track info
    track_info_path = output_dir / "track_info.pkl"
    with open(track_info_path, 'wb') as f:
        pickle.dump(track_to_info, f)
    print(f"Saved track info to {track_info_path}")

    # Save LOO split
    split_path = output_dir / "loo_split.npz"
    np.savez(
        split_path,
        train_playlist_tracks=train_playlist_tracks,
        test_playlist_observed=test_playlist_observed,
        test_playlist_masked=test_playlist_masked,
        track_to_idx=track_to_idx,
        all_track_ids=np.array(all_track_ids),
        num_tracks=len(all_track_ids),
    )
    print(f"Saved LOO split to {split_path}")

    # Save sample sequences as text for inspection
    text_path = output_dir / "sequences.txt"
    with open(text_path, 'w') as f:
        for seq in sequences[:1000]:
            f.write(' '.join(seq) + '\n')
    print(f"Saved sample sequences to {text_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Word2Vec data with LOO split")
    parser.add_argument("--csv", type=str,
                        default="data/csvs/track_playlist_counts_min5_win10.csv",
                        help="Path to track-playlist CSV")
    parser.add_argument("--output-dir", type=str, default="word2vec/data",
                        help="Output directory for processed data")
    parser.add_argument("--min-length", type=int, default=5,
                        help="Minimum playlist length")
    parser.add_argument("--max-length", type=int, default=500,
                        help="Maximum playlist length")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train/test split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Load data
    playlist_to_tracks, track_to_info, all_track_ids = load_playlist_data(args.csv)

    # Create LOO split
    train_playlist_tracks, test_playlist_observed, test_playlist_masked, track_to_idx = create_loo_split(
        playlist_to_tracks,
        all_track_ids,
        train_ratio=args.train_ratio,
        min_playlist_length=args.min_length,
        seed=args.seed
    )

    # Create sequences from train playlists
    idx_to_track = {v: k for k, v in track_to_idx.items()}
    sequences = create_sequences_from_train(
        train_playlist_tracks,
        idx_to_track,
        max_playlist_length=args.max_length
    )

    # Save
    save_data(
        sequences, track_to_info,
        train_playlist_tracks, test_playlist_observed, test_playlist_masked,
        track_to_idx, all_track_ids, args.output_dir
    )

    print("\nDone! Ready for Word2Vec training with LOO evaluation.")


if __name__ == "__main__":
    main()
