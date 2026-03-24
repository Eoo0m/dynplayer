"""
Interactive track search using learned embeddings

Usage:
    python search_tracks_interactive.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_embeddings(embedding_path, keys_path):
    """Load embeddings and keys"""
    embeddings = np.load(embedding_path)
    keys = np.load(keys_path, allow_pickle=True)
    track_to_idx = {tid: i for i, tid in enumerate(keys)}
    return embeddings, keys, track_to_idx


def load_track_metadata(csv_path):
    """Load track metadata from CSV"""
    df = pd.read_csv(csv_path)
    track_metadata = {}
    for _, row in df.iterrows():
        track_id = row['track_id']
        track_metadata[track_id] = {
            'track': row['track'],
            'artist': row['artist'],
            'album': row.get('album', 'Unknown')
        }
    return track_metadata, df


def search_by_title(query, df, track_metadata, limit=10):
    """Search tracks by title"""
    query_lower = query.lower()

    # Filter by title
    matches = df[df['track'].str.lower().str.contains(query_lower, na=False)]

    if len(matches) == 0:
        return []

    results = []
    for _, row in matches.head(limit).iterrows():
        track_id = row['track_id']
        if track_id in track_metadata:
            results.append({
                'track_id': track_id,
                'title': row['track'],
                'artist': row['artist'],
                'album': row.get('album', 'Unknown')
            })

    return results


def find_similar_tracks(query_track_id, embeddings, keys, track_to_idx, k=10):
    """Find k most similar tracks"""
    if query_track_id not in track_to_idx:
        return None

    query_idx = track_to_idx[query_track_id]
    query_emb = embeddings[query_idx]

    # Cosine similarity
    similarities = np.dot(embeddings, query_emb)
    top_k_indices = np.argsort(similarities)[::-1][:k+1]

    results = []
    for idx in top_k_indices:
        track_id = keys[idx]
        if track_id != query_track_id:
            results.append((track_id, similarities[idx]))
        if len(results) == k:
            break

    return results


def main():
    print("="*80)
    print("🎵 Interactive Track Search & Recommendation System")
    print("="*80)

    # Load embeddings
    embedding_path = "contrastive_learning/outputs/min2_win10/model_embeddings.npy"
    keys_path = "contrastive_learning/outputs/min2_win10/model_keys.npy"
    csv_path = "data/csvs/track_playlist_counts_min2_win10.csv"

    print(f"\nLoading embeddings...")
    embeddings, keys, track_to_idx = load_embeddings(embedding_path, keys_path)
    print(f"✅ Loaded {len(keys):,} tracks with {embeddings.shape[1]}-dim embeddings")

    print(f"\nLoading track metadata...")
    track_metadata, df = load_track_metadata(csv_path)
    print(f"✅ Loaded metadata for {len(track_metadata):,} tracks")

    print("\n" + "="*80)
    print("Ready! Type 'exit' or 'quit' to exit.")
    print("="*80)

    while True:
        print("\n" + "-"*80)
        query = input("\n🔍 Search track by title (or 'exit'): ").strip()

        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not query:
            continue

        # Search by title
        print(f"\n🔎 Searching for '{query}'...\n")
        matches = search_by_title(query, df, track_metadata, limit=10)

        if not matches:
            print(f"❌ No tracks found matching '{query}'")
            continue

        # Display search results
        print(f"Found {len(matches)} tracks:\n")
        for i, match in enumerate(matches, 1):
            print(f"{i:2d}. {match['title']} - {match['artist']}")
            print(f"    Album: {match['album']}")
            print(f"    ID: {match['track_id']}")
            print()

        # Ask user to select
        try:
            choice = input("Select track number (or press Enter to search again): ").strip()
            if not choice:
                continue

            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(matches):
                print("❌ Invalid selection")
                continue

            selected = matches[choice_idx]

            # Show selected track
            print(f"\n{'='*80}")
            print(f"📀 Selected Track:")
            print(f"{'='*80}")
            print(f"   Title: {selected['title']}")
            print(f"   Artist: {selected['artist']}")
            print(f"   Album: {selected['album']}")

            # Ask how many recommendations
            top_k = input(f"\nHow many similar tracks? (default 10): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 10

            # Find similar tracks
            print(f"\n🎧 Finding {top_k} similar tracks...\n")
            similar = find_similar_tracks(
                selected['track_id'],
                embeddings,
                keys,
                track_to_idx,
                k=top_k
            )

            if similar is None:
                print(f"❌ Track not found in embeddings")
                continue

            # Display recommendations
            print(f"{'='*80}")
            print(f"✨ Top {top_k} Similar Tracks:")
            print(f"{'='*80}\n")

            for rank, (track_id, similarity) in enumerate(similar, 1):
                if track_id in track_metadata:
                    meta = track_metadata[track_id]
                    print(f"{rank:2d}. {meta['track']} - {meta['artist']}")
                    print(f"    Similarity: {similarity:.4f}")
                    print(f"    Album: {meta['album']}")
                else:
                    print(f"{rank:2d}. [No metadata]")
                    print(f"    Similarity: {similarity:.4f}")
                    print(f"    Track ID: {track_id}")
                print()

        except ValueError:
            print("❌ Invalid input")
            continue
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
