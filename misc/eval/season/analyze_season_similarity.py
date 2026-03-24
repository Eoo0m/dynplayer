"""
Analyze similarity between tracks with season keywords (봄, 여름, 가을, 겨울).

Compares:
1. Average embedding similarity between tracks with same season keyword
2. OpenAI text embedding similarity between season keywords
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import os
from dotenv import load_dotenv
from openai import OpenAI


def load_data():
    """Load track tags and embeddings."""
    print("=== Loading Data ===")

    # Load filtered track tags
    tags_df = pd.read_csv("clip copy/track_tags_filtered.csv")
    tags_df["track_id"] = tags_df["track_id"].astype(str)
    print(f"Total tracks with tags: {len(tags_df):,}")

    # Load SimGCL embeddings
    track_embs = np.load("simgcl_randneg/outputs/min5_win10/model_loo_track_embeddings.npy")
    track_ids = np.load("simgcl_randneg/outputs/min5_win10/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = np.array([str(tid) for tid in track_ids])
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    print(f"Track embeddings: {track_embs.shape}")

    return tags_df, track_embs, track_ids, track_id_to_idx


def find_season_tracks(tags_df, track_id_to_idx, min_tag_count=5):
    """Find tracks containing season keywords with at least min_tag_count occurrences."""
    seasons = {
        "봄": [],
        "여름": [],
        "가을": [],
        "겨울": []
    }

    # Also check English equivalents
    season_aliases = {
        "봄": ["봄", "spring"],
        "여름": ["여름", "summer"],
        "가을": ["가을", "autumn", "fall"],
        "겨울": ["겨울", "winter"]
    }

    for _, row in tqdm(tags_df.iterrows(), total=len(tags_df), desc="Finding season tracks"):
        track_id = row["track_id"]
        if track_id not in track_id_to_idx:
            continue

        tags_str = row["tags"]
        tag_counts_str = row["tag_counts"]
        if pd.isna(tags_str) or pd.isna(tag_counts_str):
            continue

        # Parse tag_counts dict
        try:
            tag_counts = eval(tag_counts_str)
        except:
            continue

        for season, aliases in season_aliases.items():
            for alias in aliases:
                alias_lower = alias.lower()
                # Check if this alias exists with count >= min_tag_count
                if alias_lower in tag_counts and tag_counts[alias_lower] >= min_tag_count:
                    seasons[season].append(track_id)
                    break

    print(f"\n=== Season Track Counts (min_tag_count={min_tag_count}) ===")
    for season, tracks in seasons.items():
        print(f"  {season}: {len(tracks)} tracks")

    return seasons


def compute_within_season_similarity(seasons, track_embs, track_id_to_idx, normalize=True):
    """Compute average cosine similarity within each season group."""
    print("\n=== Computing Within-Season Similarity ===")

    results = {}

    for season, track_ids in seasons.items():
        if len(track_ids) < 2:
            print(f"  {season}: Not enough tracks")
            continue

        # Get embeddings
        indices = [track_id_to_idx[tid] for tid in track_ids if tid in track_id_to_idx]
        embs = track_embs[indices]

        if normalize:
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

        # Compute pairwise similarities
        sim_matrix = embs @ embs.T

        # Get upper triangle (exclude diagonal)
        n = len(embs)
        upper_indices = np.triu_indices(n, k=1)
        similarities = sim_matrix[upper_indices]

        results[season] = {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "count": len(track_ids),
            "n_pairs": len(similarities)
        }

        print(f"  {season}: mean={results[season]['mean']:.4f} (±{results[season]['std']:.4f}), n_tracks={len(track_ids)}")

    return results


def compute_between_season_similarity(seasons, track_embs, track_id_to_idx, normalize=True):
    """Compute average cosine similarity between different season groups."""
    print("\n=== Computing Between-Season Similarity ===")

    results = {}
    season_pairs = list(combinations(seasons.keys(), 2))

    for s1, s2 in season_pairs:
        tracks1 = seasons[s1]
        tracks2 = seasons[s2]

        if len(tracks1) == 0 or len(tracks2) == 0:
            continue

        # Get embeddings
        indices1 = [track_id_to_idx[tid] for tid in tracks1 if tid in track_id_to_idx]
        indices2 = [track_id_to_idx[tid] for tid in tracks2 if tid in track_id_to_idx]

        embs1 = track_embs[indices1]
        embs2 = track_embs[indices2]

        if normalize:
            embs1 = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-10)
            embs2 = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-10)

        # Compute cross similarities
        sim_matrix = embs1 @ embs2.T
        similarities = sim_matrix.flatten()

        key = f"{s1}-{s2}"
        results[key] = {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "n_pairs": len(similarities)
        }

        print(f"  {s1} vs {s2}: mean={results[key]['mean']:.4f} (±{results[key]['std']:.4f})")

    return results


def compute_text_embedding_similarity(client):
    """Compute OpenAI text embedding similarity between season keywords."""
    print("\n=== Computing Text Embedding Similarity ===")

    seasons = ["봄", "여름", "가을", "겨울"]

    # Get embeddings
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=seasons
    )

    embeddings = np.array([d.embedding for d in response.data])

    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.T

    print("\nSeason Keyword Text Embedding Similarity Matrix:")
    print(f"{'':>8}", end="")
    for s in seasons:
        print(f"{s:>8}", end="")
    print()

    for i, s1 in enumerate(seasons):
        print(f"{s1:>8}", end="")
        for j, s2 in enumerate(seasons):
            print(f"{sim_matrix[i,j]:>8.4f}", end="")
        print()

    # Extract pairwise similarities
    results = {}
    for i, s1 in enumerate(seasons):
        for j, s2 in enumerate(seasons):
            if i < j:
                key = f"{s1}-{s2}"
                results[key] = float(sim_matrix[i, j])

    return results, sim_matrix


def compute_random_baseline(track_embs, n_samples=1000, sample_size=100, normalize=True):
    """Compute baseline similarity for random track pairs."""
    print("\n=== Computing Random Baseline ===")

    n_tracks = len(track_embs)
    all_sims = []

    for _ in tqdm(range(n_samples), desc="Sampling"):
        # Sample random pairs
        idx1 = np.random.randint(0, n_tracks, sample_size)
        idx2 = np.random.randint(0, n_tracks, sample_size)

        embs1 = track_embs[idx1]
        embs2 = track_embs[idx2]

        if normalize:
            embs1 = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-10)
            embs2 = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-10)

        # Compute diagonal (paired) similarities
        sims = np.sum(embs1 * embs2, axis=1)
        all_sims.extend(sims)

    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)

    print(f"  Random baseline: mean={mean_sim:.4f} (±{std_sim:.4f})")

    return mean_sim, std_sim


def compute_track_similarity_matrix(seasons, track_embs, track_id_to_idx, normalize=True):
    """Compute 4x4 similarity matrix for track embeddings."""
    season_list = ["봄", "여름", "가을", "겨울"]
    n = len(season_list)
    sim_matrix = np.zeros((n, n))

    for i, s1 in enumerate(season_list):
        for j, s2 in enumerate(season_list):
            tracks1 = seasons[s1]
            tracks2 = seasons[s2]

            if len(tracks1) == 0 or len(tracks2) == 0:
                continue

            indices1 = [track_id_to_idx[tid] for tid in tracks1 if tid in track_id_to_idx]
            indices2 = [track_id_to_idx[tid] for tid in tracks2 if tid in track_id_to_idx]

            embs1 = track_embs[indices1]
            embs2 = track_embs[indices2]

            if normalize:
                embs1 = embs1 / (np.linalg.norm(embs1, axis=1, keepdims=True) + 1e-10)
                embs2 = embs2 / (np.linalg.norm(embs2, axis=1, keepdims=True) + 1e-10)

            # Compute mean similarity
            cross_sim = embs1 @ embs2.T
            sim_matrix[i, j] = np.mean(cross_sim)

    return sim_matrix, season_list


def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load data
    tags_df, track_embs, track_ids, track_id_to_idx = load_data()

    # Find season tracks
    seasons = find_season_tracks(tags_df, track_id_to_idx)

    # Compute random baseline
    random_mean, random_std = compute_random_baseline(track_embs)

    # Compute 4x4 track similarity matrix
    track_sim_matrix, season_list = compute_track_similarity_matrix(seasons, track_embs, track_id_to_idx)

    # Compute text embedding similarity
    text_results, text_sim_matrix = compute_text_embedding_similarity(client)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nRandom baseline: {random_mean:.4f}")

    # Print Track Embedding Similarity Matrix
    print("\n[Track Embedding Similarity Matrix (Cosine)]")
    print(f"{'':>8}", end="")
    for s in season_list:
        print(f"{s:>8}", end="")
    print()

    for i, s1 in enumerate(season_list):
        print(f"{s1:>8}", end="")
        for j, s2 in enumerate(season_list):
            print(f"{track_sim_matrix[i,j]:>8.4f}", end="")
        print()

    # Print Text Embedding Similarity Matrix (already printed, but repeat for comparison)
    print("\n[Text Embedding Similarity Matrix (OpenAI)]")
    print(f"{'':>8}", end="")
    for s in season_list:
        print(f"{s:>8}", end="")
    print()

    for i, s1 in enumerate(season_list):
        print(f"{s1:>8}", end="")
        for j, s2 in enumerate(season_list):
            print(f"{text_sim_matrix[i,j]:>8.4f}", end="")
        print()

    # Difference matrix
    print("\n[Difference: Track - Text]")
    print(f"{'':>8}", end="")
    for s in season_list:
        print(f"{s:>8}", end="")
    print()

    for i, s1 in enumerate(season_list):
        print(f"{s1:>8}", end="")
        for j, s2 in enumerate(season_list):
            diff = track_sim_matrix[i,j] - text_sim_matrix[i,j]
            print(f"{diff:>+8.4f}", end="")
        print()


if __name__ == "__main__":
    main()
