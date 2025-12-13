"""
Search playlists using text queries via trained CLIP model
Then recommend tracks similar to the playlist embedding
"""

import argparse
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
import os
from dotenv import load_dotenv

from model import CaptionPlaylistCLIP


def embed_text(client, text, model="text-embedding-3-large"):
    """Embed a single text query using OpenAI API"""
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def recommend_tracks_by_frequency(
    playlist_results, playlist_df, track_meta_df, top_k=10
):
    """
    Recommend tracks based on weighted frequency in top matching playlists
    Frequency is weighted by playlist similarity score

    Args:
        playlist_results: list of (playlist_id, title, score) tuples
        playlist_df: dataframe with playlist metadata (must have 'playlist_id' and 'track_ids' columns)
        track_meta_df: dataframe with track metadata
        top_k: number of tracks to recommend

    Returns:
        list of (track_id, track_name, artist, weighted_score) tuples
    """
    from collections import defaultdict

    # Count weighted track occurrences across all top playlists
    track_scores = defaultdict(float)

    for pid, _, similarity_score in playlist_results:
        # Get tracks for this playlist
        playlist_info = playlist_df[playlist_df["playlist_id"] == pid]
        if len(playlist_info) > 0:
            tracks_str = playlist_info.iloc[0].get("track_ids", "")
            if pd.notna(tracks_str) and tracks_str:
                track_list = [t.strip() for t in tracks_str.split("|") if t.strip()]
                # Add similarity score as weight for each track
                for track_id in track_list:
                    track_scores[track_id] += similarity_score

    # Get top-k tracks by weighted score
    sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    results = []
    for track_id, weighted_score in sorted_tracks:
        # Get track metadata
        track_info = track_meta_df[track_meta_df["track_id"] == track_id]
        if len(track_info) > 0:
            track_name = track_info.iloc[0].get("track", "Unknown")
            artist = track_info.iloc[0].get("artist", "Unknown")
            results.append((track_id, track_name, artist, weighted_score))
        else:
            results.append((track_id, "Unknown", "Unknown", weighted_score))

    return results


def search_playlists(
    query_text,
    model,
    client,
    playlist_ids,
    playlist_embs,
    playlist_df,
    device,
    embedding_model="text-embedding-3-large",
    top_k=10,
):
    """
    Search for playlists matching the query text

    Args:
        query_text: search query string
        model: trained CLIP model
        client: OpenAI client
        playlist_ids: list of playlist IDs
        playlist_embs: playlist embeddings (N, dim)
        playlist_df: dataframe with playlist metadata
        device: torch device
        embedding_model: OpenAI embedding model to use
        top_k: number of results to return

    Returns:
        list of (playlist_id, title, score) tuples
    """
    # 1. Embed query text
    query_emb = embed_text(client, query_text, model=embedding_model)
    query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Project query through caption encoder
    model.eval()
    with torch.no_grad():
        projected_query = model.caption_proj(query_tensor)  # (1, out_dim)

    # 3. Use pre-projected playlist embeddings (already in CLIP space)
    all_projected = torch.tensor(playlist_embs, dtype=torch.float32).to(
        device
    )  # (N, out_dim)

    # 4. Compute similarities
    with torch.no_grad():
        similarities = torch.matmul(projected_query, all_projected.T).squeeze(0)  # (N,)
        scores, top_indices = torch.topk(similarities, k=min(top_k, len(playlist_ids)))
        scores = scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

    # 5. Gather results
    results = []
    for score, idx in zip(scores, top_indices):
        pid = playlist_ids[idx]

        # Get playlist metadata
        playlist_info = playlist_df[playlist_df["playlist_id"] == pid]
        if len(playlist_info) > 0:
            title = playlist_info.iloc[0]["playlist_title"]
            results.append((pid, title, float(score)))
        else:
            results.append((pid, "Unknown", float(score)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="clip/clip_best.pt",
        help="Path to trained CLIP model",
    )
    parser.add_argument(
        "--playlist-ids",
        default="clip/clip_playlist_ids.npy",
        help="Playlist IDs",
    )
    parser.add_argument(
        "--playlist-embs",
        default="clip/clip_playlist_projected.npy",
        help="Playlist embeddings (CLIP projected)",
    )
    parser.add_argument(
        "--playlist-csv",
        default="clip/filtered_playlist_tracks.csv",
        help="Playlist metadata CSV",
    )
    parser.add_argument(
        "--track-ids",
        default="lightgcn/contrastive_top1pct_basic_keys.npy",
        help="Track IDs",
    )
    parser.add_argument(
        "--track-embs",
        default="lightgcn/contrastive_top1pct_basic_embeddings.npy",
        help="Track embeddings",
    )
    parser.add_argument(
        "--track-csv",
        default="contrastive_learning2/contrastive_meta.csv",
        help="Track metadata CSV",
    )
    parser.add_argument("--query", type=str, help="Search query text")
    parser.add_argument(
        "--top-k-playlists", type=int, default=50, help="Number of playlists to search"
    )
    parser.add_argument(
        "--top-k-tracks", type=int, default=10, help="Number of tracks to recommend"
    )
    parser.add_argument(
        "--caption-dim",
        type=int,
        default=3072,
        help="OpenAI embedding dimension (3072 for text-embedding-3-large, 1536 for text-embedding-3-small)",
    )
    parser.add_argument(
        "--playlist-dim",
        type=int,
        default=256,
        help="Playlist embedding dimension (only needed if using baseline embeddings)",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=512,
        help="CLIP output dimension (should match projected embeddings)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI embedding model",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load playlist data
    print("\n=== Loading Playlist Data ===")
    playlist_ids = np.load(args.playlist_ids, allow_pickle=True)
    playlist_embs = np.load(args.playlist_embs)
    playlist_df = pd.read_csv(args.playlist_csv)

    print(f"Loaded {len(playlist_ids):,} playlists")
    print(f"Playlist embeddings shape: {playlist_embs.shape}")

    # Load track data
    print("\n=== Loading Track Data ===")
    track_ids = np.load(args.track_ids, allow_pickle=True)
    track_embs = np.load(args.track_embs)
    track_df = pd.read_csv(args.track_csv)

    print(f"Loaded {len(track_ids):,} tracks")
    print(f"Track embeddings shape: {track_embs.shape}")

    # Load trained model
    print("\n=== Loading CLIP Model ===")
    model = CaptionPlaylistCLIP(
        caption_dim=args.caption_dim,
        playlist_dim=args.playlist_dim,
        out_dim=args.output_dim,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Interactive search or single query
    if args.query:
        # Single query mode
        print(f"\n=== Searching for: '{args.query}' ===")
        playlist_results = search_playlists(
            args.query,
            model,
            client,
            playlist_ids,
            playlist_embs,
            playlist_df,
            device,
            embedding_model=args.embedding_model,
            top_k=args.top_k_playlists,
        )

        print(f"\nTop {len(playlist_results)} matching playlist(s):")
        print("-" * 80)
        for i, (pid, title, score) in enumerate(
            playlist_results[:20], 1
        ):  # Show first 10
            print(f"{i}. [{score:.4f}] {title}")
            print(f"   ID: {pid}")
        if len(playlist_results) > 20:
            print(f"... and {len(playlist_results) - 20} more playlists")
        print("-" * 80)

        # Recommend tracks based on weighted frequency across top playlists
        print(
            f"\n=== Most Frequent Tracks in Top {len(playlist_results)} Playlists (Weighted by Similarity) ==="
        )
        track_results = recommend_tracks_by_frequency(
            playlist_results,
            playlist_df,
            track_df,
            top_k=args.top_k_tracks,
        )

        print(f"\nTop {len(track_results)} recommended tracks:")
        print("-" * 80)
        for i, (track_id, track_name, artist, weighted_score) in enumerate(
            track_results, 1
        ):
            print(f"{i:2d}. [Score: {weighted_score:.2f}] {track_name}")
            print(f"    Artist: {artist}")
            print(f"    ID: {track_id}")
        print("-" * 80)
    else:
        # Interactive mode
        print("\n=== Interactive Playlist Search & Track Recommendation ===")
        print("Enter search queries (or 'quit' to exit)")
        print("-" * 80)

        while True:
            query = input("\nSearch query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print(f"\nSearching for: '{query}'")
            playlist_results = search_playlists(
                query,
                model,
                client,
                playlist_ids,
                playlist_embs,
                playlist_df,
                device,
                embedding_model=args.embedding_model,
                top_k=args.top_k_playlists,
            )

            print(f"\nTop {len(playlist_results)} matching playlist(s):")
            print("-" * 80)
            for i, (pid, title, score) in enumerate(
                playlist_results[:10], 1
            ):  # Show first 10
                print(f"{i}. [{score:.4f}] {title}")
                print(f"   ID: {pid}")
            if len(playlist_results) > 10:
                print(f"... and {len(playlist_results) - 10} more playlists")
            print("-" * 80)

            # Recommend tracks based on weighted frequency across top playlists
            print(
                f"\n=== Most Frequent Tracks in Top {len(playlist_results)} Playlists (Weighted by Similarity) ==="
            )
            track_results = recommend_tracks_by_frequency(
                playlist_results,
                playlist_df,
                track_df,
                top_k=args.top_k_tracks,
            )

            print(f"\nTop {len(track_results)} recommended tracks:")
            print("-" * 80)
            for i, (track_id, track_name, artist, weighted_score) in enumerate(
                track_results, 1
            ):
                print(f"{i:2d}. [Score: {weighted_score:.2f}] {track_name}")
                print(f"    Artist: {artist}")
                print(f"    ID: {track_id}")
            print("-" * 80)


if __name__ == "__main__":
    main()
