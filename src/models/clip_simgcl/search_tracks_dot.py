"""
Search tracks using text queries via trained CLIP model (Dot Product version).

Uses dot product similarity (not cosine) to leverage popularity encoded in embedding norms.
"""

import argparse
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_simgcl.train_dot import CaptionPlaylistCLIPDot


def embed_text(client, text, model="text-embedding-3-large"):
    """Embed a single text query using OpenAI API"""
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def search_tracks_dot(
    query_text,
    model,
    client,
    track_ids,
    track_embs,
    track_df,
    tag_df,
    device,
    embedding_model="text-embedding-3-large",
    top_k=50,
    use_cosine=False,
):
    """
    Search for tracks matching the query text using dot product similarity.

    Args:
        query_text: search query string
        model: trained CLIP model (dot product version)
        client: OpenAI client
        track_ids: list of track IDs
        track_embs: SimGCL track embeddings (N, 64)
        track_df: dataframe with track metadata
        tag_df: dataframe with track tags (or None)
        device: torch device
        embedding_model: OpenAI embedding model to use
        top_k: number of results to return
        use_cosine: if True, use cosine similarity instead of dot product

    Returns:
        list of (track_id, track_name, artist, score, norm, count, tags) tuples
    """
    # 1. Embed query text
    print(f"Embedding query: '{query_text}'")
    query_emb = embed_text(client, query_text, model=embedding_model)
    query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Project query through caption encoder
    model.eval()
    with torch.no_grad():
        projected_query = model.caption_proj(query_tensor)  # (1, out_dim)

    # 3. Project all track embeddings through playlist projection
    print(f"Projecting {len(track_ids):,} track embeddings...")
    all_projected = []
    batch_size = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(track_ids), batch_size), desc="Projecting tracks"):
            batch_embs = torch.tensor(
                track_embs[i:i+batch_size],
                dtype=torch.float32
            ).to(device)

            # Project track embeddings through playlist projection
            projected = model.playlist_proj(batch_embs)  # (batch, out_dim)
            all_projected.append(projected)

    all_projected = torch.cat(all_projected, dim=0)  # (N, out_dim)

    # 4. Compute similarities
    print(f"Computing similarities ({'cosine' if use_cosine else 'dot product'})...")
    with torch.no_grad():
        if use_cosine:
            # Normalize for cosine similarity
            projected_query_norm = F.normalize(projected_query, dim=-1)
            all_projected_norm = F.normalize(all_projected, dim=-1)
            similarities = torch.matmul(projected_query_norm, all_projected_norm.T).squeeze(0)
        else:
            # Dot product (raw)
            similarities = torch.matmul(projected_query, all_projected.T).squeeze(0)

        scores, top_indices = torch.topk(similarities, k=min(top_k, len(track_ids)))
        scores = scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

    # Compute norms for analysis
    norms = torch.norm(all_projected, dim=1).cpu().numpy()

    # 5. Gather results with metadata
    results = []
    for score, idx in zip(scores, top_indices):
        track_id = str(track_ids[idx])
        norm = float(norms[idx])

        # Get track metadata
        track_info = track_df[track_df["track_id"] == track_id]
        if len(track_info) > 0:
            track_name = track_info.iloc[0].get("track", "Unknown")
            artist = track_info.iloc[0].get("artist", "Unknown")
            count = int(track_info.iloc[0].get("count", 0))
        else:
            track_name = "Unknown"
            artist = "Unknown"
            count = 0

        # Get tags
        tags = []
        if tag_df is not None:
            tag_info = tag_df[tag_df["track_id"] == track_id]
            if len(tag_info) > 0:
                tags_raw = tag_info.iloc[0].get("tags", "[]")
                try:
                    tags = ast.literal_eval(tags_raw) if isinstance(tags_raw, str) else tags_raw
                except:
                    tags = []

        results.append((track_id, track_name, artist, float(score), norm, count, tags))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="clip_simgcl/exp_dot",
        help="Directory containing trained dot product CLIP model",
    )
    parser.add_argument(
        "--simgcl-dir",
        default="models/simgcl_randneg/outputs/min5_win10",
        help="SimGCL outputs directory",
    )
    parser.add_argument(
        "--track-csv",
        default="data/csvs/track_playlist_counts_min5_win10.csv",
        help="Track metadata CSV",
    )
    parser.add_argument("--query", type=str, help="Search query text")
    parser.add_argument(
        "--top-k", type=int, default=50, help="Number of tracks to return"
    )
    parser.add_argument(
        "--caption-dim",
        type=int,
        default=3072,
        help="OpenAI embedding dimension",
    )
    parser.add_argument(
        "--track-dim",
        type=int,
        default=64,
        help="SimGCL track embedding dimension",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=512,
        help="CLIP output dimension",
    )
    parser.add_argument(
        "--use-cosine",
        action="store_true",
        help="Use cosine similarity instead of dot product",
    )
    parser.add_argument(
        "--use-best",
        action="store_true",
        help="Use best model (clip_best.pt) instead of final model",
    )
    parser.add_argument(
        "--tag-csv",
        default="clip copy/track_tags_filtered.csv",
        help="Track tags CSV",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # Load SimGCL track embeddings
    print("\n=== Loading SimGCL Track Embeddings ===")
    simgcl_path = os.path.join(base_dir, args.simgcl_dir)
    track_embs = np.load(f"{simgcl_path}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{simgcl_path}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = np.array([str(tid) for tid in track_ids])

    print(f"Loaded {len(track_ids):,} tracks")
    print(f"Track embeddings shape: {track_embs.shape}")

    # Load track metadata
    print("\n=== Loading Track Metadata ===")
    track_df = pd.read_csv(os.path.join(base_dir, args.track_csv))
    print(f"Metadata loaded: {len(track_df):,} tracks")

    # Load track tags
    print("\n=== Loading Track Tags ===")
    tag_path = os.path.join(base_dir, args.tag_csv)
    if os.path.exists(tag_path):
        tag_df = pd.read_csv(tag_path)
        print(f"Tags loaded: {len(tag_df):,} tracks with tags")
    else:
        tag_df = None
        print("No tag file found, skipping tags")

    # Load trained model
    print("\n=== Loading CLIP Model (Dot Product) ===")
    model = CaptionPlaylistCLIPDot(
        caption_dim=args.caption_dim,
        playlist_dim=args.track_dim,
        out_dim=args.output_dim,
    ).to(device)

    model_path = os.path.join(base_dir, args.model_dir, "clip_best.pt" if args.use_best else "clip.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Interactive search or single query
    if args.query:
        # Single query mode
        print(f"\n{'='*80}")
        print(f"Searching for: '{args.query}'")
        print(f"Similarity: {'cosine' if args.use_cosine else 'dot product'}")
        print('='*80)

        results = search_tracks_dot(
            args.query,
            model,
            client,
            track_ids,
            track_embs,
            track_df,
            tag_df,
            device,
            top_k=args.top_k,
            use_cosine=args.use_cosine,
        )

        print(f"\n{'='*80}")
        print(f"Top {len(results)} matching tracks:")
        print('='*80)
        for i, (track_id, track_name, artist, score, norm, count, tags) in enumerate(results, 1):
            print(f"{i:2d}. [Score: {score:.4f}] [Norm: {norm:.2f}] [Count: {count}] {track_name}")
            print(f"    Artist: {artist}")
            if tags:
                print(f"    Tags: {', '.join(tags[:10])}")
        print('='*80)
    else:
        # Interactive mode
        print("\n=== Interactive Track Search (Dot Product) ===")
        print(f"Similarity: {'cosine' if args.use_cosine else 'dot product'}")
        print("Enter search queries (or 'quit' to exit, 'cos'/'dot' to switch mode)")
        print("-" * 80)

        use_cosine = args.use_cosine

        while True:
            query = input("\nSearch query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "cos":
                use_cosine = True
                print("Switched to cosine similarity")
                continue
            elif query.lower() == "dot":
                use_cosine = False
                print("Switched to dot product")
                continue

            if not query:
                continue

            print(f"\n{'='*80}")
            print(f"Searching for: '{query}' ({'cosine' if use_cosine else 'dot product'})")
            print('='*80)

            results = search_tracks_dot(
                query,
                model,
                client,
                track_ids,
                track_embs,
                track_df,
                tag_df,
                device,
                top_k=args.top_k,
                use_cosine=use_cosine,
            )

            print(f"\n{'='*80}")
            print(f"Top {min(20, len(results))} matching tracks:")
            print('='*80)
            for i, (track_id, track_name, artist, score, norm, count, tags) in enumerate(results[:20], 1):
                print(f"{i:2d}. [Score: {score:.4f}] [Norm: {norm:.2f}] [Count: {count}] {track_name}")
                print(f"    Artist: {artist}")
                if tags:
                    print(f"    Tags: {', '.join(tags[:10])}")
            if len(results) > 20:
                print(f"... and {len(results) - 20} more tracks")
            print('='*80)


if __name__ == "__main__":
    main()
