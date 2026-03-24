"""
Search tracks using text queries via trained CLIP model (Cosine version).

Uses cosine similarity (L2 normalized embeddings).
Supports MMR (Maximal Marginal Relevance) for diverse results.
"""

import argparse
import ast
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_simgcl.model import CaptionPlaylistCLIP


def embed_text(client, text, model="text-embedding-3-large"):
    """Embed a single text query using OpenAI API"""
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def mmr_rerank(query_emb, candidate_embs, candidate_indices, top_k, lambda_param=0.5):
    """
    MMR (Maximal Marginal Relevance) reranking for diversity.

    MMR = argmax [lambda * sim(q, d) - (1-lambda) * max(sim(d, d_selected))]

    Args:
        query_emb: query embedding (dim,)
        candidate_embs: candidate embeddings (N, dim)
        candidate_indices: original indices of candidates
        top_k: number of results to return
        lambda_param: balance between relevance (1.0) and diversity (0.0)
                     - 1.0 = pure relevance (no diversity)
                     - 0.0 = pure diversity (ignore relevance)
                     - 0.5 = balanced (default)

    Returns:
        selected_indices: indices in original order
        mmr_scores: MMR scores for each selected item
    """
    if len(candidate_indices) == 0:
        return [], []

    # Normalize embeddings for cosine similarity
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
    candidate_embs_norm = candidate_embs / norms

    # Query-document similarities
    query_sims = candidate_embs_norm @ query_emb  # (N,)

    # Document-document similarity matrix
    doc_sims = candidate_embs_norm @ candidate_embs_norm.T  # (N, N)

    selected = []
    selected_mask = np.zeros(len(candidate_indices), dtype=bool)
    mmr_scores = []

    for _ in range(min(top_k, len(candidate_indices))):
        if len(selected) == 0:
            # First selection: highest relevance
            best_idx = np.argmax(query_sims)
        else:
            # MMR score for remaining candidates
            remaining_mask = ~selected_mask
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) == 0:
                break

            # Max similarity to already selected documents
            selected_indices_local = np.array(selected)
            max_sim_to_selected = np.max(doc_sims[remaining_indices][:, selected_indices_local], axis=1)

            # MMR = lambda * relevance - (1-lambda) * max_similarity_to_selected
            mmr = lambda_param * query_sims[remaining_indices] - (1 - lambda_param) * max_sim_to_selected

            best_local_idx = np.argmax(mmr)
            best_idx = remaining_indices[best_local_idx]

        selected.append(best_idx)
        selected_mask[best_idx] = True
        mmr_scores.append(float(query_sims[best_idx]))

    # Map back to original indices
    selected_original = [candidate_indices[i] for i in selected]

    return selected_original, mmr_scores


def search_tracks(
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
    use_mmr=False,
    mmr_lambda=0.5,
    mmr_candidates=200,
):
    """
    Search for tracks matching the query text using cosine similarity.

    Args:
        query_text: search query string
        model: trained CLIP model (cosine version)
        client: OpenAI client
        track_ids: list of track IDs
        track_embs: SimGCL track embeddings (N, 64)
        track_df: dataframe with track metadata
        tag_df: dataframe with track tags (or None)
        device: torch device
        embedding_model: OpenAI embedding model to use
        top_k: number of results to return
        use_mmr: whether to use MMR for diverse results
        mmr_lambda: MMR lambda (1.0=relevance, 0.0=diversity, 0.5=balanced)
        mmr_candidates: number of candidates to consider for MMR reranking

    Returns:
        list of (track_id, track_name, artist, score, count, tags) tuples
    """
    # 1. Embed query text
    print(f"Embedding query: '{query_text}'")
    query_emb = embed_text(client, query_text, model=embedding_model)
    query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Project query through caption encoder (already L2 normalized)
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

            # Project track embeddings through playlist projection (already L2 normalized)
            projected = model.playlist_proj(batch_embs)  # (batch, out_dim)
            all_projected.append(projected)

    all_projected = torch.cat(all_projected, dim=0)  # (N, out_dim)
    all_projected_np = all_projected.cpu().numpy()
    projected_query_np = projected_query.squeeze(0).cpu().numpy()

    # 4. Compute cosine similarities (embeddings are already normalized)
    print("Computing cosine similarities...")
    with torch.no_grad():
        similarities = torch.matmul(projected_query, all_projected.T).squeeze(0)

        if use_mmr:
            # Get more candidates for MMR reranking
            n_candidates = min(mmr_candidates, len(track_ids))
            _, candidate_indices = torch.topk(similarities, k=n_candidates)
            candidate_indices = candidate_indices.cpu().numpy()

            # MMR reranking
            print(f"Applying MMR (lambda={mmr_lambda}, candidates={n_candidates})...")
            candidate_embs = all_projected_np[candidate_indices]
            selected_indices, mmr_scores = mmr_rerank(
                projected_query_np,
                candidate_embs,
                candidate_indices,
                top_k=top_k,
                lambda_param=mmr_lambda
            )
            top_indices = np.array(selected_indices)
            scores = np.array(mmr_scores)
        else:
            scores, top_indices = torch.topk(similarities, k=min(top_k, len(track_ids)))
            scores = scores.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

    # 5. Gather results with metadata
    results = []
    for score, idx in zip(scores, top_indices):
        track_id = str(track_ids[idx])

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

        results.append((track_id, track_name, artist, float(score), count, tags))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="clip_simgcl/exp_text_fast",
        help="Directory containing trained cosine CLIP model",
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
        "--use-best",
        action="store_true",
        help="Use best model (clip_best.pt) instead of final model",
    )
    parser.add_argument(
        "--tag-csv",
        default="clip copy/track_tags_filtered.csv",
        help="Track tags CSV",
    )
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Use MMR (Maximal Marginal Relevance) for diverse results",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.5,
        help="MMR lambda: 1.0=relevance only, 0.0=diversity only, 0.5=balanced (default)",
    )
    parser.add_argument(
        "--mmr-candidates",
        type=int,
        default=200,
        help="Number of candidates for MMR reranking (default: 200)",
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
    print("\n=== Loading CLIP Model (Cosine) ===")
    model = CaptionPlaylistCLIP(
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
        print('='*80)

        results = search_tracks(
            args.query,
            model,
            client,
            track_ids,
            track_embs,
            track_df,
            tag_df,
            device,
            top_k=args.top_k,
            use_mmr=args.mmr,
            mmr_lambda=args.mmr_lambda,
            mmr_candidates=args.mmr_candidates,
        )

        print(f"\n{'='*80}")
        mode = f"MMR (lambda={args.mmr_lambda})" if args.mmr else "Cosine"
        print(f"Top {len(results)} matching tracks [{mode}]:")
        print('='*80)
        for i, (track_id, track_name, artist, score, count, tags) in enumerate(results, 1):
            print(f"{i:2d}. [Score: {score:.4f}] [Count: {count}] {track_name}")
            print(f"    Artist: {artist}")
            if tags:
                print(f"    Tags: {', '.join(tags[:10])}")
        print('='*80)
    else:
        # Interactive mode
        use_mmr = args.mmr
        mmr_lambda = args.mmr_lambda

        print("\n=== Interactive Track Search ===")
        print(f"Mode: {'MMR' if use_mmr else 'Cosine'} (lambda={mmr_lambda})")
        print("Commands: 'quit' to exit, 'mmr' to toggle MMR, 'lambda X' to set lambda")
        print("-" * 80)

        while True:
            query = input("\nSearch query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower() == "mmr":
                use_mmr = not use_mmr
                print(f"MMR {'enabled' if use_mmr else 'disabled'}")
                continue

            if query.lower().startswith("lambda "):
                try:
                    mmr_lambda = float(query.split()[1])
                    mmr_lambda = max(0.0, min(1.0, mmr_lambda))
                    print(f"MMR lambda set to {mmr_lambda}")
                except:
                    print("Invalid lambda value. Use: lambda 0.5")
                continue

            if not query:
                continue

            print(f"\n{'='*80}")
            mode = f"MMR (lambda={mmr_lambda})" if use_mmr else "Cosine"
            print(f"Searching for: '{query}' [{mode}]")
            print('='*80)

            results = search_tracks(
                query,
                model,
                client,
                track_ids,
                track_embs,
                track_df,
                tag_df,
                device,
                top_k=args.top_k,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                mmr_candidates=args.mmr_candidates,
            )

            print(f"\n{'='*80}")
            print(f"Top {min(20, len(results))} matching tracks [{mode}]:")
            print('='*80)
            for i, (track_id, track_name, artist, score, count, tags) in enumerate(results[:20], 1):
                print(f"{i:2d}. [Score: {score:.4f}] [Count: {count}] {track_name}")
                print(f"    Artist: {artist}")
                if tags:
                    print(f"    Tags: {', '.join(tags[:10])}")
            if len(results) > 20:
                print(f"... and {len(results) - 20} more tracks")
            print('='*80)


if __name__ == "__main__":
    main()
