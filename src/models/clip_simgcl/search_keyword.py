"""
Backend keyword search with MMR (Maximal Marginal Relevance).

Searches tracks using keyword text, applies MMR for diversity.
Uses projected embeddings from Supabase track_embeddings table.

Usage:
    python clip_simgcl/search_keyword.py --keyword "jazz" --top-k 20 --mmr-lambda 0.7
"""

import os
import sys
import argparse
import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_simgcl.model import CaptionPlaylistCLIP

load_dotenv()


def embed_text(client, text, model="text-embedding-3-large"):
    """Embed text using OpenAI API"""
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def mmr_rerank(query_emb, candidate_embs, candidate_ids, top_k, lambda_param=0.7):
    """
    MMR (Maximal Marginal Relevance) reranking for diversity.

    MMR = argmax [lambda * sim(q, d) - (1-lambda) * max(sim(d, d_selected))]

    Args:
        query_emb: query embedding (dim,)
        candidate_embs: candidate embeddings (N, dim)
        candidate_ids: track IDs of candidates
        top_k: number of results to return
        lambda_param: balance between relevance (1.0) and diversity (0.0)
                     - 1.0 = pure relevance
                     - 0.7 = slight diversity (default)
                     - 0.5 = balanced
                     - 0.0 = pure diversity

    Returns:
        selected_ids: list of selected track IDs
        selected_scores: list of relevance scores
    """
    if len(candidate_ids) == 0:
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
    selected_mask = np.zeros(len(candidate_ids), dtype=bool)
    selected_scores = []

    for _ in range(min(top_k, len(candidate_ids))):
        if len(selected) == 0:
            # First selection: highest relevance
            best_idx = int(np.argmax(query_sims))
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

            best_local_idx = int(np.argmax(mmr))
            best_idx = remaining_indices[best_local_idx]

        selected.append(best_idx)
        selected_mask[best_idx] = True
        selected_scores.append(float(query_sims[best_idx]))

    # Map back to track IDs
    selected_ids = [candidate_ids[i] for i in selected]

    return selected_ids, selected_scores


def search_keyword(
    keyword: str,
    model,
    openai_client,
    supabase_client,
    device,
    top_k: int = 20,
    mmr_lambda: float = 0.7,
    mmr_candidates: int = 200,
):
    """
    Search tracks by keyword with MMR diversity.

    Args:
        keyword: search keyword
        model: CLIP model
        openai_client: OpenAI client
        supabase_client: Supabase client
        device: torch device
        top_k: number of results to return
        mmr_lambda: MMR lambda (0.7 = slight diversity)
        mmr_candidates: number of candidates for MMR reranking

    Returns:
        list of (track_id, score) tuples
    """
    # 1. Embed keyword
    keyword_emb = embed_text(openai_client, keyword)
    keyword_tensor = torch.tensor(keyword_emb, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Project through CLIP caption encoder
    model.eval()
    with torch.no_grad():
        projected_keyword = model.caption_proj(keyword_tensor)
        projected_keyword_np = projected_keyword.squeeze(0).cpu().numpy()

    # Normalize
    projected_keyword_np = projected_keyword_np / (np.linalg.norm(projected_keyword_np) + 1e-10)

    # 3. Fetch track embeddings from Supabase
    # Get all tracks with projected embeddings
    result = supabase_client.table('track_embeddings').select(
        'track_key, projected_embedding'
    ).not_.is_('projected_embedding', 'null').execute()

    tracks = result.data
    if not tracks:
        print("No tracks with projected embeddings found")
        return []

    # Build arrays
    track_ids = []
    track_embs = []
    for t in tracks:
        proj = t.get('projected_embedding')
        if proj:
            track_ids.append(t['track_key'])
            if isinstance(proj, str):
                import json
                proj = json.loads(proj)
            track_embs.append(np.array(proj))

    track_embs = np.array(track_embs)

    # 4. Compute similarities
    norms = np.linalg.norm(track_embs, axis=1, keepdims=True) + 1e-10
    track_embs_norm = track_embs / norms
    similarities = track_embs_norm @ projected_keyword_np

    # 5. Get top candidates for MMR
    n_candidates = min(mmr_candidates, len(track_ids))
    top_indices = np.argsort(similarities)[::-1][:n_candidates]

    candidate_ids = [track_ids[i] for i in top_indices]
    candidate_embs = track_embs[top_indices]

    # 6. Apply MMR
    selected_ids, selected_scores = mmr_rerank(
        projected_keyword_np,
        candidate_embs,
        candidate_ids,
        top_k=top_k,
        lambda_param=mmr_lambda
    )

    return list(zip(selected_ids, selected_scores))


def search_keyword_with_local_embeddings(
    keyword: str,
    model,
    openai_client,
    track_ids: np.ndarray,
    track_projected: np.ndarray,
    device,
    top_k: int = 20,
    mmr_lambda: float = 0.7,
    mmr_candidates: int = 200,
):
    """
    Search tracks by keyword with MMR (using local embeddings).

    For faster testing without DB calls.
    """
    # 1. Embed keyword
    keyword_emb = embed_text(openai_client, keyword)
    keyword_tensor = torch.tensor(keyword_emb, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Project through CLIP caption encoder
    model.eval()
    with torch.no_grad():
        projected_keyword = model.caption_proj(keyword_tensor)
        projected_keyword_np = projected_keyword.squeeze(0).cpu().numpy()

    # Normalize
    projected_keyword_np = projected_keyword_np / (np.linalg.norm(projected_keyword_np) + 1e-10)

    # 3. Compute similarities
    norms = np.linalg.norm(track_projected, axis=1, keepdims=True) + 1e-10
    track_embs_norm = track_projected / norms
    similarities = track_embs_norm @ projected_keyword_np

    # 4. Get top candidates for MMR
    n_candidates = min(mmr_candidates, len(track_ids))
    top_indices = np.argsort(similarities)[::-1][:n_candidates]

    candidate_ids = [track_ids[i] for i in top_indices]
    candidate_embs = track_projected[top_indices]

    # 5. Apply MMR
    selected_ids, selected_scores = mmr_rerank(
        projected_keyword_np,
        candidate_embs,
        candidate_ids,
        top_k=top_k,
        lambda_param=mmr_lambda
    )

    return list(zip(selected_ids, selected_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True, help="Search keyword")
    parser.add_argument("--top-k", type=int, default=20, help="Number of results")
    parser.add_argument("--mmr-lambda", type=float, default=0.7,
                        help="MMR lambda (1.0=relevance, 0.0=diversity, 0.7=default)")
    parser.add_argument("--mmr-candidates", type=int, default=200,
                        help="Number of candidates for MMR")
    parser.add_argument("--use-local", action="store_true",
                        help="Use local embeddings instead of Supabase")
    parser.add_argument("--model-dir", default="clip_simgcl/exp_individual_tags",
                        help="Model directory")
    args = parser.parse_args()

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # Load CLIP model
    print("Loading CLIP model...")
    model = CaptionPlaylistCLIP(
        caption_dim=3072,
        playlist_dim=64,
        out_dim=512,
    ).to(device)

    model_path = os.path.join(base_dir, args.model_dir, "clip_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\nSearching for: '{args.keyword}'")
    print(f"MMR lambda: {args.mmr_lambda}, candidates: {args.mmr_candidates}")

    if args.use_local:
        # Load local embeddings
        print("Loading local embeddings...")
        track_ids = np.load(os.path.join(base_dir, args.model_dir, "track_ids.npy"), allow_pickle=True)
        track_projected = np.load(os.path.join(base_dir, args.model_dir, "track_projected.npy"))
        print(f"Loaded {len(track_ids)} tracks")

        results = search_keyword_with_local_embeddings(
            args.keyword,
            model,
            openai_client,
            track_ids,
            track_projected,
            device,
            top_k=args.top_k,
            mmr_lambda=args.mmr_lambda,
            mmr_candidates=args.mmr_candidates,
        )
    else:
        # Use Supabase
        print("Fetching from Supabase...")
        supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

        results = search_keyword(
            args.keyword,
            model,
            openai_client,
            supabase_client,
            device,
            top_k=args.top_k,
            mmr_lambda=args.mmr_lambda,
            mmr_candidates=args.mmr_candidates,
        )

    # Print results
    print(f"\n{'='*60}")
    print(f"Top {len(results)} results for '{args.keyword}' [MMR lambda={args.mmr_lambda}]:")
    print('='*60)
    for i, (track_id, score) in enumerate(results, 1):
        print(f"{i:2d}. [Score: {score:.4f}] {track_id}")
    print('='*60)


if __name__ == "__main__":
    main()
