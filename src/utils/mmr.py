"""
MMR (Maximal Marginal Relevance) for diverse search results.
"""

import numpy as np


def mmr_rerank(query_emb, candidate_embs, candidate_ids, top_k, lambda_param=0.7):
    """
    MMR (Maximal Marginal Relevance) reranking for diversity.

    MMR = argmax [lambda * sim(q, d) - (1-lambda) * max(sim(d, d_selected))]

    Args:
        query_emb: query embedding (dim,)
        candidate_embs: candidate embeddings (N, dim)
        candidate_ids: IDs of candidates (list)
        top_k: number of results to return
        lambda_param: balance between relevance (1.0) and diversity (0.0)
                     - 1.0 = pure relevance
                     - 0.7 = slight diversity (default)
                     - 0.5 = balanced
                     - 0.0 = pure diversity

    Returns:
        selected_ids: list of selected IDs
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

    # Map back to IDs
    selected_ids = [candidate_ids[i] for i in selected]

    return selected_ids, selected_scores
