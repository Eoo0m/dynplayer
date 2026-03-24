"""
Evaluate Word2Vec embeddings using LOO (Leave-One-Out) test set.

Computes Recall@K and NDCG@K metrics, comparing with other embedding methods.
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


def load_embeddings(output_dir):
    """Load Word2Vec embeddings."""
    output_dir = Path(output_dir)

    embeddings = np.load(output_dir / "track_embeddings.npy")
    track_ids = np.load(output_dir / "track_ids.npy", allow_pickle=True)

    with open(output_dir / "track_to_idx.pkl", 'rb') as f:
        track_to_idx = pickle.load(f)

    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings, track_ids, track_to_idx


def load_loo_split(split_path):
    """Load LOO split data."""
    split = np.load(split_path, allow_pickle=True)

    train_dict = split["train_dict"].item()
    test_dict = split["test_dict"].item()
    num_users = int(split["num_users"])
    num_items = int(split["num_items"])
    loo_track_to_idx = split["track_to_idx"].item()

    print(f"LOO split: {num_users:,} users, {num_items:,} items")
    return train_dict, test_dict, num_users, num_items, loo_track_to_idx


def align_embeddings(w2v_embeddings, w2v_track_ids, w2v_track_to_idx, loo_track_to_idx, num_items):
    """
    Align Word2Vec embeddings to LOO split indices.

    Word2Vec may have different track ordering than LOO split.
    Creates aligned embedding matrix where index matches LOO track_to_idx.
    """
    # Create reverse mapping for LOO
    loo_idx_to_track = {v: k for k, v in loo_track_to_idx.items()}

    embedding_dim = w2v_embeddings.shape[1]
    aligned_embeddings = np.zeros((num_items, embedding_dim), dtype=np.float32)
    found_count = 0

    for loo_idx in range(num_items):
        track_id = loo_idx_to_track.get(loo_idx)
        if track_id and track_id in w2v_track_to_idx:
            w2v_idx = w2v_track_to_idx[track_id]
            aligned_embeddings[loo_idx] = w2v_embeddings[w2v_idx]
            found_count += 1

    print(f"Aligned {found_count:,} / {num_items:,} tracks ({100*found_count/num_items:.1f}%)")
    return aligned_embeddings


def evaluate_recall(train_dict, test_dict, embeddings, k_list=[10, 20, 50], use_cosine=True):
    """
    Evaluate Recall@K and NDCG@K.

    For each user:
    1. Compute user embedding as average of train track embeddings
    2. Score all tracks by similarity to user embedding
    3. Exclude train tracks and compute Recall/NDCG for test tracks
    """
    # Normalize embeddings for cosine similarity
    if use_cosine:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms

    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    users = list(test_dict.keys())

    for user in tqdm(users, desc="Evaluating"):
        if user not in train_dict:
            continue

        train_items = train_dict[user]
        test_items = test_dict[user]

        if len(train_items) == 0 or len(test_items) == 0:
            continue

        # Skip if any train items have zero embeddings (not in W2V vocab)
        train_embs = embeddings[train_items]
        valid_mask = np.linalg.norm(train_embs, axis=1) > 0
        if valid_mask.sum() == 0:
            continue

        # User embedding = average of train track embeddings
        user_emb = train_embs[valid_mask].mean(axis=0)
        if use_cosine:
            user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-10)

        # Compute scores
        scores = embeddings @ user_emb

        # Mask train items
        for ti in train_items:
            scores[ti] = -np.inf

        # Get top-k
        max_k = max(k_list)
        topk = np.argpartition(scores, -max_k)[-max_k:]
        topk = topk[np.argsort(scores[topk])[::-1]]

        # Compute metrics
        test_set = set(test_items)
        rel = np.array([1 if i in test_set else 0 for i in topk])

        for k in k_list:
            r = rel[:k]
            recall = r.sum() / len(test_items)

            # NDCG
            dcg = (r / np.log2(np.arange(2, len(r) + 2))).sum()
            n_relevant = min(len(test_items), k)
            idcg = (np.ones(n_relevant) / np.log2(np.arange(2, n_relevant + 2))).sum()
            ndcg = dcg / idcg if idcg > 0 else 0.0

            recalls[k].append(recall)
            ndcgs[k].append(ndcg)

    # Compute averages
    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Word2Vec embeddings")
    parser.add_argument("--embeddings-dir", type=str, default="word2vec/outputs",
                        help="Directory with Word2Vec embeddings")
    parser.add_argument("--loo-split", type=str,
                        default="data/loo_split_min5_win10.npz",
                        help="Path to LOO split")
    parser.add_argument("--k", nargs="+", type=int, default=[10, 20, 50],
                        help="K values for Recall/NDCG")
    parser.add_argument("--use-cosine", action="store_true", default=True,
                        help="Use cosine similarity")
    args = parser.parse_args()

    print("=== Word2Vec Evaluation ===")

    # Load embeddings
    embeddings, track_ids, track_to_idx = load_embeddings(args.embeddings_dir)

    # Load LOO split
    train_dict, test_dict, num_users, num_items, loo_track_to_idx = load_loo_split(args.loo_split)

    # Align embeddings
    aligned_embeddings = align_embeddings(
        embeddings, track_ids, track_to_idx,
        loo_track_to_idx, num_items
    )

    # Evaluate
    print(f"\nSimilarity: {'Cosine' if args.use_cosine else 'Dot product'}")
    metrics = evaluate_recall(train_dict, test_dict, aligned_embeddings, args.k, args.use_cosine)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for k in args.k:
        print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}, NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")


if __name__ == "__main__":
    main()
