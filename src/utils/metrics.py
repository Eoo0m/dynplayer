"""
Evaluation metrics for recommendation systems.
"""

import numpy as np


def recall_at_k(predictions, targets, k):
    """
    Compute Recall@K.
    
    Args:
        predictions: list of predicted item indices (sorted by score)
        targets: set of ground truth item indices
        k: top-k to consider
        
    Returns:
        recall score
    """
    top_k = set(predictions[:k])
    hits = len(top_k & targets)
    return hits / len(targets) if len(targets) > 0 else 0.0


def ndcg_at_k(predictions, targets, k):
    """
    Compute NDCG@K.
    
    Args:
        predictions: list of predicted item indices (sorted by score)
        targets: set of ground truth item indices
        k: top-k to consider
        
    Returns:
        NDCG score
    """
    top_k = predictions[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in targets:
            dcg += 1.0 / np.log2(i + 2)
    
    # Ideal DCG
    n_relevant = min(len(targets), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(predictions, target, k):
    """
    Compute Hit Rate@K (for single target item, e.g., LOO evaluation).
    
    Args:
        predictions: list of predicted item indices (sorted by score)
        target: single ground truth item index
        k: top-k to consider
        
    Returns:
        1 if target in top-k, else 0
    """
    return 1 if target in predictions[:k] else 0
