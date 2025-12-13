"""
Compare Positive & Negative Pair Cosine Similarity
for u00 / u05 / u10 embeddings
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


# ======================================================
# Utils
# ======================================================
def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


# ======================================================
# Build pos_sets (train-only)
# ======================================================
def build_track_neighbors_from_train(train_dict, idx2tid):
    pos_sets = defaultdict(set)
    for lst in tqdm(train_dict.values(), desc="Building pos_sets"):
        track_ids = [idx2tid[idx] for idx in lst]
        for i, t1 in enumerate(track_ids):
            for t2 in track_ids[i + 1 :]:
                pos_sets[t1].add(t2)
                pos_sets[t2].add(t1)
    return pos_sets


# ======================================================
# Global pos_mask
# ======================================================
def build_global_pos_mask(pos_sets, tid2idx):
    N = len(tid2idx)
    mask = torch.zeros((N, N), dtype=torch.bool)
    for t, neighs in pos_sets.items():
        i = tid2idx[t]
        for nb in neighs:
            if nb in tid2idx:
                j = tid2idx[nb]
                if i != j:
                    mask[i, j] = True
    return mask


# ======================================================
# Cosine distributions
# ======================================================
def cosine_distributions_sampled(
    embeddings,
    pos_mask,
    num_neg_samples=30000,
    max_pos_pairs=30000,
):
    z = torch.from_numpy(
        embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    ).float()

    N = z.size(0)

    # ------------------
    # Positive pairs
    # ------------------
    pi, pj = torch.where(pos_mask)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]

    if len(pi) > max_pos_pairs:
        perm = torch.randperm(len(pi))[:max_pos_pairs]
        pi, pj = pi[perm], pj[perm]

    pos_cos = torch.sum(z[pi] * z[pj], dim=1)

    # ------------------
    # Negative pairs (RANDOM SAMPLING)
    # ------------------
    ni = torch.randint(0, N, (num_neg_samples,))
    nj = torch.randint(0, N, (num_neg_samples,))

    # self-pair 제거
    keep = ni != nj
    ni, nj = ni[keep], nj[keep]

    # positive 제거
    neg_keep = ~pos_mask[ni, nj]
    ni, nj = ni[neg_keep], nj[neg_keep]

    neg_cos = torch.sum(z[ni] * z[nj], dim=1)

    return pos_cos.numpy(), neg_cos.numpy()


def plot_pos_neg_with_stats(axes, col, name, pos_cos, neg_cos):
    # ---------- POS ----------
    pos_mean = pos_cos.mean()
    pos_std = pos_cos.std()

    axes[0, col].hist(pos_cos, bins=60, density=True, alpha=0.7, color="seagreen")
    axes[0, col].axvline(pos_mean, color="black", linestyle="--", linewidth=2)
    axes[0, col].set_title(f"{name} – POS", fontweight="bold")
    axes[0, col].set_xlim(-1, 1)

    axes[0, col].text(
        0.5,
        -0.25,
        f"mean = {pos_mean:.3f}\nstd = {pos_std:.3f}",
        ha="center",
        va="top",
        transform=axes[0, col].transAxes,
        fontsize=10,
    )

    # ---------- NEG ----------
    neg_mean = neg_cos.mean()
    neg_std = neg_cos.std()

    axes[1, col].hist(neg_cos, bins=60, density=True, alpha=0.7, color="indianred")
    axes[1, col].axvline(neg_mean, color="black", linestyle="--", linewidth=2)
    axes[1, col].set_title(f"{name} – NEG", fontweight="bold")
    axes[1, col].set_xlim(-1, 1)

    axes[1, col].text(
        0.5,
        -0.25,
        f"mean = {neg_mean:.3f}\nstd = {neg_std:.3f}",
        ha="center",
        va="top",
        transform=axes[1, col].transAxes,
        fontsize=10,
    )


# ======================================================
# Main
# ======================================================
def main():
    split = np.load("contrastive_learning/train_test_split.npz", allow_pickle=True)
    train_dict = split["train_dict"].item()
    all_track_ids = split["all_track_ids"]

    idx2tid = {i: tid for i, tid in enumerate(all_track_ids)}
    pos_sets = build_track_neighbors_from_train(train_dict, idx2tid)

    configs = [
        (
            "u00",
            "contrastive_learning/u00_embeddings.npy",
            "contrastive_learning/u00_keys.npy",
        ),
        (
            "u05",
            "contrastive_learning/u05_embeddings.npy",
            "contrastive_learning/u05_keys.npy",
        ),
        (
            "u10",
            "contrastive_learning/u10_embeddings.npy",
            "contrastive_learning/u10_keys.npy",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey="row")

    for i, (name, emb_path, key_path) in enumerate(configs):
        emb = np.load(emb_path)
        tids = np.load(key_path)
        tid2idx = {tid: i for i, tid in enumerate(tids)}

        pos_mask = build_global_pos_mask(pos_sets, tid2idx)

        pos_cos, neg_cos = cosine_distributions_sampled(
            emb,
            pos_mask,
            num_neg_samples=30000,
        )

        plot_pos_neg_with_stats(axes, i, name, pos_cos, neg_cos)

    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlabel("Cosine Similarity")

    plt.tight_layout()
    plt.savefig("pos_neg_cosine_with_stats.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
