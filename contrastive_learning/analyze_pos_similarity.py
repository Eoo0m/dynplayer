"""
u00 / u05 / u10 3개의 embedding에 대해
Popular/Unpopular intra-group similarity를 한 번에 비교하는 그래프
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# 공통 함수
# ===============================


def compute_avg_similarity(embs):
    """그룹 내 전체 pairwise cosine similarity 계산"""
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = embs_norm @ embs_norm.T
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    return sim_matrix[mask]


# ===============================
# 메타 + Embedding 정의
# ===============================

meta_path = "lightgcn/contrastive_top1pct_basic_meta.csv"

embedding_sets = {
    "u00 (no uniformity loss)": {
        "emb": "/Users/eomjoonseo/projects/dynplayer-1/contrastive_learning/u00_embeddings.npy",
        "keys": "/Users/eomjoonseo/projects/dynplayer-1/contrastive_learning/u00_keys.npy",
    },
    "u05 (weight=0.5)": {
        "emb": "contrastive_learning/u05_embeddings.npy",
        "keys": "contrastive_learning/u05_keys.npy",
    },
    "u10 (weight=1.0)": {
        "emb": "contrastive_learning/u10_embeddings.npy",
        "keys": "contrastive_learning/u10_keys.npy",
    },
}

meta_df = pd.read_csv(meta_path)

# ===============================
# 각 실험 결과 계산
# ===============================

results = {}

for tag, paths in embedding_sets.items():

    # ---- 임베딩 로드 ----
    track_ids = np.load(paths["keys"], allow_pickle=True)
    track_embs = np.load(paths["emb"])

    track_index = {tid: i for i, tid in enumerate(track_ids)}

    df = meta_df[meta_df["track_id"].isin(track_index)].copy()
    df["emb_idx"] = df["track_id"].map(track_index)

    popular_df = df[df["pos_count"] >= 700]
    unpopular_df = df[df["pos_count"] <= 8]

    pop_embs = track_embs[popular_df["emb_idx"].values]
    unpop_embs = track_embs[unpopular_df["emb_idx"].values]

    pop_sims = compute_avg_similarity(pop_embs)
    unpop_sims = compute_avg_similarity(unpop_embs)

    results[tag] = (pop_sims, unpop_sims)


# ===============================
# 그래프 그리기 (한 화면에 3개)
# ===============================

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (tag, (pop_sims, unpop_sims)) in zip(axes, results.items()):

    vp = ax.violinplot(
        [pop_sims, unpop_sims],
        positions=[1, 2],
        showmeans=True,
        showmedians=True,
    )

    for pc in vp["bodies"]:
        pc.set_facecolor("#D43F3A")
        pc.set_alpha(0.65)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Popular\n(≥700)", "Unpopular\n(≤8)"], fontsize=11)
    ax.set_title(tag, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # mean annotation
    ax.text(
        1,
        pop_sims.max() * 1.02,
        f"mean={pop_sims.mean():.3f}",
        ha="center",
        fontsize=10,
    )
    ax.text(
        2,
        unpop_sims.max() * 1.02,
        f"mean={unpop_sims.mean():.3f}",
        ha="center",
        fontsize=10,
    )

axes[0].set_ylabel("Cosine Similarity", fontsize=12)

plt.suptitle(
    "Intra-group Similarity Comparison (Uniformity Loss Variants)",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("similarity_violin_all_three.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Saved: similarity_violin_all_three.png")
