"""
Analyze embedding distance distributions between popular and unpopular tracks.

Compare simgcl_weighted vs simgcl_randneg to see how random negatives
affect the embedding distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def load_embeddings(model_dir):
    """Load track embeddings."""
    track_embs = np.load(f"{model_dir}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{model_dir}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = [str(t) for t in track_ids]
    track_id_to_idx = {t: i for i, t in enumerate(track_ids)}
    return track_embs, track_ids, track_id_to_idx


def compute_alignment(track_embs, track_id_to_idx, train_dict, idx_to_track, n_samples=10000):
    """
    Compute alignment loss from "Understanding Contrastive Representation Learning"

    L_align = E[||f(x) - f(y)||^2] for positive pairs

    where f(x) are L2-normalized embeddings.
    Lower alignment = positive pairs are closer (better).

    Uses co-occurring tracks in same playlist as positive pairs.
    """
    import random

    # Normalize embeddings
    embs_norm = track_embs / (np.linalg.norm(track_embs, axis=1, keepdims=True) + 1e-10)

    # Build track co-occurrence from train_dict
    # Sample positive pairs: tracks that appear in same playlist
    positive_pairs = []

    playlist_list = list(train_dict.keys())
    random.shuffle(playlist_list)

    for playlist_idx in playlist_list:
        track_indices = train_dict[playlist_idx]
        if len(track_indices) < 2:
            continue

        # Get track_ids that have embeddings
        valid_tracks = []
        for ti in track_indices:
            tid = idx_to_track[ti]
            if tid in track_id_to_idx:
                valid_tracks.append(track_id_to_idx[tid])

        if len(valid_tracks) >= 2:
            # Sample pairs from this playlist
            for _ in range(min(5, len(valid_tracks))):
                if len(valid_tracks) >= 2:
                    i, j = random.sample(valid_tracks, 2)
                    positive_pairs.append((i, j))

        if len(positive_pairs) >= n_samples:
            break

    positive_pairs = positive_pairs[:n_samples]

    # Compute alignment
    sq_distances = []
    for i, j in positive_pairs:
        diff = embs_norm[i] - embs_norm[j]
        sq_distances.append(np.sum(diff ** 2))

    alignment = np.mean(sq_distances)
    return alignment


def compute_uniformity(embeddings, t=2, n_samples=10000):
    """
    Compute uniformity loss from "Understanding Contrastive Representation Learning"

    L_uniform = log E[exp(-t * ||f(x) - f(y)||^2)]

    where f(x) are L2-normalized embeddings.
    Lower uniformity = more uniform distribution on hypersphere (better).

    Args:
        embeddings: [N, dim] embeddings (will be normalized)
        t: temperature parameter (default: 2)
        n_samples: number of pairs to sample

    Returns:
        uniformity: scalar uniformity loss
    """
    # Normalize embeddings
    embs_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    n = len(embs_norm)

    # Sample random pairs
    idx1 = np.random.choice(n, n_samples, replace=True)
    idx2 = np.random.choice(n, n_samples, replace=True)

    # Compute squared L2 distances
    diff = embs_norm[idx1] - embs_norm[idx2]
    sq_distances = np.sum(diff ** 2, axis=1)

    # Compute uniformity: log E[exp(-t * ||x - y||^2)]
    uniformity = np.log(np.mean(np.exp(-t * sq_distances)))

    return uniformity


def main():
    print("=== Loading Track Popularity ===")
    track_meta_df = pd.read_csv("data/csvs/track_playlist_counts_min5_win10.csv")
    track_meta_df["track_id"] = track_meta_df["track_id"].astype(str)
    track_popularity = dict(zip(track_meta_df["track_id"], track_meta_df["count"]))
    print(f"Tracks: {len(track_popularity):,}")

    # Calculate thresholds (top/bottom 10%)
    pop_values = sorted(track_popularity.values(), reverse=True)
    n = len(pop_values)
    popular_threshold = pop_values[int(n * 0.1)]
    unpopular_threshold = pop_values[int(n * 0.9)]
    print(f"Popular threshold (top 10%): >= {popular_threshold}")
    print(f"Unpopular threshold (bottom 10%): <= {unpopular_threshold}")

    models = {
        "lightgcn": "models/lightgcn_loo/outputs/min5_win10",
        "simgcl_weighted": "models/simgcl_weighted/outputs/min5_win10",
        "simgcl_randneg": "models/simgcl_randneg/outputs/min5_win10",
    }

    results = {}
    n_samples = 5000
    random.seed(42)

    for model_name, model_dir in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        track_embs, track_ids, track_id_to_idx = load_embeddings(model_dir)
        print(f"Track embeddings: {track_embs.shape}")

        # Normalize embeddings
        track_embs_norm = track_embs / (np.linalg.norm(track_embs, axis=1, keepdims=True) + 1e-10)

        # Separate popular and unpopular tracks
        popular_indices = []
        unpopular_indices = []

        for tid, idx in track_id_to_idx.items():
            pop = track_popularity.get(tid, 0)
            if pop >= popular_threshold:
                popular_indices.append(idx)
            elif pop <= unpopular_threshold:
                unpopular_indices.append(idx)

        print(f"Popular tracks: {len(popular_indices):,}")
        print(f"Unpopular tracks: {len(unpopular_indices):,}")

        # Sample pairs and calculate distances
        def sample_pairwise_distances(indices, n_samples, embs_norm):
            """Sample pairs and calculate cosine distances."""
            distances = []
            for _ in range(n_samples):
                i, j = random.sample(indices, 2)
                # Cosine distance = 1 - cosine_similarity
                cos_sim = np.dot(embs_norm[i], embs_norm[j])
                distances.append(1 - cos_sim)
            return distances

        print("\nSampling pairwise distances...")

        # Popular-Popular distances
        pop_pop_dist = sample_pairwise_distances(popular_indices, n_samples, track_embs_norm)

        # Unpopular-Unpopular distances
        unpop_unpop_dist = sample_pairwise_distances(unpopular_indices, n_samples, track_embs_norm)

        # Popular-Unpopular distances (cross)
        cross_dist = []
        for _ in range(n_samples):
            i = random.choice(popular_indices)
            j = random.choice(unpopular_indices)
            cos_sim = np.dot(track_embs_norm[i], track_embs_norm[j])
            cross_dist.append(1 - cos_sim)

        # Calculate norms
        pop_norms = [np.linalg.norm(track_embs[i]) for i in popular_indices]
        unpop_norms = [np.linalg.norm(track_embs[i]) for i in unpopular_indices]

        # Compute uniformity for popular and unpopular tracks
        pop_embs = track_embs[popular_indices]
        unpop_embs = track_embs[unpopular_indices]
        all_embs = track_embs

        pop_uniformity = compute_uniformity(pop_embs, t=2, n_samples=10000)
        unpop_uniformity = compute_uniformity(unpop_embs, t=2, n_samples=10000)
        all_uniformity = compute_uniformity(all_embs, t=2, n_samples=10000)

        # Load LOO split for alignment calculation
        split = np.load(f"{model_dir}/loo_split.npz", allow_pickle=True)
        train_dict = split["train_dict"].item()
        idx_to_track = {v: k for k, v in split["track_to_idx"].item().items()}

        # Compute alignment
        alignment = compute_alignment(track_embs, track_id_to_idx, train_dict, idx_to_track, n_samples=10000)

        results[model_name] = {
            "pop_pop": pop_pop_dist,
            "unpop_unpop": unpop_unpop_dist,
            "cross": cross_dist,
            "pop_norms": pop_norms,
            "unpop_norms": unpop_norms,
            "pop_uniformity": pop_uniformity,
            "unpop_uniformity": unpop_uniformity,
            "all_uniformity": all_uniformity,
            "alignment": alignment,
        }

        print(f"\nPopular-Popular distance:     mean={np.mean(pop_pop_dist):.4f}, std={np.std(pop_pop_dist):.4f}")
        print(f"Unpopular-Unpopular distance: mean={np.mean(unpop_unpop_dist):.4f}, std={np.std(unpop_unpop_dist):.4f}")
        print(f"Popular-Unpopular distance:   mean={np.mean(cross_dist):.4f}, std={np.std(cross_dist):.4f}")
        print(f"\nPopular track norms:   mean={np.mean(pop_norms):.4f}, std={np.std(pop_norms):.4f}")
        print(f"Unpopular track norms: mean={np.mean(unpop_norms):.4f}, std={np.std(unpop_norms):.4f}")
        print(f"\nUniformity (lower = better):")
        print(f"  All tracks:      {all_uniformity:.4f}")
        print(f"  Popular tracks:  {pop_uniformity:.4f}")
        print(f"  Unpopular tracks: {unpop_uniformity:.4f}")
        print(f"\nAlignment (lower = better): {alignment:.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {"lightgcn": "green", "simgcl_weighted": "blue", "simgcl_randneg": "orange"}

    # Row 1: Distance distributions
    for col, (dist_type, title) in enumerate([
        ("pop_pop", "Popular-Popular"),
        ("unpop_unpop", "Unpopular-Unpopular"),
        ("cross", "Popular-Unpopular")
    ]):
        ax = axes[0, col]
        for model_name in models.keys():
            data = results[model_name][dist_type]
            ax.hist(data, bins=50, alpha=0.5, label=model_name, color=colors[model_name], density=True)
        ax.set_xlabel("Cosine Distance (1 - similarity)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} Distance")
        ax.legend()

    # Row 2: Norms and summary
    # Norm distributions
    ax = axes[1, 0]
    for model_name in models.keys():
        ax.hist(results[model_name]["pop_norms"], bins=50, alpha=0.5,
                label=f"{model_name} (popular)", color=colors[model_name], density=True)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Popular Track Norms")
    ax.legend()

    ax = axes[1, 1]
    for model_name in models.keys():
        ax.hist(results[model_name]["unpop_norms"], bins=50, alpha=0.5,
                label=f"{model_name} (unpopular)", color=colors[model_name], density=True)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Unpopular Track Norms")
    ax.legend()

    # Alignment-Uniformity 2D plot
    ax = axes[1, 2]
    for model_name in models.keys():
        alignment = results[model_name]["alignment"]
        uniformity = results[model_name]["all_uniformity"]
        ax.scatter(uniformity, alignment, s=150, c=colors[model_name], label=model_name, marker='o', edgecolors='black')
        ax.annotate(model_name, (uniformity, alignment), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Uniformity (lower = better)")
    ax.set_ylabel("Alignment (lower = better)")
    ax.set_title("Alignment vs Uniformity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Embedding Analysis: LightGCN vs SimGCL variants", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("eval/popularity/embedding_distance_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to: eval/popularity/embedding_distance_comparison.png")

    # Create separate Alignment-Uniformity figure
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    markers = {"lightgcn": "s", "simgcl_weighted": "o", "simgcl_randneg": "^"}

    for model_name in models.keys():
        alignment = results[model_name]["alignment"]
        uniformity = results[model_name]["all_uniformity"]
        ax2.scatter(uniformity, alignment, s=200, c=colors[model_name], label=model_name,
                   marker=markers[model_name], edgecolors='black', linewidths=1.5)

    ax2.set_xlabel("Uniformity (lower = more uniform)", fontsize=12)
    ax2.set_ylabel("Alignment (lower = better alignment)", fontsize=12)
    ax2.set_title("Alignment vs Uniformity Trade-off", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add arrow indicating "better" direction
    ax2.annotate("", xy=(ax2.get_xlim()[0], ax2.get_ylim()[0]),
                xytext=(ax2.get_xlim()[0] + 0.1, ax2.get_ylim()[0] + 0.1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax2.text(ax2.get_xlim()[0] + 0.02, ax2.get_ylim()[0] + 0.02, "Better", fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig("eval/popularity/alignment_uniformity.png", dpi=150, bbox_inches="tight")
    print(f"Saved to: eval/popularity/alignment_uniformity.png")

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Metric':<30} | {'lightgcn':<18} | {'simgcl_weighted':<18} | {'simgcl_randneg':<18}")
    print("-"*100)

    for metric, key in [
        ("Pop-Pop Distance", "pop_pop"),
        ("Unpop-Unpop Distance", "unpop_unpop"),
        ("Pop-Unpop Distance", "cross"),
    ]:
        l_mean = np.mean(results["lightgcn"][key])
        w_mean = np.mean(results["simgcl_weighted"][key])
        r_mean = np.mean(results["simgcl_randneg"][key])
        print(f"{metric:<30} | {l_mean:.4f}            | {w_mean:.4f}            | {r_mean:.4f}")

    print("-"*100)

    for metric, key in [
        ("Popular Norm", "pop_norms"),
        ("Unpopular Norm", "unpop_norms"),
    ]:
        l_mean = np.mean(results["lightgcn"][key])
        w_mean = np.mean(results["simgcl_weighted"][key])
        r_mean = np.mean(results["simgcl_randneg"][key])
        print(f"{metric:<30} | {l_mean:.4f}            | {w_mean:.4f}            | {r_mean:.4f}")

    print("-"*100)
    print("UNIFORMITY & ALIGNMENT (lower = better)")
    print("-"*100)

    for metric, key in [
        ("All Tracks Uniformity", "all_uniformity"),
        ("Popular Tracks Uniformity", "pop_uniformity"),
        ("Unpopular Tracks Uniformity", "unpop_uniformity"),
        ("Alignment", "alignment"),
    ]:
        l_val = results["lightgcn"][key]
        w_val = results["simgcl_weighted"][key]
        r_val = results["simgcl_randneg"][key]
        print(f"{metric:<30} | {l_val:.4f}            | {w_val:.4f}            | {r_val:.4f}")


if __name__ == "__main__":
    main()
