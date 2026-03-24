"""
Visualize Alignment and Uniformity for GCN embeddings.

1. Alignment plot: histogram of distances between positive pairs
2. Uniformity plot: KDE on unit circle + angle distribution

Compares SimGCL vs LightGCN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import torch

# Paths
SIMGCL_DIR = "simgcl_weighted/outputs/min5_win10"
LIGHTGCN_DIR = "lightgcn_loo/outputs/min5_win10"


def normalize(x):
    """L2 normalize."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def load_embeddings_with_ids(emb_dir):
    """Load track and playlist embeddings with their IDs."""
    track_embs = np.load(f"{emb_dir}/model_loo_track_embeddings.npy")
    playlist_embs = np.load(f"{emb_dir}/model_loo_playlist_embeddings.npy")
    track_ids = np.load(f"{emb_dir}/model_loo_track_ids.npy", allow_pickle=True)
    playlist_ids = np.load(f"{emb_dir}/model_loo_playlist_ids.npy", allow_pickle=True)

    track_ids = [str(t) for t in track_ids]
    playlist_ids = [str(p) for p in playlist_ids]

    track_index = {t: i for i, t in enumerate(track_ids)}
    playlist_index = {p: i for i, p in enumerate(playlist_ids)}

    return track_embs, playlist_embs, track_index, playlist_index


def load_positive_pairs_from_csv(csv_path):
    """Load positive pairs (playlist_id -> [track_ids]) from CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    pairs = {}
    for _, row in df.iterrows():
        pid = str(row['playlist_id'])
        tracks = row['track_ids'].split('|')
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for t in tracks:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        pairs[pid] = unique

    return pairs


def compute_alignment(track_embs, playlist_embs, track_index, playlist_index, positive_pairs, n_samples=10000):
    """
    Compute alignment: L2 distance between positive pairs (playlist, track).

    For GCN models, positive pair = (playlist_embedding, track_embedding)
    where track is in playlist.
    """
    distances = []
    count = 0

    for pid, track_ids in positive_pairs.items():
        if count >= n_samples:
            break

        if pid not in playlist_index:
            continue

        playlist_emb = playlist_embs[playlist_index[pid]]
        playlist_emb = playlist_emb / (np.linalg.norm(playlist_emb) + 1e-8)

        for tid in track_ids:
            if count >= n_samples:
                break
            if tid not in track_index:
                continue

            track_emb = track_embs[track_index[tid]]
            track_emb = track_emb / (np.linalg.norm(track_emb) + 1e-8)

            # L2 distance
            dist = np.linalg.norm(playlist_emb - track_emb)
            distances.append(dist)
            count += 1

    return np.array(distances)


def project_to_2d(embeddings, n_samples=5000):
    """Project embeddings to 2D using PCA, then normalize to unit circle."""
    # Sample if too many
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]

    # PCA to 2D
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    # Normalize to unit circle
    emb_2d = emb_2d / (np.linalg.norm(emb_2d, axis=1, keepdims=True) + 1e-8)

    return emb_2d


def plot_single_model(name, distances, emb_2d, uniformity_loss, color, save_path):
    """Plot alignment and uniformity for a single model (1x2 layout: left=alignment, right=uniformity with 2D+angles)."""
    fig = plt.figure(figsize=(12, 6))

    # Left: Alignment (L2 distance histogram)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(distances, bins=50, alpha=0.7, color=color, edgecolor='white')
    ax1.axvline(distances.mean(), color='black', linestyle='--', linewidth=1.5, label='Mean')
    ax1.set_xlabel('$\ell_2$ Distances', fontsize=12)
    ax1.set_ylabel('Counts', fontsize=12)
    ax1.set_title('Alignment\nPositive Pair Feature Distances', fontsize=14, fontweight='bold')
    ax1.legend()

    # Right: Uniformity (2D scatter on top, angle histogram on bottom)
    # Create a gridspec for the right side
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=fig.add_subplot(1, 2, 2).get_subplotspec(),
                                        height_ratios=[2, 1], hspace=0.3)
    # Remove the placeholder subplot
    fig.axes[-1].remove()

    # Top right: 2D scatter with KDE contour
    ax2 = fig.add_subplot(gs_right[0])
    x, y = emb_2d[:, 0], emb_2d[:, 1]

    # KDE contour
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        xmin, xmax = -1.5, 1.5
        ymin, ymax = -1.5, 1.5
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = np.reshape(kde(positions).T, xx.shape)
        cmap = 'Blues' if color == 'blue' else 'Oranges'
        ax2.contourf(xx, yy, zz, levels=20, cmap=cmap, alpha=0.8)
    except:
        pass

    # Scatter points
    ax2.scatter(x, y, s=1, alpha=0.3, color=color)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.7, linewidth=1.5)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Features', fontsize=10)
    ax2.set_ylabel('', fontsize=10)
    ax2.set_title('Uniformity\nFeature Distribution', fontsize=14, fontweight='bold')

    # Bottom right: Angle histogram
    ax3 = fig.add_subplot(gs_right[1])
    angles = np.arctan2(y, x)  # [-pi, pi]
    ax3.hist(angles, bins=72, alpha=0.7, color=color, edgecolor='white')
    ax3.set_xlabel('Angles', fontsize=10)
    ax3.set_ylabel('Counts', fontsize=10)
    ax3.set_xlim(-np.pi, np.pi)

    plt.suptitle(f'{name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def compute_uniformity_loss(embeddings, t=2.0):
    """
    Compute uniformity loss as in the paper.
    L_uniform = log E[e^(-t||z_i - z_j||^2)]

    Lower is better (more uniform).
    """
    embs = normalize(embeddings)
    n = min(len(embs), 5000)  # Sample for efficiency
    if len(embs) > n:
        indices = np.random.choice(len(embs), n, replace=False)
        embs = embs[indices]

    # Compute pairwise distances
    embs_t = torch.from_numpy(embs).float()
    sq_dist = torch.cdist(embs_t, embs_t, p=2).pow(2)

    # Remove diagonal
    mask = ~torch.eye(len(embs_t), dtype=torch.bool)
    sq_dist = sq_dist[mask]

    # Uniformity loss
    uniformity = torch.log(torch.exp(-t * sq_dist).mean()).item()

    return uniformity


def main():
    # Paths
    PLAYLIST_CSV = "clip copy/playlists_filtered.csv"

    print("=== Loading Embeddings ===")
    track_simgcl, playlist_simgcl, track_idx_simgcl, playlist_idx_simgcl = load_embeddings_with_ids(SIMGCL_DIR)
    track_lightgcn, playlist_lightgcn, track_idx_lightgcn, playlist_idx_lightgcn = load_embeddings_with_ids(LIGHTGCN_DIR)

    print(f"SimGCL - Tracks: {track_simgcl.shape}, Playlists: {playlist_simgcl.shape}")
    print(f"LightGCN - Tracks: {track_lightgcn.shape}, Playlists: {playlist_lightgcn.shape}")

    # Load positive pairs from CSV
    print("\n=== Loading Positive Pairs from CSV ===")
    positive_pairs = load_positive_pairs_from_csv(PLAYLIST_CSV)
    print(f"Loaded {len(positive_pairs)} playlists with positive pairs")

    # === SimGCL ===
    print("\n=== Processing SimGCL ===")
    distances_simgcl = compute_alignment(
        track_simgcl, playlist_simgcl, track_idx_simgcl, playlist_idx_simgcl, positive_pairs
    )
    emb_2d_simgcl = project_to_2d(track_simgcl)
    uni_simgcl = compute_uniformity_loss(track_simgcl)

    print(f"SimGCL - Mean L2 distance: {distances_simgcl.mean():.4f}")
    print(f"SimGCL - Uniformity loss: {uni_simgcl:.4f}")

    plot_single_model("SimGCL", distances_simgcl, emb_2d_simgcl, uni_simgcl, 'blue', "simgcl_alignment_uniformity.png")

    # === LightGCN ===
    print("\n=== Processing LightGCN ===")
    distances_lightgcn = compute_alignment(
        track_lightgcn, playlist_lightgcn, track_idx_lightgcn, playlist_idx_lightgcn, positive_pairs
    )
    emb_2d_lightgcn = project_to_2d(track_lightgcn)
    uni_lightgcn = compute_uniformity_loss(track_lightgcn)

    print(f"LightGCN - Mean L2 distance: {distances_lightgcn.mean():.4f}")
    print(f"LightGCN - Uniformity loss: {uni_lightgcn:.4f}")

    plot_single_model("LightGCN", distances_lightgcn, emb_2d_lightgcn, uni_lightgcn, 'orange', "lightgcn_alignment_uniformity.png")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'SimGCL':<15} {'LightGCN':<15}")
    print("-" * 60)
    print(f"{'Alignment (mean L2 dist)':<30} {distances_simgcl.mean():<15.4f} {distances_lightgcn.mean():<15.4f}")
    print(f"{'Uniformity loss':<30} {uni_simgcl:<15.4f} {uni_lightgcn:<15.4f}")
    print("=" * 60)
    print("\nLower alignment distance = better positive pair alignment")
    print("Lower uniformity loss = more uniform distribution")


if __name__ == "__main__":
    main()
