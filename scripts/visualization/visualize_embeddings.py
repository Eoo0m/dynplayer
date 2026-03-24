"""
Visualize and compare embeddings from two models

Usage:
    python visualize_embeddings.py \
        --model1-path contrastive_learning_loo/outputs/min5_win10/model_loo_track_embeddings.npy \
        --model1-name "Contrastive Learning" \
        --model2-path simgcl_loo/outputs/min5_win10/model_loo_track_embeddings.npy \
        --model2-name "SimGCL" \
        --method pca \
        --n-samples 5000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde


def load_embeddings(model_path):
    """Load track embeddings and track IDs"""
    embeddings = np.load(model_path)

    # Try to load corresponding track_ids file (multiple naming conventions)
    track_ids_path = model_path.replace('_track_embeddings.npy', '_track_ids.npy')
    track_ids_path_alt = model_path.replace('_track_embeddings.npy', '_track_keys.npy')
    track_ids_path_alt2 = model_path.replace('_embeddings.npy', '_track_ids.npy')

    if Path(track_ids_path).exists():
        track_ids = np.load(track_ids_path, allow_pickle=True)
        return embeddings, track_ids
    elif Path(track_ids_path_alt).exists():
        track_ids = np.load(track_ids_path_alt, allow_pickle=True)
        return embeddings, track_ids
    elif Path(track_ids_path_alt2).exists():
        track_ids = np.load(track_ids_path_alt2, allow_pickle=True)
        return embeddings, track_ids
    else:
        return embeddings, None


def reduce_dimensions(embeddings, method='pca', n_components=2, random_state=42):
    """
    Reduce embeddings to 2D

    Args:
        embeddings: (N, D) high-dimensional embeddings
        method: 'pca' or 'tsne'
        n_components: number of dimensions (2 for visualization)
        random_state: random seed

    Returns:
        reduced: (N, 2) 2D coordinates
    """
    print(f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using {method.upper()}...")

    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f} (total: {sum(explained_var):.3f})")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30, n_iter=1000)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    return reduced


def compute_angles(coords_2d):
    """
    Compute angles from 2D coordinates

    Args:
        coords_2d: (N, 2) array of (x, y) coordinates

    Returns:
        angles: (N,) array of angles in radians [-pi, pi]
    """
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]
    angles = np.arctan2(y, x)
    return angles


def plot_scatter(coords_2d, title, ax):
    """Plot 2D scatter plot with density coloring"""
    x = coords_2d[:, 0]
    y = coords_2d[:, 1]

    # Normalize to unit circle (like in the paper)
    norms = np.sqrt(x**2 + y**2)
    x_norm = x / (norms + 1e-10)
    y_norm = y / (norms + 1e-10)

    # Compute point density for coloring
    try:
        xy = np.vstack([x_norm, y_norm])
        density = gaussian_kde(xy)(xy)

        # Sort points by density so dense points are plotted on top
        idx = density.argsort()
        x_norm, y_norm, density = x_norm[idx], y_norm[idx], density[idx]

        scatter = ax.scatter(x_norm, y_norm, c=density, s=8, alpha=0.6,
                           cmap='viridis', edgecolors='none', rasterized=True)
    except:
        # Fallback if KDE fails
        scatter = ax.scatter(x_norm, y_norm, s=8, alpha=0.4,
                           c='#2ecc71', edgecolors='none', rasterized=True)

    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('Features', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add circle outline for reference
    circle = plt.Circle((0, 0), 1.0, fill=False, color='gray',
                       linewidth=0.5, linestyle='--', alpha=0.3)
    ax.add_patch(circle)


def plot_angle_distribution(angles, title, ax):
    """Plot angle distribution as smooth density curve"""
    # Keep in radians for better visualization
    n_bins = 50
    density, bins = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi), density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot as filled area
    ax.fill_between(bin_centers, density, alpha=0.7, color='#2ecc71', edgecolor='#27ae60', linewidth=1.5)
    ax.plot(bin_centers, density, color='#27ae60', linewidth=2)

    ax.set_xlabel('Angles', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, None)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def plot_angle_polar(angles, title, ax):
    """Plot angle distribution as polar histogram (removed - not in paper style)"""
    # This function is no longer used in the main visualization
    pass


def main():
    parser = argparse.ArgumentParser(description="Visualize and compare embeddings")
    parser.add_argument("--model1-path", required=True, help="Path to model 1 embeddings")
    parser.add_argument("--model1-name", default="Model 1", help="Name of model 1")
    parser.add_argument("--model2-path", required=True, help="Path to model 2 embeddings")
    parser.add_argument("--model2-name", default="Model 2", help="Name of model 2")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples to visualize (0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output file path (default: show plot)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"\n=== Embedding Visualization ===")
    print(f"Model 1: {args.model1_name}")
    print(f"Model 2: {args.model2_name}")
    print(f"Method: {args.method.upper()}")

    # Load embeddings
    print(f"\n=== Loading Embeddings ===")
    emb1, track_ids1 = load_embeddings(args.model1_path)
    emb2, track_ids2 = load_embeddings(args.model2_path)

    print(f"{args.model1_name} shape: {emb1.shape}")
    print(f"{args.model2_name} shape: {emb2.shape}")

    # Sample if requested
    if args.n_samples > 0 and args.n_samples < len(emb1):
        print(f"\nSampling {args.n_samples} tracks for visualization...")
        indices = np.random.choice(len(emb1), args.n_samples, replace=False)
        emb1_vis = emb1[indices]
        emb2_vis = emb2[indices]
    else:
        print(f"\nVisualizing all {len(emb1)} tracks...")
        emb1_vis = emb1
        emb2_vis = emb2

    # Reduce dimensions
    print(f"\n=== Dimensionality Reduction ===")
    coords1_2d = reduce_dimensions(emb1_vis, method=args.method, random_state=args.seed)
    coords2_2d = reduce_dimensions(emb2_vis, method=args.method, random_state=args.seed)

    # Compute angles
    angles1 = compute_angles(coords1_2d)
    angles2 = compute_angles(coords2_2d)

    print(f"\n{args.model1_name} angles: mean={np.degrees(np.mean(angles1)):.1f}°, std={np.degrees(np.std(angles1)):.1f}°")
    print(f"{args.model2_name} angles: mean={np.degrees(np.mean(angles2)):.1f}°, std={np.degrees(np.std(angles2)):.1f}°")

    # Create visualization (paper style: 2 rows x 2 columns)
    print(f"\n=== Creating Visualization ===")
    fig = plt.figure(figsize=(10, 8))

    # Model 1 visualizations
    ax1 = plt.subplot(2, 2, 1)
    plot_scatter(coords1_2d, args.model1_name, ax1)

    ax2 = plt.subplot(2, 2, 2)
    plot_angle_distribution(angles1, args.model1_name, ax2)

    # Model 2 visualizations
    ax3 = plt.subplot(2, 2, 3)
    plot_scatter(coords2_2d, args.model2_name, ax3)

    ax4 = plt.subplot(2, 2, 4)
    plot_angle_distribution(angles2, args.model2_name, ax4)

    plt.tight_layout(pad=2.0)

    # Save or show
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nSaved visualization to {output_path}")
    else:
        print(f"\nDisplaying visualization...")
        plt.show()

    plt.close()

    # Additional statistics
    print(f"\n=== Statistics ===")
    print(f"\n{args.model1_name}:")
    print(f"  Angle range: [{np.degrees(angles1.min()):.1f}°, {np.degrees(angles1.max()):.1f}°]")
    print(f"  Median angle: {np.degrees(np.median(angles1)):.1f}°")

    print(f"\n{args.model2_name}:")
    print(f"  Angle range: [{np.degrees(angles2.min()):.1f}°, {np.degrees(angles2.max()):.1f}°]")
    print(f"  Median angle: {np.degrees(np.median(angles2)):.1f}°")

    # Compute angle difference
    angle_diff = np.abs(angles1 - angles2)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # Wrap around
    print(f"\nAngle difference between models:")
    print(f"  Mean: {np.degrees(np.mean(angle_diff)):.1f}°")
    print(f"  Median: {np.degrees(np.median(angle_diff)):.1f}°")
    print(f"  Std: {np.degrees(np.std(angle_diff)):.1f}°")


if __name__ == "__main__":
    main()
