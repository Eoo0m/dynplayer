"""
Visualize high vs low popularity playlists with track distributions.

Compares playlists based on average track popularity and shows
how their tracks distribute in the embedding space.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.patches as mpatches


def load_data(model_dir, model_name):
    """Load all required data."""
    print("=== Loading Data ===")

    # Load playlist-track mapping
    playlist_tracks_df = pd.read_csv("clip/filtered_playlist_tracks.csv")
    print(f"Playlist-track mappings: {len(playlist_tracks_df):,}")

    # Load track metadata with popularity (count)
    track_meta_df = pd.read_csv("data/csvs/track_playlist_counts_min5_win10.csv")
    track_meta_df["track_id"] = track_meta_df["track_id"].astype(str)
    print(f"Track metadata: {len(track_meta_df):,}")

    # Create track_id -> count mapping
    track_popularity = dict(zip(track_meta_df["track_id"], track_meta_df["count"]))
    print(f"Tracks with popularity: {len(track_popularity):,}")

    # Load SimGCL embeddings
    print(f"Loading embeddings from: {model_dir} ({model_name})")
    track_embs = np.load(f"{model_dir}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{model_dir}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = np.array([str(tid) for tid in track_ids])
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    print(f"Track embeddings: {track_embs.shape}")

    playlist_embs = np.load(f"{model_dir}/model_loo_playlist_embeddings.npy")
    playlist_ids = np.load(f"{model_dir}/model_loo_playlist_ids.npy", allow_pickle=True)
    playlist_ids = np.array([str(pid) for pid in playlist_ids])
    playlist_id_to_idx = {pid: i for i, pid in enumerate(playlist_ids)}
    print(f"Playlist embeddings: {playlist_embs.shape}")

    # Load playlist metadata
    playlist_meta_df = pd.read_csv("clip copy/playlists_with_captions.csv")
    playlist_meta_df["playlist_id"] = playlist_meta_df["playlist_id"].astype(str)
    print(f"Playlist metadata: {len(playlist_meta_df):,}")

    return (
        playlist_tracks_df, track_popularity, track_meta_df,
        track_embs, track_ids, track_id_to_idx,
        playlist_embs, playlist_ids, playlist_id_to_idx,
        playlist_meta_df
    )


def calculate_playlist_popularity(playlist_tracks_df, track_popularity, playlist_id_to_idx):
    """Calculate average track popularity for each playlist."""
    print("\n=== Calculating Playlist Popularity ===")

    playlist_pop = {}

    for _, row in tqdm(playlist_tracks_df.iterrows(), total=len(playlist_tracks_df)):
        pid = str(row["playlist_id"])
        if pid not in playlist_id_to_idx:
            continue

        track_ids_str = row["track_ids"]
        if pd.isna(track_ids_str):
            continue

        track_ids_list = track_ids_str.split("|")

        # Get popularity for each track
        popularities = []
        for tid in track_ids_list:
            if tid in track_popularity:
                popularities.append(track_popularity[tid])

        if len(popularities) >= 5:  # Need at least 5 tracks
            playlist_pop[pid] = {
                "avg_pop": np.mean(popularities),
                "max_pop": np.max(popularities),
                "min_pop": np.min(popularities),
                "track_ids": track_ids_list,
                "popularities": popularities
            }

    print(f"Playlists with popularity calculated: {len(playlist_pop):,}")
    return playlist_pop


def select_playlists(playlist_pop, n_each=20):
    """Select top and bottom playlists by average popularity."""
    print(f"\n=== Selecting Top/Bottom {n_each} Playlists ===")

    # Sort by average popularity
    sorted_playlists = sorted(playlist_pop.items(), key=lambda x: x[1]["avg_pop"], reverse=True)

    high_pop_playlists = sorted_playlists[:n_each]
    low_pop_playlists = sorted_playlists[-n_each:]

    print(f"High popularity range: {high_pop_playlists[-1][1]['avg_pop']:.1f} - {high_pop_playlists[0][1]['avg_pop']:.1f}")
    print(f"Low popularity range: {low_pop_playlists[-1][1]['avg_pop']:.1f} - {low_pop_playlists[0][1]['avg_pop']:.1f}")

    return high_pop_playlists, low_pop_playlists


def sample_tracks_from_playlists(playlists, track_id_to_idx, track_popularity, n_tracks_per_playlist=5):
    """Sample tracks from each playlist."""
    sampled = []

    for pid, info in playlists:
        track_ids = info["track_ids"]
        popularities = info["popularities"]

        # Filter tracks that have embeddings
        valid_tracks = [(tid, pop) for tid, pop in zip(track_ids, popularities) if tid in track_id_to_idx]

        if len(valid_tracks) < n_tracks_per_playlist:
            sample = valid_tracks
        else:
            # Sample evenly across popularity range
            sorted_tracks = sorted(valid_tracks, key=lambda x: x[1])
            indices = np.linspace(0, len(sorted_tracks) - 1, n_tracks_per_playlist, dtype=int)
            sample = [sorted_tracks[i] for i in indices]

        for tid, pop in sample:
            sampled.append({
                "playlist_id": pid,
                "track_id": tid,
                "popularity": pop,
                "playlist_avg_pop": info["avg_pop"]
            })

    return sampled


def popularity_to_color(popularity, min_pop, max_pop):
    """Convert popularity to color on blue-red gradient."""
    # Normalize to 0-1 (log scale for better distribution)
    log_pop = np.log1p(popularity)
    log_min = np.log1p(min_pop)
    log_max = np.log1p(max_pop)

    if log_max == log_min:
        norm = 0.5
    else:
        norm = (log_pop - log_min) / (log_max - log_min)

    # Blue (low) to Red (high)
    r = norm
    b = 1 - norm
    g = 0.2

    return (r, g, b)


def visualize_embeddings(
    high_pop_playlists, low_pop_playlists,
    high_pop_tracks, low_pop_tracks,
    playlist_embs, playlist_id_to_idx,
    track_embs, track_id_to_idx,
    playlist_meta_df,
    output_path="eval/popularity/popularity_visualization.png"
):
    """Create t-SNE visualization."""
    print("\n=== Creating Visualization ===")

    # Collect all embeddings
    embeddings = []
    labels = []  # 'high_playlist', 'low_playlist', 'high_track', 'low_track'
    metadata = []

    # Add high popularity playlists
    for pid, info in high_pop_playlists:
        if pid in playlist_id_to_idx:
            embeddings.append(playlist_embs[playlist_id_to_idx[pid]])
            labels.append("high_playlist")
            metadata.append({"id": pid, "avg_pop": info["avg_pop"], "type": "playlist"})

    # Add low popularity playlists
    for pid, info in low_pop_playlists:
        if pid in playlist_id_to_idx:
            embeddings.append(playlist_embs[playlist_id_to_idx[pid]])
            labels.append("low_playlist")
            metadata.append({"id": pid, "avg_pop": info["avg_pop"], "type": "playlist"})

    # Add high popularity tracks
    for track_info in high_pop_tracks:
        tid = track_info["track_id"]
        if tid in track_id_to_idx:
            embeddings.append(track_embs[track_id_to_idx[tid]])
            labels.append("high_track")
            metadata.append({
                "id": tid,
                "popularity": track_info["popularity"],
                "playlist_id": track_info["playlist_id"],
                "type": "track"
            })

    # Add low popularity tracks
    for track_info in low_pop_tracks:
        tid = track_info["track_id"]
        if tid in track_id_to_idx:
            embeddings.append(track_embs[track_id_to_idx[tid]])
            labels.append("low_track")
            metadata.append({
                "id": tid,
                "popularity": track_info["popularity"],
                "playlist_id": track_info["playlist_id"],
                "type": "track"
            })

    embeddings = np.array(embeddings)
    print(f"Total embeddings for t-SNE: {len(embeddings)}")

    # Get min/max popularity for color scaling
    all_track_pops = [m["popularity"] for m in metadata if m["type"] == "track"]
    min_pop = min(all_track_pops)
    max_pop = max(all_track_pops)
    print(f"Track popularity range: {min_pop} - {max_pop}")

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Playlist colors
    playlist_colors = {
        "high_playlist": "#FF4444",  # Red for high pop playlists
        "low_playlist": "#4444FF",   # Blue for low pop playlists
    }

    # Plot tracks first (smaller, in background) with gradient colors
    for i, (label, meta) in enumerate(zip(labels, metadata)):
        if meta["type"] == "track":
            color = popularity_to_color(meta["popularity"], min_pop, max_pop)
            ax.scatter(coords[i, 0], coords[i, 1], c=[color], s=60, alpha=0.7, edgecolors='white', linewidths=0.5)
            # Add popularity number
            ax.annotate(
                str(int(meta["popularity"])),
                (coords[i, 0], coords[i, 1]),
                fontsize=6, ha='center', va='center',
                color='black', alpha=0.9, fontweight='bold'
            )

    # Plot playlists (larger circles, in foreground)
    for i, (label, meta) in enumerate(zip(labels, metadata)):
        if meta["type"] == "playlist":
            color = playlist_colors[label]
            ax.scatter(
                coords[i, 0], coords[i, 1],
                c=color, s=400, alpha=0.9,
                edgecolors='black', linewidths=2,
                marker='o'  # Circle for playlists
            )
            # Add average popularity
            ax.annotate(
                f"{meta['avg_pop']:.0f}",
                (coords[i, 0], coords[i, 1]),
                fontsize=8, ha='center', va='center',
                color='white', fontweight='bold'
            )

    # Draw lines connecting playlists to their tracks
    playlist_to_coord = {}
    playlist_to_label = {}
    for i, (label, meta) in enumerate(zip(labels, metadata)):
        if meta["type"] == "playlist":
            playlist_to_coord[meta["id"]] = coords[i]
            playlist_to_label[meta["id"]] = label

    for i, (label, meta) in enumerate(zip(labels, metadata)):
        if meta["type"] == "track" and meta["playlist_id"] in playlist_to_coord:
            p_coord = playlist_to_coord[meta["playlist_id"]]
            p_label = playlist_to_label[meta["playlist_id"]]
            line_color = playlist_colors[p_label]
            ax.plot(
                [p_coord[0], coords[i, 0]],
                [p_coord[1], coords[i, 1]],
                color=line_color, alpha=0.15, linewidth=0.5
            )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=playlist_colors["high_playlist"], edgecolor='black', label='High Popularity Playlists'),
        mpatches.Patch(facecolor=playlist_colors["low_playlist"], edgecolor='black', label='Low Popularity Playlists'),
    ]

    # Add colorbar for track popularity
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('pop_cmap', [(0, 0.2, 1), (1, 0.2, 0)])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.log1p(min_pop), vmax=np.log1p(max_pop)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label('Track Popularity (log scale)', fontsize=10)
    # Set colorbar ticks to actual popularity values
    tick_values = [min_pop, 10, 50, 100, 500, max_pop]
    tick_values = [v for v in tick_values if min_pop <= v <= max_pop]
    cbar.set_ticks([np.log1p(v) for v in tick_values])
    cbar.set_ticklabels([str(int(v)) for v in tick_values])

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Calculate statistics
    high_pop_avg = np.mean([m["avg_pop"] for m in metadata if m["type"] == "playlist" and "avg_pop" in m and labels[metadata.index(m)] == "high_playlist"])
    low_pop_avg = np.mean([m["avg_pop"] for m in metadata if m["type"] == "playlist" and "avg_pop" in m and labels[metadata.index(m)] == "low_playlist"])

    ax.set_title(
        f"High vs Low Popularity Playlists\n"
        f"High Pop Avg: {high_pop_avg:.1f} | Low Pop Avg: {low_pop_avg:.1f}",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel("t-SNE dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE dimension 2", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")

    # Show some statistics
    print("\n=== Statistics ===")
    print(f"High popularity playlists: {len(high_pop_playlists)}")
    print(f"Low popularity playlists: {len(low_pop_playlists)}")
    print(f"High pop tracks sampled: {len(high_pop_tracks)}")
    print(f"Low pop tracks sampled: {len(low_pop_tracks)}")

    # Calculate average similarity between playlists and their tracks
    print("\n=== Playlist-Track Similarity ===")

    # High popularity: playlist-track similarity
    high_pop_sims = []
    for pid, info in high_pop_playlists:
        if pid not in playlist_id_to_idx:
            continue
        p_emb = playlist_embs[playlist_id_to_idx[pid]]
        p_emb_norm = p_emb / (np.linalg.norm(p_emb) + 1e-10)

        for tid in info["track_ids"]:
            if tid in track_id_to_idx:
                t_emb = track_embs[track_id_to_idx[tid]]
                t_emb_norm = t_emb / (np.linalg.norm(t_emb) + 1e-10)
                sim = np.dot(p_emb_norm, t_emb_norm)
                high_pop_sims.append(sim)

    # Low popularity: playlist-track similarity
    low_pop_sims = []
    for pid, info in low_pop_playlists:
        if pid not in playlist_id_to_idx:
            continue
        p_emb = playlist_embs[playlist_id_to_idx[pid]]
        p_emb_norm = p_emb / (np.linalg.norm(p_emb) + 1e-10)

        for tid in info["track_ids"]:
            if tid in track_id_to_idx:
                t_emb = track_embs[track_id_to_idx[tid]]
                t_emb_norm = t_emb / (np.linalg.norm(t_emb) + 1e-10)
                sim = np.dot(p_emb_norm, t_emb_norm)
                low_pop_sims.append(sim)

    print(f"High popularity playlist-track avg similarity: {np.mean(high_pop_sims):.4f} (std: {np.std(high_pop_sims):.4f})")
    print(f"Low popularity playlist-track avg similarity: {np.mean(low_pop_sims):.4f} (std: {np.std(low_pop_sims):.4f})")

    return fig


def run_analysis(model_dir, model_name, output_path):
    """Run analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load data
    (
        playlist_tracks_df, track_popularity, track_meta_df,
        track_embs, track_ids, track_id_to_idx,
        playlist_embs, playlist_ids, playlist_id_to_idx,
        playlist_meta_df
    ) = load_data(model_dir, model_name)

    # Calculate playlist popularity
    playlist_pop = calculate_playlist_popularity(
        playlist_tracks_df, track_popularity, playlist_id_to_idx
    )

    # Select top/bottom playlists
    high_pop_playlists, low_pop_playlists = select_playlists(playlist_pop, n_each=20)

    # Sample tracks from each playlist
    high_pop_tracks = sample_tracks_from_playlists(
        high_pop_playlists, track_id_to_idx, track_popularity, n_tracks_per_playlist=5
    )
    low_pop_tracks = sample_tracks_from_playlists(
        low_pop_playlists, track_id_to_idx, track_popularity, n_tracks_per_playlist=5
    )

    # Visualize
    visualize_embeddings(
        high_pop_playlists, low_pop_playlists,
        high_pop_tracks, low_pop_tracks,
        playlist_embs, playlist_id_to_idx,
        track_embs, track_id_to_idx,
        playlist_meta_df,
        output_path=output_path
    )

    # Print sample playlists
    print("\n=== Sample High Popularity Playlists ===")
    for pid, info in high_pop_playlists[:5]:
        meta = playlist_meta_df[playlist_meta_df["playlist_id"] == pid]
        title = meta.iloc[0]["playlist_title"] if len(meta) > 0 else "Unknown"
        print(f"  [{info['avg_pop']:.1f}] {title[:50]}")

    print("\n=== Sample Low Popularity Playlists ===")
    for pid, info in low_pop_playlists[:5]:
        meta = playlist_meta_df[playlist_meta_df["playlist_id"] == pid]
        title = meta.iloc[0]["playlist_title"] if len(meta) > 0 else "Unknown"
        print(f"  [{info['avg_pop']:.1f}] {title[:50]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["randneg", "weighted", "both"], default="both")
    args = parser.parse_args()

    models = {
        "randneg": "models/simgcl_randneg/outputs/min5_win10",
        "weighted": "models/simgcl_weighted/outputs/min5_win10",
    }

    if args.model == "both":
        for name, path in models.items():
            output_path = f"eval/popularity/popularity_visualization_{name}.png"
            run_analysis(path, name, output_path)
    else:
        output_path = f"eval/popularity/popularity_visualization_{args.model}.png"
        run_analysis(models[args.model], args.model, output_path)


if __name__ == "__main__":
    main()
