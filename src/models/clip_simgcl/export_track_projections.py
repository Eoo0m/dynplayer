"""
Export projected track embeddings using trained CLIP model.

Saves:
- clip_simgcl/track_ids.npy
- clip_simgcl/track_projected.npy
"""

import numpy as np
import torch
from tqdm import tqdm

from model import CaptionPlaylistCLIP

# Config
SIMGCL_DIR = "simgcl_weighted/outputs/min5_win10"
MODEL_PATH = "clip_simgcl/clip_best.pt"
CAPTION_DIM = 1536
PLAYLIST_DIM = 64
OUTPUT_DIM = 256


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SimGCL track embeddings
    print("\n=== Loading SimGCL Track Embeddings ===")
    track_embs = np.load(f"{SIMGCL_DIR}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{SIMGCL_DIR}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = np.array([str(tid) for tid in track_ids])
    print(f"Tracks: {len(track_ids)}, dim: {track_embs.shape[1]}")

    # Load model
    print("\n=== Loading CLIP Model ===")
    model = CaptionPlaylistCLIP(CAPTION_DIM, PLAYLIST_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {MODEL_PATH}")

    # Project all tracks
    print("\n=== Projecting Track Embeddings ===")
    all_projected = []
    batch_size = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(track_ids), batch_size), desc="Projecting"):
            batch = torch.tensor(track_embs[i:i+batch_size], dtype=torch.float32).to(device)
            projected = model.playlist_proj(batch).cpu().numpy()
            all_projected.append(projected)

    all_projected = np.vstack(all_projected)
    print(f"Projected shape: {all_projected.shape}")

    # Save
    np.save("clip_simgcl/track_ids.npy", track_ids)
    np.save("clip_simgcl/track_projected.npy", all_projected)

    print(f"\nSaved:")
    print(f"  - clip_simgcl/track_ids.npy ({len(track_ids)} tracks)")
    print(f"  - clip_simgcl/track_projected.npy {all_projected.shape}")


if __name__ == "__main__":
    main()
