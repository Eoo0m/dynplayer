"""
Export track projections through CLIP model and upload to Supabase.

1. Load SimGCL track embeddings
2. Project through CLIP playlist_proj
3. Upload to track_embeddings.projected_embedding column
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_simgcl.model import CaptionPlaylistCLIP

load_dotenv()


def main():
    # Config
    MODEL_DIR = "clip_simgcl/exp_individual_tags"
    SIMGCL_DIR = "models/simgcl_randneg/outputs/min5_win10"
    USE_BEST = True

    # Dimensions
    CAPTION_DIM = 3072
    PLAYLIST_DIM = 64
    OUTPUT_DIM = 512

    # Base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SimGCL track embeddings
    print("\n=== Loading SimGCL Track Embeddings ===")
    simgcl_path = os.path.join(base_dir, SIMGCL_DIR)
    track_embs = np.load(f"{simgcl_path}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{simgcl_path}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = np.array([str(tid) for tid in track_ids])

    print(f"Loaded {len(track_ids):,} tracks")
    print(f"Track embeddings shape: {track_embs.shape}")

    # Load CLIP model
    print("\n=== Loading CLIP Model ===")
    model = CaptionPlaylistCLIP(
        caption_dim=CAPTION_DIM,
        playlist_dim=PLAYLIST_DIM,
        out_dim=OUTPUT_DIM,
    ).to(device)

    model_file = "clip_best.pt" if USE_BEST else "clip.pt"
    model_path = os.path.join(base_dir, MODEL_DIR, model_file)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Project tracks through CLIP
    print("\n=== Projecting Track Embeddings ===")
    all_projected = []
    batch_size = 512

    with torch.no_grad():
        for i in tqdm(range(0, len(track_ids), batch_size), desc="Projecting"):
            batch_embs = torch.tensor(
                track_embs[i:i+batch_size],
                dtype=torch.float32
            ).to(device)

            projected = model.playlist_proj(batch_embs)  # (batch, out_dim)
            all_projected.append(projected.cpu().numpy())

    track_projected = np.vstack(all_projected)
    print(f"Projected embeddings shape: {track_projected.shape}")

    # Save locally
    output_dir = os.path.join(base_dir, MODEL_DIR)
    np.save(f"{output_dir}/track_ids.npy", track_ids)
    np.save(f"{output_dir}/track_projected.npy", track_projected)
    print(f"\nSaved locally:")
    print(f"  - {output_dir}/track_ids.npy")
    print(f"  - {output_dir}/track_projected.npy")

    # Upload to Supabase
    print("\n=== Uploading to Supabase ===")
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL or SUPABASE_KEY not found in environment")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Prepare updates
    print("Preparing updates...")
    updates = []
    for i, track_id in enumerate(track_ids):
        proj_list = track_projected[i].tolist()
        updates.append({
            'track_key': str(track_id),
            'projected_embedding': proj_list,
        })

    print(f"Total updates: {len(updates)}")

    # Upload in batches
    batch_size = 500
    print(f"Uploading (batch size: {batch_size})...")

    success_count = 0
    error_count = 0

    for i in tqdm(range(0, len(updates), batch_size), desc="Uploading"):
        batch = updates[i:i+batch_size]
        try:
            supabase.table('track_embeddings').upsert(
                batch,
                on_conflict='track_key'
            ).execute()
            success_count += len(batch)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            # Try one by one
            for row in batch:
                try:
                    supabase.table('track_embeddings').upsert(
                        [row],
                        on_conflict='track_key'
                    ).execute()
                    success_count += 1
                except Exception as e2:
                    print(f"  Failed: {row['track_key']} - {e2}")
                    error_count += 1

    print(f"\nUpload complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")

    # Verify
    result = supabase.table('track_embeddings').select('track_key, projected_embedding').limit(3).execute()
    print(f"\nSample uploaded rows:")
    for row in result.data:
        proj = row.get('projected_embedding')
        if proj:
            if isinstance(proj, list):
                preview = f"[{proj[0]:.4f}, {proj[1]:.4f}, ...] (dim={len(proj)})"
            else:
                preview = str(proj)[:50] + "..."
        else:
            preview = "None"
        print(f"  {row['track_key']}: {preview}")


if __name__ == "__main__":
    main()
