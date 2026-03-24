"""
Upload projected track embeddings to Supabase track_embeddings table.
Uses Supabase REST API. Supports resume from checkpoint with retry logic.
"""

import os
import time
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TRACK_IDS_PATH = "clip_simgcl/exp_individual_tags/track_ids.npy"
TRACK_PROJ_PATH = "clip_simgcl/exp_individual_tags/track_projected.npy"
CHECKPOINT_PATH = "upload_checkpoint.txt"

BATCH_SIZE = 50
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(idx):
    with open(CHECKPOINT_PATH, 'w') as f:
        f.write(str(idx))


def upload_with_retry(supabase, track_id, emb, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            supabase.table("track_embeddings").update({
                "projected_embedding": emb.tolist()
            }).eq("track_key", str(track_id)).execute()
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"\nFailed after {retries} retries - {track_id}: {e}")
                return False
    return False


def main():
    print("Loading projected embeddings...")
    track_ids = np.load(TRACK_IDS_PATH, allow_pickle=True)
    track_projected = np.load(TRACK_PROJ_PATH)

    print(f"Track IDs: {len(track_ids)}")
    print(f"Projected embeddings: {track_projected.shape}")

    start_idx = load_checkpoint()
    if start_idx > 0:
        print(f"\nResuming from index {start_idx} ({start_idx}/{len(track_ids)} done)")

    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_KEY')
    supabase = create_client(url, key)

    print(f"\nUploading to projected_embedding column (batch_size={BATCH_SIZE})...")

    success_count = 0
    error_count = 0

    pbar = tqdm(range(start_idx, len(track_ids), BATCH_SIZE),
                desc="Uploading",
                initial=start_idx // BATCH_SIZE,
                total=(len(track_ids) + BATCH_SIZE - 1) // BATCH_SIZE)

    for batch_start in pbar:
        batch_end = min(batch_start + BATCH_SIZE, len(track_ids))
        batch_ids = track_ids[batch_start:batch_end]
        batch_embs = track_projected[batch_start:batch_end]

        batch_success = 0
        for track_id, emb in zip(batch_ids, batch_embs):
            if upload_with_retry(supabase, track_id, emb):
                success_count += 1
                batch_success += 1
            else:
                error_count += 1

        # Save checkpoint after each batch
        save_checkpoint(batch_end)
        pbar.set_postfix(success=success_count, errors=error_count)

    print(f"\nDone!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")


if __name__ == "__main__":
    main()
