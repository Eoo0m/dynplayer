"""
Upload both SimGCL embeddings and projected embeddings to Supabase.

Uploads to track_embeddings table:
- embedding: SimGCL 64-dim track embeddings
- projected_embedding: CLIP projected 512-dim embeddings (large model)

Prerequisites:
1. Run SQL to set up columns:
   ALTER TABLE track_embeddings DROP COLUMN IF EXISTS embedding;
   ALTER TABLE track_embeddings ADD COLUMN embedding vector(64);
   ALTER TABLE track_embeddings ADD COLUMN projected_embedding vector(512);

2. Generate projected embeddings:
   python "clip copy/export_track_projections.py"
"""

import os
import csv
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Supabase client
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Paths
SIMGCL_DIR = "simgcl_weighted/outputs/min5_win10"
SIMGCL_TRACK_EMBS = f"{SIMGCL_DIR}/model_loo_track_embeddings.npy"
SIMGCL_TRACK_IDS = f"{SIMGCL_DIR}/model_loo_track_ids.npy"

# Projected embeddings (large model, 512-dim)
PROJECTED_IDS = "clip copy/clip_simgcl_track_ids.npy"
PROJECTED_EMBS = "clip copy/clip_simgcl_track_projected.npy"

# Metadata
METADATA_CSV = "data/csvs/track_playlist_counts_min5_win10.csv"


def load_track_metadata(csv_path):
    """Load track metadata from CSV."""
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['track_id']] = {
                'title': row['track'],
                'artist': row['artist'],
                'album': row['album'],
                'playlist_count': int(row.get('count', 0) or 0),
            }
    return metadata


def main():
    # Load SimGCL embeddings (64-dim)
    print("=== Loading SimGCL Track Embeddings ===")
    simgcl_embs = np.load(SIMGCL_TRACK_EMBS)
    simgcl_ids = np.load(SIMGCL_TRACK_IDS, allow_pickle=True)
    simgcl_ids = np.array([str(tid) for tid in simgcl_ids])

    simgcl_index = {tid: i for i, tid in enumerate(simgcl_ids)}
    print(f"SimGCL embeddings: {simgcl_embs.shape}")

    # Load projected embeddings (512-dim)
    print("\n=== Loading Projected Track Embeddings ===")
    proj_embs = np.load(PROJECTED_EMBS)
    proj_ids = np.load(PROJECTED_IDS, allow_pickle=True)
    proj_ids = np.array([str(tid) for tid in proj_ids])

    proj_index = {tid: i for i, tid in enumerate(proj_ids)}
    print(f"Projected embeddings: {proj_embs.shape}")

    # Load metadata
    print("\n=== Loading Metadata ===")
    metadata = load_track_metadata(METADATA_CSV)
    print(f"Metadata: {len(metadata)} tracks")

    # Find common track IDs
    common_ids = set(simgcl_ids) & set(proj_ids)
    print(f"\nCommon tracks: {len(common_ids)}")

    # Prepare rows
    print("\n=== Preparing Upload Data ===")
    rows = []

    for track_id in tqdm(common_ids, desc="Preparing"):
        meta = metadata.get(track_id, {})

        # Get embeddings
        simgcl_emb = simgcl_embs[simgcl_index[track_id]].tolist()
        proj_emb = proj_embs[proj_index[track_id]].tolist()

        # Cover image URL
        cover_url = f"https://xopigdxsobjbagmazgba.supabase.co/storage/v1/object/public/new_cover_image/{track_id}.webp"

        row = {
            'track_key': track_id,
            'title': meta.get('title', ''),
            'artist': meta.get('artist', ''),
            'album': meta.get('album', ''),
            'playlist_count': meta.get('playlist_count', 0),
            'embedding': simgcl_emb,  # vector(64)
            'projected_embedding': proj_emb,  # vector(512)
            'cover_image_url': cover_url,
        }
        rows.append(row)

    print(f"Total rows to upload: {len(rows)}")

    # Upload in batches (smaller batch to avoid timeout)
    batch_size = 100
    print(f"\n=== Uploading to track_embeddings (batch size: {batch_size}) ===")

    failed = []
    for i in tqdm(range(0, len(rows), batch_size), desc="Uploading"):
        batch = rows[i:i+batch_size]
        try:
            supabase.table('track_embeddings').upsert(
                batch,
                on_conflict='track_key'
            ).execute()
        except Exception as e:
            print(f"\nError at batch {i}: {e}")
            # Try one by one
            for row in batch:
                try:
                    supabase.table('track_embeddings').upsert(
                        [row],
                        on_conflict='track_key'
                    ).execute()
                except Exception as e2:
                    print(f"  Failed: {row['track_key']} - {e2}")
                    failed.append(row['track_key'])

    print(f"\n=== Upload Complete ===")
    print(f"Uploaded: {len(rows) - len(failed)} tracks")
    if failed:
        print(f"Failed: {len(failed)} tracks")

    # Verify
    print("\n=== Verification ===")
    result = supabase.table('track_embeddings').select('track_key, title, artist, embedding, projected_embedding').limit(3).execute()

    for row in result.data:
        emb = row.get('embedding')
        proj = row.get('projected_embedding')

        # pgvector returns as string "[0.1,0.2,...]", parse to get actual dim
        if emb:
            emb_dim = emb.count(',') + 1 if isinstance(emb, str) else len(emb)
        else:
            emb_dim = 0

        if proj:
            proj_dim = proj.count(',') + 1 if isinstance(proj, str) else len(proj)
        else:
            proj_dim = 0

        print(f"  {row['track_key']}: {row['title']} - {row['artist']}")
        print(f"    embedding: dim={emb_dim}, projected_embedding: dim={proj_dim}")


if __name__ == "__main__":
    main()
