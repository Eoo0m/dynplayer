"""
Upload SimGCL weighted embeddings to Supabase track_embeddings table.
Format matches new_track_embeddings table structure.
"""

import os
import json
import numpy as np
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm
import csv

load_dotenv()

# Supabase client
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


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
    # Load SimGCL embeddings
    print("Loading SimGCL embeddings...")
    simgcl_dir = Path("simgcl_weighted/outputs/min5_win10")
    embeddings = np.load(simgcl_dir / "model_loo_track_embeddings.npy")
    track_ids = np.load(simgcl_dir / "model_loo_track_ids.npy", allow_pickle=True)

    print(f"Embeddings: {embeddings.shape}")
    print(f"Track IDs: {len(track_ids)}")

    # Load track metadata
    print("Loading track metadata...")
    metadata = load_track_metadata("data/csvs/track_playlist_counts_min5_win10.csv")
    print(f"Metadata loaded: {len(metadata)} tracks")

    # Prepare data for upload
    print("Preparing data for upload...")
    rows = []
    for i, track_id in enumerate(track_ids):
        track_id = str(track_id)
        meta = metadata.get(track_id, {})

        # Convert embedding to list (for vector type)
        embedding_list = embeddings[i].tolist()

        # Cover image URL format from Supabase storage
        cover_url = f"https://xopigdxsobjbagmazgba.supabase.co/storage/v1/object/public/new_cover_image/{track_id}.webp"

        row = {
            'track_key': track_id,
            'title': meta.get('title', ''),
            'artist': meta.get('artist', ''),
            'album': meta.get('album', ''),
            'playlist_count': meta.get('playlist_count', 0),
            'embedding': embedding_list,  # vector type - send as array
            'cover_image_url': cover_url,
        }
        rows.append(row)

    print(f"Total rows to upload: {len(rows)}")

    # Upload in batches
    batch_size = 500
    print(f"\nUploading to track_embeddings table (batch size: {batch_size})...")

    for i in tqdm(range(0, len(rows), batch_size), desc="Uploading"):
        batch = rows[i:i+batch_size]
        try:
            supabase.table('track_embeddings').upsert(batch, on_conflict='track_key').execute()
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            # Try inserting one by one to find problematic rows
            for row in batch:
                try:
                    supabase.table('track_embeddings').upsert([row], on_conflict='track_key').execute()
                except Exception as e2:
                    print(f"  Failed: {row['track_key']} - {e2}")

    print("\nDone! Uploaded to track_embeddings table.")

    # Verify
    result = supabase.table('track_embeddings').select('*').limit(3).execute()
    print(f"\nSample uploaded rows:")
    for row in result.data:
        emb_preview = row['embedding'][:50] + "..." if row['embedding'] else "None"
        print(f"  {row['track_key']}: {row['title']} - {row['artist']} | emb: {emb_preview}")


if __name__ == "__main__":
    main()
