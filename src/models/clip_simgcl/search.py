"""
Search tracks using text queries via CLIP model.
"""

import os
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from model import CaptionPlaylistCLIP

# Config
SIMGCL_DIR = "simgcl_weighted/outputs/min5_win10"
TRACK_CSV = "data/csvs/track_playlist_counts_min5_win10.csv"
MODEL_PATH = "clip_simgcl/clip_best.pt"
EMBEDDING_MODEL = "text-embedding-3-small"
CAPTION_DIM = 1536
PLAYLIST_DIM = 64
OUTPUT_DIM = 256


def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SimGCL track embeddings
    print("\n=== Loading Track Embeddings ===")
    track_embs = np.load(f"{SIMGCL_DIR}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{SIMGCL_DIR}/model_loo_track_ids.npy", allow_pickle=True)
    track_ids = [str(tid) for tid in track_ids]
    print(f"Tracks: {len(track_ids)}, dim: {track_embs.shape[1]}")

    # Load metadata
    track_df = pd.read_csv(TRACK_CSV)
    track_meta = {
        row["track_id"]: (row["track"], row["artist"])
        for _, row in track_df.iterrows()
    }

    # Load model
    print("\n=== Loading Model ===")
    model = CaptionPlaylistCLIP(CAPTION_DIM, PLAYLIST_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {MODEL_PATH}")

    # Pre-project all tracks
    print("\n=== Projecting Tracks ===")
    all_proj = []
    with torch.no_grad():
        for i in tqdm(range(0, len(track_ids), 512)):
            batch = torch.tensor(track_embs[i:i+512], dtype=torch.float32).to(device)
            all_proj.append(model.playlist_proj(batch))
    all_proj = torch.cat(all_proj, dim=0)
    print(f"Projected: {all_proj.shape}")

    # Interactive search
    print("\n=== Search (type 'q' to quit) ===")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            break
        if not query:
            continue

        # Embed query
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        q_emb = torch.tensor(res.data[0].embedding, dtype=torch.float32).unsqueeze(0).to(device)

        # Search
        with torch.no_grad():
            q_proj = model.caption_proj(q_emb)
            sims = torch.matmul(q_proj, all_proj.T).squeeze(0)
            scores, indices = sims.topk(20)

        # Print results
        print(f"\n{'='*60}")
        print(f"Results for: '{query}'")
        print('='*60)
        for i, (score, idx) in enumerate(zip(scores.cpu().numpy(), indices.cpu().numpy()), 1):
            tid = track_ids[idx]
            title, artist = track_meta.get(tid, ("Unknown", "Unknown"))
            print(f"{i:2d}. [{score:.3f}] {title} - {artist}")
        print('='*60)


if __name__ == "__main__":
    main()
