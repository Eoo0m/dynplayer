"""
Compare search result diversity between two CLIP models.

For each keyword, search top 10 tracks and compute average pairwise similarity.
Lower similarity = more diverse results.
"""

import os
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clip_simgcl.model import CaptionPlaylistCLIP

load_dotenv()

# Keywords to test
KEYWORDS = [
    # Moods
    "happy", "sad", "energetic", "calm", "romantic", "angry", "melancholic", "uplifting",
    "dark", "bright", "dreamy", "intense", "peaceful", "aggressive", "nostalgic",
    # Genres
    "pop", "rock", "hip hop", "jazz", "classical", "electronic", "r&b", "country",
    "indie", "metal", "folk", "blues", "reggae", "punk", "soul",
    # Activities
    "workout", "party", "study", "sleep", "driving", "cooking", "running", "meditation",
    "road trip", "beach", "morning", "night", "coffee shop", "gym",
    # Descriptive
    "chill vibes", "summer hits", "90s throwback", "acoustic", "piano", "guitar",
    "female vocals", "male vocals", "instrumental", "lo-fi", "bass heavy",
]

TOP_K = 10


def embed_text(client, text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return np.array(response.data[0].embedding)


def load_model_and_data(model_dir, device):
    """Load CLIP model and compute projected track embeddings."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    simgcl_dir = "models/simgcl_randneg/outputs/min5_win10"

    # Load SimGCL track embeddings
    track_embs = np.load(f"{base_dir}/{simgcl_dir}/model_loo_track_embeddings.npy")
    track_ids = np.load(f"{base_dir}/{simgcl_dir}/model_loo_track_ids.npy", allow_pickle=True)

    # Load CLIP model
    model = CaptionPlaylistCLIP(
        caption_dim=3072,
        playlist_dim=64,
        out_dim=512,
    ).to(device)

    model_path = os.path.join(base_dir, model_dir, "clip_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Project all track embeddings
    all_projected = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(track_ids), batch_size):
            batch_embs = torch.tensor(track_embs[i:i+batch_size], dtype=torch.float32).to(device)
            projected = model.playlist_proj(batch_embs)
            all_projected.append(projected.cpu().numpy())

    track_projected = np.concatenate(all_projected, axis=0)

    return model, track_ids, track_projected


def search_and_compute_similarity(query_emb, model, track_projected, device, top_k=10):
    """Search for top_k tracks and compute average pairwise similarity."""
    # Project query
    query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        projected_query = model.caption_proj(query_tensor).cpu().numpy().squeeze()

    # Normalize
    projected_query = projected_query / (np.linalg.norm(projected_query) + 1e-10)
    track_norms = np.linalg.norm(track_projected, axis=1, keepdims=True) + 1e-10
    track_projected_norm = track_projected / track_norms

    # Find top-k
    similarities = track_projected_norm @ projected_query
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    # Get top-k embeddings
    top_embs = track_projected_norm[top_indices]

    # Compute pairwise similarity matrix
    pairwise_sim = top_embs @ top_embs.T

    # Average pairwise similarity (excluding diagonal)
    n = len(top_indices)
    mask = ~np.eye(n, dtype=bool)
    avg_pairwise_sim = pairwise_sim[mask].mean()

    return {
        "avg_relevance": float(top_scores.mean()),
        "avg_pairwise_sim": float(avg_pairwise_sim),
        "top_scores": top_scores.tolist(),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load both models
    print("\n=== Loading exp_text_fast model ===")
    model_fast, track_ids, track_proj_fast = load_model_and_data("clip_simgcl/exp_text_fast", device)

    print("\n=== Loading exp_baseline model ===")
    model_baseline, _, track_proj_baseline = load_model_and_data("clip_simgcl/exp_baseline", device)

    print(f"\nTracks: {len(track_ids):,}")
    print(f"Keywords to test: {len(KEYWORDS)}")

    # Embed all keywords first
    print("\nEmbedding keywords...")
    keyword_embs = {}
    for kw in tqdm(KEYWORDS):
        keyword_embs[kw] = embed_text(client, kw)

    # Compare models
    print("\nComparing models...")
    results = []

    for kw in tqdm(KEYWORDS):
        query_emb = keyword_embs[kw]

        # Fast model
        res_fast = search_and_compute_similarity(query_emb, model_fast, track_proj_fast, device, TOP_K)

        # Baseline model
        res_baseline = search_and_compute_similarity(query_emb, model_baseline, track_proj_baseline, device, TOP_K)

        results.append({
            "keyword": kw,
            "fast_relevance": res_fast["avg_relevance"],
            "fast_pairwise_sim": res_fast["avg_pairwise_sim"],
            "baseline_relevance": res_baseline["avg_relevance"],
            "baseline_pairwise_sim": res_baseline["avg_pairwise_sim"],
        })

    # Create summary
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print("\n[Average across all keywords]")
    print(f"{'Model':<20} {'Relevance':>12} {'Pairwise Sim':>15} {'Diversity':>12}")
    print("-" * 60)

    fast_rel = df["fast_relevance"].mean()
    fast_sim = df["fast_pairwise_sim"].mean()
    baseline_rel = df["baseline_relevance"].mean()
    baseline_sim = df["baseline_pairwise_sim"].mean()

    print(f"{'exp_text_fast':<20} {fast_rel:>12.4f} {fast_sim:>15.4f} {1-fast_sim:>12.4f}")
    print(f"{'exp_baseline':<20} {baseline_rel:>12.4f} {baseline_sim:>15.4f} {1-baseline_sim:>12.4f}")

    print("\n[Interpretation]")
    print("- Relevance: Higher = search results match query better")
    print("- Pairwise Sim: Lower = more diverse results (tracks are different from each other)")
    print("- Diversity = 1 - Pairwise Sim")

    if fast_sim < baseline_sim:
        diff = (baseline_sim - fast_sim) / baseline_sim * 100
        print(f"\n=> exp_text_fast produces {diff:.1f}% more diverse results")
    else:
        diff = (fast_sim - baseline_sim) / fast_sim * 100
        print(f"\n=> exp_baseline produces {diff:.1f}% more diverse results")

    # Detailed per-keyword results
    print("\n" + "=" * 80)
    print("PER-KEYWORD RESULTS (sorted by diversity difference)")
    print("=" * 80)

    df["diversity_diff"] = df["baseline_pairwise_sim"] - df["fast_pairwise_sim"]
    df_sorted = df.sort_values("diversity_diff", ascending=False)

    print(f"\n{'Keyword':<20} {'Fast Sim':>10} {'Base Sim':>10} {'Diff':>10}")
    print("-" * 55)
    for _, row in df_sorted.iterrows():
        diff = row["baseline_pairwise_sim"] - row["fast_pairwise_sim"]
        sign = "+" if diff > 0 else ""
        print(f"{row['keyword']:<20} {row['fast_pairwise_sim']:>10.4f} {row['baseline_pairwise_sim']:>10.4f} {sign}{diff:>9.4f}")

    # Save results
    df.to_csv("model_diversity_comparison.csv", index=False)
    print("\nResults saved to model_diversity_comparison.csv")


if __name__ == "__main__":
    main()
