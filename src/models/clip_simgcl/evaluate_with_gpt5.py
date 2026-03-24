"""
Evaluate CLIP models using keyword search and GPT-5 API verification.

Compares exp_text_fast vs exp_baseline models:
1. Search top 10 tracks per keyword (direct track search via CLIP projection)
2. Verify relevance using GPT-5 API
3. Score: 0 (not relevant), 0.5 (partially relevant), 1 (relevant), - (unknown)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Track playlist counts CSV
TRACK_CSV = "data/csvs/track_playlist_counts_min5_win10.csv"


KEYWORDS = [
    # Moods
    "chill", "relaxing", "happy", "sad", "melancholy", "energetic", "calm", "peaceful",
    "romantic", "nostalgic", "dreamy", "uplifting", "dark", "intense", "mellow",
    # Activities
    "workout", "study", "sleep", "driving", "party", "cooking", "meditation", "yoga",
    "running", "reading", "focus", "commute", "cleaning", "gaming",
    # Genres
    "jazz", "classical", "hip hop", "rock", "indie", "electronic", "acoustic",
    "r&b", "soul", "funk", "blues", "country", "folk", "pop", "metal",
    # Korean - Moods
    "감성", "새벽", "힐링", "드라이브", "카페", "비오는날", "봄", "여름", "가을", "겨울",
    "쓸쓸한", "설레는", "잔잔한", "신나는", "우울할때", "기분전환",
    "따뜻한", "시원한", "청량한", "몽환적인", "아련한", "달달한", "씁쓸한",
    "편안한", "활기찬", "고요한", "포근한", "상쾌한", "나른한", "애절한",
    # Korean - Activities
    "공부할때", "운동할때", "출퇴근", "자기전", "아침", "저녁", "새벽감성",
    "혼술", "혼밥", "산책", "독서", "집중", "휴식", "명상",
    # Korean - Genres/Styles
    "발라드", "힙합", "알앤비", "재즈", "클래식", "어쿠스틱", "인디",
    "트로트", "OST", "CCM", "EDM", "케이팝",
    # Korean - Situations
    "이별", "사랑", "짝사랑", "고백", "추억", "여행", "퇴근길",
    "불금", "주말", "월요병", "야근", "밤샘", "소풍", "데이트",
    # Vibes
    "lo-fi", "ambient", "synthwave", "chillhop", "bossa nova", "tropical",
    "cinematic", "epic", "minimalist", "groovy", "funky", "soulful",
]


def load_simgcl_track_data():
    """Load SimGCL track embeddings."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    simgcl_dir = os.path.join(base_dir, "models/simgcl_randneg/outputs/min5_win10")

    track_ids = np.load(f"{simgcl_dir}/model_loo_track_ids.npy", allow_pickle=True)
    track_embs = np.load(f"{simgcl_dir}/model_loo_track_embeddings.npy")

    return track_ids, track_embs


def project_tracks_through_clip(model, track_embs, device, batch_size=512):
    """Project track embeddings through CLIP playlist projection."""
    import torch

    all_projected = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(track_embs), batch_size), desc="Projecting tracks"):
            batch = torch.tensor(track_embs[i:i+batch_size], dtype=torch.float32).to(device)
            projected = model.playlist_proj(batch)
            all_projected.append(projected.cpu().numpy())

    projected = np.vstack(all_projected)

    # Normalize
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / (norms + 1e-10)

    return projected


def get_track_metadata_from_supabase(track_ids, supabase):
    """Fetch track metadata from Supabase."""
    print(f"Fetching metadata for {len(track_ids)} tracks...")

    metadata = {}
    batch_size = 100

    for i in tqdm(range(0, len(track_ids), batch_size), desc="Fetching tracks"):
        batch = track_ids[i:i+batch_size]

        try:
            result = supabase.table("track_embeddings").select("track_key, title, artist").in_("track_key", list(batch)).execute()

            for row in result.data:
                track_id = row["track_key"]
                title = row.get("title", "Unknown")
                artist = row.get("artist", "Unknown")

                metadata[track_id] = {
                    "name": title,
                    "artists": artist,
                    "display": f"{title} - {artist}"
                }
        except Exception as e:
            print(f"Error fetching batch: {e}")
            continue

    return metadata


def embed_keywords(client, keywords):
    """Embed keywords using OpenAI."""
    print(f"Embedding {len(keywords)} keywords...")

    embeddings = {}
    batch_size = 50

    for i in tqdm(range(0, len(keywords), batch_size), desc="Embedding"):
        batch = keywords[i:i+batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch
        )

        for j, kw in enumerate(batch):
            embeddings[kw] = np.array(response.data[j].embedding)

    return embeddings


def search_tracks_direct(keyword_emb, model, track_embs_projected, track_ids, device,
                         track_counts, top_k_search=100, top_k_popular=10):
    """
    Search tracks directly by projecting keyword through caption encoder
    and comparing with projected track embeddings.

    1. Get top-100 by similarity
    2. Re-rank by playlist count (popularity)
    3. Return top-10 most popular among those
    """
    import torch

    model.eval()

    with torch.no_grad():
        # Project keyword through caption encoder
        kw_tensor = torch.tensor(keyword_emb, dtype=torch.float32).unsqueeze(0).to(device)
        projected_kw = model.caption_proj(kw_tensor).cpu().numpy()[0]

    # Normalize
    projected_kw = projected_kw / (np.linalg.norm(projected_kw) + 1e-10)

    # Compute similarities with all tracks
    similarities = track_embs_projected @ projected_kw

    # Get top-100 by similarity
    top_100_indices = np.argsort(similarities)[::-1][:top_k_search]

    # Re-rank by playlist count (popularity)
    top_100_tracks = []
    for idx in top_100_indices:
        tid = str(track_ids[idx])
        sim = float(similarities[idx])
        count = track_counts.get(tid, 0)
        top_100_tracks.append((tid, sim, count))

    # Sort by count (descending), take top-10
    top_100_tracks.sort(key=lambda x: x[2], reverse=True)
    results = [(tid, sim) for tid, sim, count in top_100_tracks[:top_k_popular]]

    return results


_call_count = 0

def verify_with_gpt5(client, keyword, tracks_with_meta, max_tracks=10, debug_interval=10):
    """
    Use GPT-5 API to verify if tracks are relevant to keyword.

    Returns: dict with relevance scores (0, 0.5, 1) or "-" if unknown
    """
    global _call_count
    _call_count += 1
    show_debug = (_call_count % debug_interval == 1)  # Show every N calls

    track_list = "\n".join([
        f"{i+1}. {meta['display']}"
        for i, (tid, meta) in enumerate(tracks_with_meta[:max_tracks])
    ])

    prompt = f"""Given the keyword "{keyword}", score each track's relevance.

Tracks:
{track_list}

Scoring:
- 1: Good match
- 0.5: Partial match
- 0: Not a match
- "-": You have absolutely no idea what this song sounds like

IMPORTANT: If you know the artist or have any sense of the song's mood/genre, you MUST give a numeric score (0, 0.5, or 1). Only use "-" for completely unknown songs.

Output JSON only: {{"1": score, "2": score, ...}}"""

    if show_debug:
        print(f"\n[DEBUG #{_call_count}] Keyword: {keyword}")
        print(f"[DEBUG] Track list:\n{track_list}")

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a music expert. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=8000
        )

        if show_debug:
            print(f"[DEBUG] Response: {repr(response.choices[0].message.content)}")
            print(f"[DEBUG] Usage: {response.usage}")

        if not response.choices:
            print(f"GPT-5 returned no choices: {response}")
            return {tid: "-" for tid, _ in tracks_with_meta[:max_tracks]}

        message = response.choices[0].message
        content = message.content

        if hasattr(message, 'refusal') and message.refusal:
            print(f"GPT-5 refused: {message.refusal}")
            return {tid: "-" for tid, _ in tracks_with_meta[:max_tracks]}

        if content is None:
            print(f"GPT-5 returned None content, finish_reason: {response.choices[0].finish_reason}")
            return {tid: "-" for tid, _ in tracks_with_meta[:max_tracks]}

        content = content.strip()
        original_content = content

        # Clean up markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{"):
                    content = cleaned
                    break

        # Find JSON object in content
        if "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end]

        if not content or content == "":
            print(f"Empty content after parsing. Original: {original_content[:300]}")
            return {tid: "-" for tid, _ in tracks_with_meta[:max_tracks]}

        scores = json.loads(content)

        relevance = {}
        for i, (tid, _) in enumerate(tracks_with_meta[:max_tracks]):
            key = str(i + 1)
            if key in scores:
                val = scores[key]
                if val == "-" or val == "unknown" or val is None:
                    relevance[tid] = "-"
                else:
                    try:
                        relevance[tid] = float(val)
                    except (ValueError, TypeError):
                        relevance[tid] = "-"
            else:
                relevance[tid] = "-"

        return relevance

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw content: {repr(content[:500]) if content else 'None'}")
        return {tid: "-" for tid, _ in tracks_with_meta[:max_tracks]}
    except Exception as e:
        print(f"GPT-5 API error: {type(e).__name__}: {e}")
        return {tid: "-" for tid, _ in tracks_with_meta[:max_tracks]}


def compute_metrics(relevance_scores):
    """Compute metrics excluding unknown (-) scores."""
    known_scores = [s for s in relevance_scores if s != "-"]

    if not known_scores:
        return {
            "avg": None,
            "known_count": 0,
            "unknown_count": len(relevance_scores),
            "total": len(relevance_scores)
        }

    return {
        "avg": float(np.mean(known_scores)),
        "known_count": len(known_scores),
        "unknown_count": len(relevance_scores) - len(known_scores),
        "total": len(relevance_scores)
    }


def main():
    load_dotenv()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    print("\n=== Loading CLIP Models ===")

    from model import CaptionPlaylistCLIP
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    caption_dim = 3072
    playlist_dim = 64
    output_dim = 512

    # Load text_fast model
    model_text_fast = CaptionPlaylistCLIP(caption_dim, playlist_dim, output_dim).to(device)
    model_text_fast.load_state_dict(torch.load(f"{script_dir}/exp_text_fast/clip.pt", map_location=device))
    model_text_fast.eval()
    print(f"Loaded exp_text_fast model")

    # Load baseline model
    model_baseline = CaptionPlaylistCLIP(caption_dim, playlist_dim, output_dim).to(device)
    model_baseline.load_state_dict(torch.load(f"{script_dir}/exp_baseline/clip.pt", map_location=device))
    model_baseline.eval()
    print(f"Loaded exp_baseline model")

    # Load track data
    print("\n=== Loading Track Data ===")
    track_ids, track_embs = load_simgcl_track_data()
    print(f"Tracks: {len(track_ids)}")

    # Load track playlist counts for popularity filtering
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    track_df = pd.read_csv(os.path.join(base_dir, TRACK_CSV))
    track_counts = {str(row["track_id"]): row["count"] for _, row in track_df.iterrows()}
    print(f"Track counts loaded: {len(track_counts)}")

    # Project tracks through each model
    print("\n=== Projecting Tracks ===")
    print("Projecting through text_fast model...")
    track_embs_tf = project_tracks_through_clip(model_text_fast, track_embs, device)

    print("Projecting through baseline model...")
    track_embs_bl = project_tracks_through_clip(model_baseline, track_embs, device)

    test_keywords = KEYWORDS
    print(f"\n=== Testing with {len(test_keywords)} keywords ===")

    # Embed keywords
    keyword_embs = embed_keywords(openai_client, test_keywords)

    # Search results
    print("\n=== Searching Tracks (Direct) ===")
    search_results = {}
    all_needed_tracks = set()

    for keyword in tqdm(test_keywords, desc="Searching"):
        kw_emb = keyword_embs[keyword]

        # Direct track search: top-100 by similarity → top-10 by popularity
        tracks_tf = search_tracks_direct(kw_emb, model_text_fast, track_embs_tf, track_ids, device,
                                         track_counts, top_k_search=100, top_k_popular=10)
        tracks_bl = search_tracks_direct(kw_emb, model_baseline, track_embs_bl, track_ids, device,
                                         track_counts, top_k_search=100, top_k_popular=10)

        search_results[keyword] = {
            "text_fast": tracks_tf,  # list of (track_id, score)
            "baseline": tracks_bl
        }

        all_needed_tracks.update([tid for tid, _ in tracks_tf])
        all_needed_tracks.update([tid for tid, _ in tracks_bl])

    # Fetch metadata
    track_metadata = get_track_metadata_from_supabase(list(all_needed_tracks), supabase)
    print(f"Got metadata for {len(track_metadata)} tracks")

    # Verify with GPT-5
    print("\n=== Verifying with GPT-5 ===")

    results = {
        "exp_text_fast": {"all_scores": []},
        "exp_baseline": {"all_scores": []}
    }
    detailed_results = {}

    for keyword in tqdm(test_keywords, desc="Verifying"):
        detailed_results[keyword] = {
            "text_fast": {"tracks": [], "relevance": {}},
            "baseline": {"tracks": [], "relevance": {}}
        }

        # Verify text_fast
        tracks_tf = search_results[keyword]["text_fast"]
        tracks_with_meta_tf = [
            (tid, track_metadata.get(tid, {"display": f"Unknown ({tid})", "name": "Unknown", "artists": "Unknown"}))
            for tid, _ in tracks_tf
        ]

        relevance_tf = verify_with_gpt5(openai_client, keyword, tracks_with_meta_tf)

        for (tid, sim_score), (_, meta) in zip(tracks_tf, tracks_with_meta_tf):
            score = relevance_tf.get(tid, "-")
            detailed_results[keyword]["text_fast"]["tracks"].append({
                "track_id": tid,
                "display": meta.get("display"),
                "similarity": sim_score,
                "relevance": score
            })
            results["exp_text_fast"]["all_scores"].append(score)

        # Verify baseline
        tracks_bl = search_results[keyword]["baseline"]
        tracks_with_meta_bl = [
            (tid, track_metadata.get(tid, {"display": f"Unknown ({tid})", "name": "Unknown", "artists": "Unknown"}))
            for tid, _ in tracks_bl
        ]

        relevance_bl = verify_with_gpt5(openai_client, keyword, tracks_with_meta_bl)

        for (tid, sim_score), (_, meta) in zip(tracks_bl, tracks_with_meta_bl):
            score = relevance_bl.get(tid, "-")
            detailed_results[keyword]["baseline"]["tracks"].append({
                "track_id": tid,
                "display": meta.get("display"),
                "similarity": sim_score,
                "relevance": score
            })
            results["exp_baseline"]["all_scores"].append(score)

        # Rate limit
        time.sleep(0.5)

        # Progress
        tf_metrics = compute_metrics([t["relevance"] for t in detailed_results[keyword]["text_fast"]["tracks"]])
        bl_metrics = compute_metrics([t["relevance"] for t in detailed_results[keyword]["baseline"]["tracks"]])

        tf_avg = f"{tf_metrics['avg']:.2f}" if tf_metrics['avg'] is not None else "N/A"
        bl_avg = f"{bl_metrics['avg']:.2f}" if bl_metrics['avg'] is not None else "N/A"

        print(f"\n{keyword}: text_fast={tf_avg} (known:{tf_metrics['known_count']}), baseline={bl_avg} (known:{bl_metrics['known_count']})")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS (GPT-5 Evaluation)")
    print("=" * 70)

    for model_name, data in results.items():
        metrics = compute_metrics(data["all_scores"])

        print(f"\n{model_name}:")
        if metrics["avg"] is not None:
            print(f"  Average relevance: {metrics['avg']:.4f}")
        else:
            print(f"  Average relevance: N/A")
        print(f"  Known tracks: {metrics['known_count']}")
        print(f"  Unknown tracks: {metrics['unknown_count']}")
        print(f"  Total tracks: {metrics['total']}")

    # Save results
    output = {
        "model": "gpt-5",
        "n_keywords": len(test_keywords),
        "keywords": test_keywords,
        "summary": {
            name: compute_metrics(data["all_scores"])
            for name, data in results.items()
        },
        "detailed_results": detailed_results
    }

    with open(f"{script_dir}/gpt5_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {script_dir}/gpt5_evaluation_results.json")


if __name__ == "__main__":
    main()
