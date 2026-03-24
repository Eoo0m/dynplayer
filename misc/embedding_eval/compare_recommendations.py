"""
Compare recommendations from two different embeddings (SimGCL LOO vs SimGCL Weighted)

Usage:
    python embedding_eval/compare_recommendations.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_model_data(model_dir):
    """Load track embeddings and mappings for a model"""
    outputs_dir = Path(model_dir) / "outputs" / "min5_win10"

    track_emb = np.load(outputs_dir / "model_loo_track_embeddings.npy")
    track_ids = np.load(outputs_dir / "model_loo_track_ids.npy", allow_pickle=True)

    # Create id to idx mapping
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    print(f"Loaded {model_dir}: {track_emb.shape}")
    return track_emb, track_ids, id_to_idx


def search_tracks(df, query):
    """Search tracks by name (case-insensitive partial match)"""
    mask = df['track'].str.lower().str.contains(query.lower(), na=False)
    results = df[mask][['track_id', 'track', 'artist']].head(20)
    return results


def get_similar_tracks(query_emb, track_emb, track_ids, df, id_to_count, exclude_indices=None, top_k=10):
    """Get top-k similar tracks using dot product"""
    # Dot product scores
    scores = track_emb @ query_emb

    # Exclude specified indices (query tracks)
    if exclude_indices:
        for idx in exclude_indices:
            scores[idx] = -np.inf

    # Top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        tid = track_ids[idx]
        track_info = df[df['track_id'] == tid]
        if len(track_info) > 0:
            name = track_info['track'].values[0]
            artist = track_info['artist'].values[0]
        else:
            name = "Unknown"
            artist = "Unknown"
        popularity = id_to_count.get(tid, 0)
        results.append({
            'rank': len(results) + 1,
            'id': tid,
            'name': name,
            'artist': artist,
            'score': scores[idx],
            'popularity': popularity
        })

    return results


def main():
    # Load track metadata
    csv_path = "data/csvs/track_playlist_counts_min5_win10.csv"
    print(f"Loading track metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} tracks")

    # Create track_id to popularity (count) mapping
    id_to_count = dict(zip(df['track_id'], df['count']))

    # Load both models
    print("\nLoading embeddings...")
    loo_emb, loo_ids, loo_id_to_idx = load_model_data("simgcl_loo")
    weighted_emb, weighted_ids, weighted_id_to_idx = load_model_data("simgcl_weighted")

    print("\n" + "=" * 60)
    print("Track Recommendation Comparison")
    print("SimGCL LOO vs SimGCL Weighted")
    print("=" * 60)

    # Mode selection
    print("\n모드를 선택하세요:")
    print("  [1] 단일 모드 - 한 곡씩 검색하여 추천")
    print("  [2] 연속 모드 - 여러 곡을 추가하여 평균 기반 추천")
    mode = input("> ").strip()
    continuous_mode = (mode == '2')

    if continuous_mode:
        print("\n연속 모드 활성화됨")
        print("  - 곡을 검색/선택하면 리스트에 추가됩니다")
        print("  - 'clear': 리스트 초기화")
        print("  - 'list': 현재 리스트 보기")
        print("  - 'quit': 종료")

    # Track lists for continuous mode
    selected_tracks = []  # [(track_id, track_name, track_artist)]
    loo_indices = []
    weighted_indices = []

    while True:
        if continuous_mode and selected_tracks:
            print(f"\n현재 선택된 곡 ({len(selected_tracks)}개): ", end="")
            print(", ".join([f"{t[1]}" for t in selected_tracks[-3:]]))  # Show last 3
            if len(selected_tracks) > 3:
                print(f"  ... 외 {len(selected_tracks) - 3}곡")

        print("\n검색어를 입력하세요 (quit으로 종료):")
        query = input("> ").strip()

        if query.lower() == 'quit':
            break

        if query.lower() == 'clear' and continuous_mode:
            selected_tracks = []
            loo_indices = []
            weighted_indices = []
            print("리스트가 초기화되었습니다.")
            continue

        if query.lower() == 'list' and continuous_mode:
            if not selected_tracks:
                print("선택된 곡이 없습니다.")
            else:
                print(f"\n선택된 곡 목록 ({len(selected_tracks)}개):")
                for i, (tid, name, artist) in enumerate(selected_tracks):
                    print(f"  {i+1}. {name} - {artist}")
            continue

        if not query:
            continue

        # Search tracks
        results = search_tracks(df, query)

        if len(results) == 0:
            print("검색 결과가 없습니다.")
            continue

        print(f"\n검색 결과 ({len(results)}개):")
        print("-" * 60)
        for i, (_, row) in enumerate(results.iterrows()):
            print(f"  [{i}] {row['track']} - {row['artist']}")

        print("\n선택할 번호를 입력하세요:")
        try:
            choice = int(input("> "))
            selected = results.iloc[choice]
        except (ValueError, IndexError):
            print("잘못된 선택입니다.")
            continue

        track_id = selected['track_id']
        track_name = selected['track']
        track_artist = selected['artist']

        print(f"\n선택한 곡: {track_name} - {track_artist}")
        print(f"ID: {track_id}")

        # Check if track exists in both models
        if track_id not in loo_id_to_idx:
            print("이 곡은 SimGCL LOO 임베딩에 없습니다.")
            continue
        if track_id not in weighted_id_to_idx:
            print("이 곡은 SimGCL Weighted 임베딩에 없습니다.")
            continue

        loo_idx = loo_id_to_idx[track_id]
        weighted_idx = weighted_id_to_idx[track_id]

        if continuous_mode:
            # Add to list
            selected_tracks.append((track_id, track_name, track_artist))
            loo_indices.append(loo_idx)
            weighted_indices.append(weighted_idx)
            print(f"곡이 추가되었습니다. (총 {len(selected_tracks)}곡)")

            # Compute average embedding
            loo_query_emb = loo_emb[loo_indices].mean(axis=0)
            weighted_query_emb = weighted_emb[weighted_indices].mean(axis=0)
            exclude_loo = set(loo_indices)
            exclude_weighted = set(weighted_indices)
        else:
            # Single mode - use single track embedding
            loo_query_emb = loo_emb[loo_idx]
            weighted_query_emb = weighted_emb[weighted_idx]
            exclude_loo = {loo_idx}
            exclude_weighted = {weighted_idx}

        # Get recommendations from both models
        loo_recs = get_similar_tracks(loo_query_emb, loo_emb, loo_ids, df, id_to_count, exclude_loo, top_k=10)
        weighted_recs = get_similar_tracks(weighted_query_emb, weighted_emb, weighted_ids, df, id_to_count, exclude_weighted, top_k=10)

        # Display side by side
        print("\n" + "=" * 120)
        print(f"{'SimGCL LOO':<60} | {'SimGCL Weighted':<60}")
        print("=" * 120)

        for i in range(10):
            loo_r = loo_recs[i]
            weighted_r = weighted_recs[i]

            loo_str = f"{loo_r['rank']}. {loo_r['name'][:22]} - {loo_r['artist'][:12]} (s:{loo_r['score']:.2f}, p:{loo_r['popularity']})"
            weighted_str = f"{weighted_r['rank']}. {weighted_r['name'][:22]} - {weighted_r['artist'][:12]} (s:{weighted_r['score']:.2f}, p:{weighted_r['popularity']})"

            print(f"{loo_str:<60} | {weighted_str:<60}")

        # Check overlap
        loo_ids_set = set(r['id'] for r in loo_recs)
        weighted_ids_set = set(r['id'] for r in weighted_recs)
        overlap = loo_ids_set & weighted_ids_set

        # Calculate average popularity
        loo_avg_pop = np.mean([r['popularity'] for r in loo_recs])
        weighted_avg_pop = np.mean([r['popularity'] for r in weighted_recs])

        print("-" * 120)
        print(f"공통 추천곡: {len(overlap)}개 / 10개")
        print(f"평균 인기도 - LOO: {loo_avg_pop:.1f}, Weighted: {weighted_avg_pop:.1f}")


if __name__ == "__main__":
    main()
