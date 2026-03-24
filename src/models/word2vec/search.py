"""
Interactive track search and recommendation using Word2Vec embeddings.

Similar to compare_recommendations.py but for Word2Vec model.
"""

import numpy as np
import pickle
from pathlib import Path
from gensim.models import Word2Vec
import pandas as pd


def load_model_and_data(model_dir, csv_path):
    """Load Word2Vec model and track metadata."""
    model_dir = Path(model_dir)

    # Load model
    model = Word2Vec.load(str(model_dir / "word2vec.model"))
    print(f"Loaded Word2Vec model: {len(model.wv):,} tracks, {model.wv.vector_size}d")

    # Load track info
    with open(model_dir.parent / "data" / "track_info.pkl", 'rb') as f:
        track_info = pickle.load(f)

    # Load CSV for search (has track names)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} tracks from CSV")

    # Create id to count mapping
    id_to_count = dict(zip(df['track_id'], df['count']))

    return model, track_info, df, id_to_count


def search_tracks(df, query):
    """Search tracks by name."""
    mask = df['track'].str.lower().str.contains(query.lower(), na=False)
    results = df[mask][['track_id', 'track', 'artist']].drop_duplicates().head(20)
    return results


def get_similar_tracks(model, track_id, track_info, id_to_count, top_k=10):
    """Get similar tracks using Word2Vec."""
    if track_id not in model.wv:
        return None

    similar = model.wv.most_similar(track_id, topn=top_k)

    results = []
    for sim_id, score in similar:
        info = track_info.get(sim_id, {})
        results.append({
            'rank': len(results) + 1,
            'id': sim_id,
            'name': info.get('track', 'Unknown'),
            'artist': info.get('artist', 'Unknown'),
            'score': score,
            'popularity': id_to_count.get(sim_id, 0)
        })

    return results


def get_similar_from_multiple(model, track_ids, track_info, id_to_count, top_k=10):
    """Get similar tracks from multiple seed tracks (average embedding)."""
    # Filter valid track IDs
    valid_ids = [tid for tid in track_ids if tid in model.wv]
    if not valid_ids:
        return None

    # Compute average embedding
    embeddings = [model.wv[tid] for tid in valid_ids]
    avg_emb = np.mean(embeddings, axis=0)

    # Find similar
    similar = model.wv.similar_by_vector(avg_emb, topn=top_k + len(valid_ids))

    # Filter out seed tracks
    seed_set = set(valid_ids)
    results = []
    for sim_id, score in similar:
        if sim_id in seed_set:
            continue
        if len(results) >= top_k:
            break

        info = track_info.get(sim_id, {})
        results.append({
            'rank': len(results) + 1,
            'id': sim_id,
            'name': info.get('track', 'Unknown'),
            'artist': info.get('artist', 'Unknown'),
            'score': score,
            'popularity': id_to_count.get(sim_id, 0)
        })

    return results


def main():
    model_dir = "word2vec/outputs"
    csv_path = "data/csvs/track_playlist_counts_min5_win10.csv"

    print("Loading Word2Vec model and data...")
    model, track_info, df, id_to_count = load_model_and_data(model_dir, csv_path)

    print("\n" + "=" * 60)
    print("Word2Vec Track Recommendation")
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

    selected_tracks = []

    while True:
        if continuous_mode and selected_tracks:
            print(f"\n현재 선택된 곡 ({len(selected_tracks)}개): ", end="")
            names = [track_info.get(t, {}).get('track', t)[:20] for t in selected_tracks[-3:]]
            print(", ".join(names))

        print("\n검색어를 입력하세요 (quit으로 종료):")
        query = input("> ").strip()

        if query.lower() == 'quit':
            break

        if query.lower() == 'clear' and continuous_mode:
            selected_tracks = []
            print("리스트가 초기화되었습니다.")
            continue

        if query.lower() == 'list' and continuous_mode:
            if not selected_tracks:
                print("선택된 곡이 없습니다.")
            else:
                print(f"\n선택된 곡 목록 ({len(selected_tracks)}개):")
                for i, tid in enumerate(selected_tracks):
                    info = track_info.get(tid, {})
                    print(f"  {i+1}. {info.get('track', tid)} - {info.get('artist', 'Unknown')}")
            continue

        if not query:
            continue

        # Search
        results = search_tracks(df, query)

        if len(results) == 0:
            print("검색 결과가 없습니다.")
            continue

        print(f"\n검색 결과 ({len(results)}개):")
        print("-" * 60)
        for i, (_, row) in enumerate(results.iterrows()):
            in_vocab = "✓" if row['track_id'] in model.wv else "✗"
            print(f"  [{i}] {in_vocab} {row['track']} - {row['artist']}")

        print("\n선택할 번호를 입력하세요:")
        try:
            choice = int(input("> "))
            selected = results.iloc[choice]
        except (ValueError, IndexError):
            print("잘못된 선택입니다.")
            continue

        track_id = selected['track_id']

        if track_id not in model.wv:
            print("이 곡은 Word2Vec 어휘에 없습니다. (min_count 미만)")
            continue

        info = track_info.get(track_id, {})
        print(f"\n선택한 곡: {info.get('track', track_id)} - {info.get('artist', 'Unknown')}")

        if continuous_mode:
            selected_tracks.append(track_id)
            print(f"곡이 추가되었습니다. (총 {len(selected_tracks)}곡)")
            recs = get_similar_from_multiple(model, selected_tracks, track_info, id_to_count)
        else:
            recs = get_similar_tracks(model, track_id, track_info, id_to_count)

        if recs is None:
            print("추천을 생성할 수 없습니다.")
            continue

        # Display
        print("\n" + "=" * 80)
        print("Word2Vec Recommendations")
        print("=" * 80)
        for r in recs:
            print(f"  {r['rank']:2}. {r['name'][:30]:<30} - {r['artist'][:15]:<15} "
                  f"(score:{r['score']:.3f}, pop:{r['popularity']})")

        avg_pop = np.mean([r['popularity'] for r in recs])
        print("-" * 80)
        print(f"평균 인기도: {avg_pop:.1f}")


if __name__ == "__main__":
    main()
