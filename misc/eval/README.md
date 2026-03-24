# Evaluation Scripts

평가 관련 스크립트 모음

## 폴더 구조

```
eval/
├── perplexity/          # Perplexity API 기반 키워드 검색 평가
├── season/              # 계절 키워드 유사도 분석
├── popularity/          # 인기도 기반 Recall 분석
├── embedding/           # 임베딩 비교 및 Recall 평가
└── alignment/           # Alignment & Uniformity 시각화
```

## 각 폴더 설명

### perplexity/
키워드 기반 트랙 검색 후 Perplexity API로 relevance 검증
- `evaluate_with_perplexity.py`: 155개 키워드로 CLIP 모델 평가
- `perplexity_evaluation_results.json`: 평가 결과 (키워드별 추천 곡 포함)

### season/
계절 키워드 (봄, 여름, 가을, 겨울) 트랙 유사도 분석
- `analyze_season_similarity.py`: 계절별 트랙/텍스트 임베딩 유사도 비교

### popularity/
트랙 인기도에 따른 Recall 성능 분석
- `compare_recall_by_popularity.py`: 인기도 구간별 Recall 비교
- `visualize_popularity.py`: 인기도 분포 시각화
- `popularity_visualization_*.png`: 시각화 결과

### embedding/
임베딩 품질 및 추천 성능 평가
- `evaluate_recall.py`: Recall@K 평가
- `compare_recommendations.py`: 추천 결과 비교
- `compare_embeddings.py`: 임베딩 유사도 비교
- `compare_models.py`: 모델 간 성능 비교

### alignment/
Alignment & Uniformity 메트릭 시각화
- `visualize_alignment_uniformity.py`: A&U 시각화
- `*_alignment_uniformity.png`: 시각화 결과
