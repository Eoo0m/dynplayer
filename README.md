

Overview

Dynplayer는 Spotify 플레이리스트 co-occurrence 데이터를 implicit feedback으로 활용하여
트랙 임베딩을 학습하고, 키워드·텍스트 기반 음악 추천까지 확장한
임베딩 기반 음악 추천 시스템입니다.
	•	InfoNCE 기반 contrastive learning으로 트랙 representation 학습
	•	popularity bias(hubness) 문제를 uniformity loss로 완화
	•	FastAPI + pgvector(HNSW) 기반 실서비스 배포

⸻

Key Results

Task	Metric	Score
Track Retrieval	Recall@10	0.1919
Track Retrieval	NDCG@10	0.2871
Keyword → Playlist Retrieval	Recall@10	0.2693
Keyword → Playlist Retrieval	Recall@20	0.3784
Genre Linear Eval (51 genres)	Top-1 Acc	0.8177
	Top-5 Acc	0.9751


⸻

Data Processing
	•	Spotify Playlist Crawling
	•	~40,000 playlists
	•	~1.7M track occurrences
	•	Playlist 내 co-occurrence 기반 이웃 정의
	•	window size = 10 기준
	•	이웃 수 상위 5% 약 29,000곡만 학습 대상으로 사용

⸻

Track Embedding Learning (Contrastive Learning)

Problem Setting
	•	Playlist를 user로 간주하지 않고 context 집합으로 해석
	•	같은 playlist에 등장한 트랙 → positive pair
	•	다른 트랙 → negative

Training Strategy
	•	InfoNCE loss
	•	anchor 트랙당 동일 playlist 내 positive 1개 샘플링
	•	batch 내 implicit negative 구성

Anchor Track ↔ Positive Track (same playlist)

Popularity Bias (Hubness)
	•	인기 트랙이 embedding space에서 중심으로 쏠리는 문제 발생
	•	평균 cosine similarity가 높아 추천에 과도하게 등장

Solution: Uniformity Loss

\mathcal{L}_{uniform}(f;t)
= \log \mathbb{E}_{x,y \sim p_{data}}
\left[e^{-t\|f(x)-f(y)\|^2}\right]
	•	embedding 공간 전반의 분산 유지
	•	인기곡/비인기곡 간 유사도 분포 차이 감소

Uniformity Weight 실험 결과

weight	NDCG@10	Recall@10	NDCG@20	Recall@20
u00	0.2827	0.1886	0.3557	0.2781
u05	0.2871	0.1919	0.3580	0.2789
u10	0.2848	0.1909	0.3569	0.2796


⸻

Evaluation Protocol
	•	Playlist 단위 train / test split
	•	일부 트랙을 test로 제거
	•	Train 트랙 평균 → playlist embedding 생성
	•	cosine similarity 기반 retrieval
	•	train 트랙은 추천 대상에서 masking

Metrics
	•	Recall@K (playlist별 정규화, macro average)
	•	NDCG@K

임베딩 representation 자체 성능을 평가하기 위한 offline retrieval 평가

⸻

Multimodal Keyword Retrieval (CLIP-style)

Motivation
	•	트랙 임베딩 ↔ 캡션 직접 학습 시
	•	인기 트랙이 다양한 캡션에 의해 semantic blur 발생

Approach
	•	Playlist caption ↔ Playlist embedding CLIP 방식 학습
	•	이후 keyword → playlist → track 구조로 추천

Model Design
	•	Two-layer projection head
	•	Residual connection 적용

⸻

Recommendation Serving

Similarity-Weighted Collaborative Filtering
	1.	텍스트 입력 → 텍스트 임베딩
	2.	Top-50 유사 playlist 검색
	3.	playlist 내 트랙 등장 빈도 × 유사도 가중합
	4.	최종 트랙 추천

⸻

System Architecture
	•	Frontend: Cloudflare Pages
	•	Backend: FastAPI
	•	Model: PyTorch (custom embedding / projection)
	•	Vector DB: Supabase (pgvector + HNSW)

User Query
 → Text Embedding
 → ANN Playlist Retrieval
 → Weighted Track Aggregation
 → Recommendation

