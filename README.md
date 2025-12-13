## Dynplayer

Implicit Feedback 기반 임베딩 음악 추천 시스템

Dynplayer는 Spotify 플레이리스트 co-occurrence 데이터를 implicit feedback으로 활용하여
트랙 임베딩을 학습하고, 키워드·텍스트 기반 음악 추천까지 확장한
임베딩 기반 음악 추천 시스템입니다.
	•	InfoNCE 기반 contrastive learning으로 트랙 representation 학습
	•	Popularity bias (hubness) 문제를 uniformity loss로 완화
	•	FastAPI + pgvector(HNSW) 기반 실서비스 배포

⸻

### Key Results

Task	Metric	Score
Playlist → Track Retrieval  Recall@10	0.1919
Playlist → Track Retrieval	NDCG@10	0.2871
Keyword → Playlist Retrieval	Recall@10	0.2693
Keyword → Playlist Retrieval	Recall@20	0.3784
Genre Linear Evaluation (51 genres)	Top-1 Accuracy	0.8177
	Top-5 Accuracy	0.9751


⸻

### Data Processing
	•	Spotify Playlist Crawling
	•	~40,000 playlists
	•	~1.7M track occurrences
	•	Playlist 내 co-occurrence 기반 이웃 정의
	•	window size = 10
	•	이웃 수 상위 **5% (약 29,000곡)**만 학습 대상으로 사용

⸻

### Track Embedding Learning

(Contrastive Learning)

Problem Setting
	•	Playlist를 user가 아닌 context 집합으로 해석
	•	같은 playlist에 등장한 트랙 → positive pair
	•	다른 트랙 → negative

⸻

Training Strategy
	•	InfoNCE loss
	•	anchor 트랙당 동일 playlist 내 positive 1개 샘플링
	•	batch 내 트랙들을 implicit negative로 활용

Anchor Track  ↔  Positive Track  (same playlist)


⸻

Popularity Bias (Hubness)
	•	인기 트랙이 embedding space 중심으로 몰리며
평균 cosine similarity가 과도하게 높아지는 문제 발생

→ Uniformity loss를 추가하여 embedding 공간 분산 유지


