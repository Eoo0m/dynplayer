# Feature 추가시 그래프 임베딩 개선 가능성 분석

## 질문: 오디오 + 가사 임베딩을 사용하면 그래프 방법이 나아질까?

---

## 1. Feature가 그래프 방법에 도움이 되는 경우

### Pinterest PinSage의 성공 사례

**원본 PinSage 구조**:
```python
class PinSageWithFeatures:
    def __init__(self):
        # Multi-modal features
        self.image_cnn = ResNet50()      # 이미지 특징
        self.text_encoder = BERT()       # 텍스트 특징
        self.metadata_mlp = MLP()        # 카테고리, 가격 등

    def get_node_features(self, item):
        # Rich features 결합
        img_feat = self.image_cnn(item.image)      # [2048]
        txt_feat = self.text_encoder(item.text)    # [768]
        meta_feat = self.metadata_mlp(item.meta)   # [128]

        # Concatenate
        features = concat([img_feat, txt_feat, meta_feat])  # [2944]

        # Graph propagation
        return self.graph_conv(features)
```

**왜 성공했나?**:
1. **Rich Features**: 2944차원의 풍부한 특징
2. **Cold-start 해결**: 새 핀도 이미지/텍스트로 표현 가능
3. **의미적 유사도**: Feature 공간에서 이미 유사함
4. **그래프 보완**: Feature만으로 부족한 부분을 그래프가 보완

---

## 2. 우리 태스크에 Feature를 추가하면?

### 현재 상황
```python
# 현재: ID만 사용
embedding = nn.Embedding(num_items, 64)  # [81,701 x 64]
# → 순수하게 co-occurrence 패턴만 학습
```

### Feature 추가시
```python
# 제안: ID + Audio + Lyrics
id_emb = nn.Embedding(num_items, 64)           # [64]
audio_emb = load_audio_features()              # [512] (frozen)
lyrics_emb = load_lyrics_features()            # [768] (frozen)

# Projection layers
audio_proj = nn.Linear(512, 64)                # 512 → 64
lyrics_proj = nn.Linear(768, 64)               # 768 → 64

# Combined features
features = concat([id_emb, audio_proj, lyrics_proj])  # [192]
```

---

## 3. Feature 추가가 도움이 되는 시나리오

### ✅ Scenario 1: Cold-start (새로운 트랙)

**문제**:
```python
# 학습 중 본 적 없는 새 트랙
new_track = "new_song_2024"
# 현재: ID embedding이 랜덤 초기화 → 추천 불가
```

**해결**:
```python
# Audio + Lyrics feature 활용
audio_feat = extract_audio(new_track)    # [512]
lyrics_feat = extract_lyrics(new_track)  # [768]

# 비슷한 음악적 특징을 가진 트랙 찾기
similar_tracks = find_similar_by_features(audio_feat, lyrics_feat)
```

**그래프 기여**:
- Feature로 초기 표현
- 그래프 전파로 이웃 정보 통합
- Co-occurrence + Content 결합

---

### ✅ Scenario 2: Long-tail 트랙 (데이터 부족)

**문제**:
```python
# 플레이리스트에 5번만 등장한 트랙
rare_track = "indie_unknown_song"
# 현재: Co-occurrence 정보 부족 → 학습 어려움
```

**해결**:
```python
# Audio similarity로 보완
similar_mainstream_tracks = find_by_audio_similarity(rare_track)
# → 메이저 트랙과의 유사도로 임베딩 개선
```

**그래프 기여**:
- 희소한 co-occurrence를 feature로 보완
- Content-based + Collaborative filtering 결합

---

### ✅ Scenario 3: 음악적 유사도 반영

**문제**:
```python
# 다른 플레이리스트에 있지만 음악적으로 유사한 트랙
track_a = "Classical Piano Sonata #1"
track_b = "Classical Piano Sonata #2"
# 현재: 같은 플레이리스트에 없으면 유사도 낮음
```

**해결**:
```python
# Audio feature로 음악적 유사도 캡처
audio_similarity = cosine_sim(audio_a, audio_b)  # 높음
# → Co-occurrence 없어도 유사하다고 학습
```

**그래프 기여**:
- Feature 기반 유사도
- 그래프로 컨텍스트 정보 추가
- 더 풍부한 표현

---

## 4. 그래프 vs Contrastive: Feature 추가 비교

### 방법 1: 그래프 + Features

```python
class GraphWithFeatures:
    def __init__(self):
        self.id_emb = nn.Embedding(num_items, 64)
        self.audio_proj = nn.Linear(512, 64)
        self.lyrics_proj = nn.Linear(768, 64)
        self.graph_conv = GraphConvLayer()

    def forward(self, user, items):
        # 1. Feature 결합
        id_feat = self.id_emb(items)
        audio_feat = self.audio_proj(audio_emb[items])
        lyrics_feat = self.lyrics_proj(lyrics_emb[items])
        combined = concat([id_feat, audio_feat, lyrics_feat])  # [192]

        # 2. Graph propagation (여전히 over-smoothing 위험)
        propagated = self.graph_conv(combined)  # 3-layer

        # 3. User-item scoring
        user_emb = self.user_embedding(user)
        scores = user_emb @ propagated.T

        return scores
```

**장점**:
- ✅ Cold-start 개선
- ✅ Long-tail 개선
- ✅ Content + Collaborative

**단점**:
- ❌ **Over-smoothing 여전히 존재**: 3-layer 전파
- ❌ **User embedding overfitting**: User별 학습
- ❌ **복잡도 증가**: Feature + Graph + 두 Loss
- ❌ **학습 느림**: 여전히 그래프 전파 필요

---

### 방법 2: Contrastive + Features

```python
class ContrastiveWithFeatures:
    def __init__(self):
        self.id_emb = nn.Embedding(num_items, 64)
        self.audio_proj = nn.Linear(512, 64)
        self.lyrics_proj = nn.Linear(768, 64)

    def forward(self, items):
        # Feature 결합만 (그래프 전파 없음)
        id_feat = self.id_emb(items)
        audio_feat = self.audio_proj(audio_emb[items])
        lyrics_feat = self.lyrics_proj(lyrics_emb[items])
        combined = concat([id_feat, audio_feat, lyrics_feat])  # [192]

        return F.normalize(combined, dim=1)
```

**InfoNCE Loss**:
```python
# Co-occurrence + Audio similarity + Lyrics similarity
def multi_modal_loss(anchor, positive, negatives):
    # 1. Co-occurrence 기반
    co_occur_loss = infonce(anchor, positive, negatives)

    # 2. Audio similarity 보조 loss (optional)
    audio_sim_loss = mse(audio_proj(anchor), audio_proj(positive))

    # 3. Combined
    return co_occur_loss + 0.1 * audio_sim_loss
```

**장점**:
- ✅ **Over-smoothing 없음**: 그래프 전파 제거
- ✅ **Inductive**: User embedding 불필요
- ✅ **단순함**: 단일 목적 함수
- ✅ **빠름**: 그래프 전파 없음
- ✅ **Cold-start 해결**: Feature로 표현
- ✅ **Content + Collaborative**: 자연스럽게 결합

**단점**:
- ⚠️ Feature projection layer 학습 필요
- ⚠️ Feature 차원 증가 (64 → 192)

---

## 5. 실험 예측: Feature 추가시 성능 변화

### 현재 성능 (ID만)
```
Contrastive:  NDCG@10 = 0.25
SimGCL:       NDCG@10 = 0.19
```

### Feature 추가 후 예측

| 모델 | ID만 | + Audio | + Audio + Lyrics |
|------|------|---------|------------------|
| **Contrastive** | 0.25 | **0.28** | **0.30** |
| **SimGCL** | 0.19 | 0.22 | 0.24 |
| **Gap** | 0.06 | 0.06 | 0.06 |

**예측 근거**:
1. **둘 다 개선**: Feature는 모든 방법에 도움
2. **Gap 유지**: Over-smoothing, User overfitting은 여전히 존재
3. **Contrastive 여전히 우수**: 구조적 장점 유지

---

## 6. Feature 추가시 주의사항

### 문제 1: Feature Dominance

```python
# Audio feature가 너무 강하면
combined = concat([id_emb, audio_proj, lyrics_proj])
#                  [64]    [64]        [64]

# Co-occurrence 정보가 묻힐 수 있음
# → Audio만으로 추천 (장르 기반 추천)
```

**해결**:
```python
# Weighted combination
combined = concat([
    id_emb * 2.0,        # Co-occurrence 강조
    audio_proj * 0.5,    # Audio 보조
    lyrics_proj * 0.3    # Lyrics 보조
])
```

---

### 문제 2: Feature Overfitting

```python
# Audio feature가 고정되어 있으면
audio_emb = load_pretrained()  # Frozen

# 학습 데이터의 편향을 그대로 가져옴
# 예: 특정 장르에 편향된 feature
```

**해결**:
```python
# Fine-tuning option
audio_proj = nn.Linear(512, 64)
audio_proj.requires_grad = True  # 학습 가능

# 또는 Adapter
audio_adapter = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
)
```

---

### 문제 3: Computational Cost

```python
# Feature 차원 증가 → 메모리 증가
ID only:              64 dim
+ Audio:             128 dim
+ Audio + Lyrics:    192 dim

# 학습 시간도 증가
```

---

## 7. 결론: Feature 추가하면 그래프가 나아질까?

### ✅ 개선되는 부분:

1. **Cold-start**: 새 트랙도 표현 가능
2. **Long-tail**: 희소한 트랙도 feature로 보완
3. **Content similarity**: 음악적 유사도 반영
4. **일반화**: Feature 공간의 의미 활용

### ❌ 여전히 남는 문제:

1. **Over-smoothing**: 그래프 전파의 근본적 문제
2. **User overfitting**: Transductive learning의 한계
3. **복잡도**: 여전히 Contrastive보다 복잡
4. **학습 속도**: 그래프 전파 오버헤드

---

## 8. 추천 전략

### 현재 태스크에 최적인 방법:

```python
class BestApproach:
    """
    Contrastive Learning + Multi-modal Features
    """
    def __init__(self):
        # Simple embedding + features
        self.id_emb = nn.Embedding(num_items, 64)
        self.audio_proj = nn.Linear(512, 64)
        self.lyrics_proj = nn.Linear(768, 64)

    def forward(self, items):
        # Weighted combination
        id_feat = self.id_emb(items) * 2.0
        audio_feat = self.audio_proj(audio_emb[items]) * 0.5
        lyrics_feat = self.lyrics_proj(lyrics_emb[items]) * 0.3

        combined = concat([id_feat, audio_feat, lyrics_feat])
        return F.normalize(combined, dim=1)

    def train(self):
        # InfoNCE with co-occurrence
        loss = infonce_loss(anchor, positive, negatives)

        # Optional: Audio/Lyrics alignment loss
        # aux_loss = align_loss(audio_proj, lyrics_proj)
        # total_loss = loss + 0.1 * aux_loss
```

**이유**:
1. ✅ Feature 활용 (Cold-start, Long-tail 해결)
2. ✅ Over-smoothing 없음 (그래프 전파 제거)
3. ✅ Inductive (일반화 우수)
4. ✅ 단순하고 빠름
5. ✅ 확장 가능 (더 많은 feature 추가 쉬움)

---

## 9. 실험 계획

### Step 1: Contrastive + Audio
```bash
python contrastive_learning/train_with_audio.py \
    --dataset min5_win10 \
    --audio-path data/audio_embeddings/audio_embeddings.npz \
    --audio-weight 0.5
```

### Step 2: Contrastive + Audio + Lyrics
```bash
python contrastive_learning/train_with_multimodal.py \
    --dataset min5_win10 \
    --audio-path data/audio_embeddings/audio_embeddings.npz \
    --lyrics-path data/lyrics_embeddings/lyrics_embeddings.npz \
    --audio-weight 0.5 \
    --lyrics-weight 0.3
```

### Step 3: SimGCL + Audio (비교용)
```bash
python simgcl/train_with_audio.py \
    --dataset min5_win10 \
    --audio-path data/audio_embeddings/audio_embeddings.npz
```

### 예상 결과:
```
Method                      NDCG@10  학습시간
─────────────────────────  ────────  ────────
Contrastive (ID only)        0.25     5min
Contrastive + Audio          0.28     6min  ← 최고 성능
Contrastive + Audio+Lyrics   0.30     7min  ← 최고 성능
SimGCL (ID only)             0.19    20min
SimGCL + Audio               0.22    25min  ← 여전히 낮음
```

---

## 최종 답변

**Q: 오디오 + 가사 임베딩을 사용하면 그래프 방법이 나아질까?**

**A: 네, 개선되지만 여전히 Contrastive보다 낮을 것입니다.**

**이유**:
1. Feature는 모든 방법에 도움이 됨 (Cold-start, Long-tail 해결)
2. **하지만** 그래프의 구조적 문제는 여전히 존재:
   - Over-smoothing (3-layer 전파)
   - User embedding overfitting
   - 복잡도와 학습 속도
3. Contrastive도 같은 feature를 사용하면 더 개선됨
4. Feature + Contrastive가 최적 조합

**추천**:
- **단기**: Contrastive + Audio 먼저 시도
- **장기**: Contrastive + Audio + Lyrics + 다른 feature들
- **비교**: SimGCL + Audio도 실험해서 Gap 확인
