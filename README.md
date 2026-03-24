# 서비스의 개요

### **GOAL: 사용자의 engagement를 높일 수 있는 서비스를 만들자**

- **pinterest처럼 계속 탐색할 수 있는 UI**를 구현합니다.
    
    이 과정에서 리롤 애니메이션과 빈티지한 커버 처리를 통해 계속해서 디깅하고 싶은 화면을 구성하였습니다.
    
- 실제 유저들의 음악청취 패턴을 파악하기 위해 **플레이리스트 데이터를 실제 유저의 행동 로그**라고 가정하였습니다.
- **recommendation: simGCL 기반 트랙 임베딩 + 유저 시퀀스를 반영한 투타워 모델**
- **keyword search: clip 방식의 트랙임베딩과 키워드 임베딩의 alignment + 짧은 키워드 검색을 위한 전처리**

# Graph Embedding based Music Recommendation

### **DATA**

- Spotify 플레이리스트 데이터를 크롤링하여 구축
- **약 80K tracks, 40K playlists 규모**
- 각 **playlist를 하나의 user sequence로 가정**하여 추천 문제를 구성

### **Graph Embedding**

- **Item2Vec**
    - co-occurrence 기반으로 학습되지만
    - **그래프 이웃 구조를 직접 활용하지 못한다는 한계** 존재
- **LightGCN**
    - 사용자–아이템 이분 그래프 기반 임베딩
    - **Recall 성능은 개선**되었으나
    - 임베딩 공간이 uniform하지 않음
- **SimGCL**
    - 그래프에 **contrastive perturbation**을 적용한 방식
    - 특히 **long-tail 아이템 추천 성능이 크게 향상**
    - 임베딩 공간에서
        
        **uniformity 증가, alignment 개선**
        
    - 결과적으로 더 **균형 잡힌 representation** 형성
        
        $\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{x,y \sim p_{\text{data}}} \left[ e^{-2 \| f(x) - f(y) \|^2} \right]$
        
    - in-batch Negative로 학습하며 random Negative를 배치에 추가한 경우와도 비교

<img width="724" height="203" alt="image" src="https://github.com/user-attachments/assets/0d068e19-c472-407e-8b92-992f48bbc17e" />


| **Metric** | Item2vec | LightGCN | simGCL | **simGCL+randNeg** |
| --- | --- | --- | --- | --- |
| Recall@10 | 0.0486 | 0.1594 | 0.2051 | 0.2068 |
| Recall@20 | 0.0708 | 0.2253 | 0.2772 | 0.2828 |
| NDCG@10 | 0.0262 | 0.0898 | 0.1191 | 0.1199 |
| NDCG@20 | 0.0318 | 0.1063 | 0.1372 | 0.139 |

**그래프 모델은 유저 임베딩을 정적으로 학습하기 때문에, 현재 세션 기반의 동적 유저 표현이 불가능합니다.** 

이를 보완하기 위해 Two-Tower 모델을 도입하여 세션 내 트랙 임베딩으로부터 유저 임베딩을 실시간으로 생성합니다.

### **Two-Tower Model**

<img width="305" height="261" alt="image" src="https://github.com/user-attachments/assets/611940bc-fcc4-414c-a257-673c5bc5a8ab" />

- **Architecture**
    - **User Tower**
        - 2-layer Transformer Encoder
        - 사용자 **track sequence → user embedding**
    - **Item Tower**
        - 트랙 feature를 **MLP**로 변환하여 **item embedding** 생성
- **Training**
    - **In-batch negative sampling + infoNCE**
- **Serving**
    - item embedding을 **DB에 미리 저장**
    - **ANN 검색**으로 빠른 추천 수행
- **Evaluation (LOO)**
    
    <aside>
    
    Recall@10: **0.1982**
    
    Recall@20: **0.3127**
    
    Recall@50: **0.4631**
    
    </aside>
    
<img width="603" height="270" alt="image" src="https://github.com/user-attachments/assets/031e42ea-d57c-470c-ba95-91bd2b932e19" />
    

# Keyword search

**GOAL: 사용자가 검색한 무드/장르에 맞는 곡을 추천!**

- **Training**
    - 플레이리스트 제목은 대부분 길지만 (ex 눈 오는 겨울에 듣는 감성힙합/발라드, 70s Hits - The Biggest Hits of the 70's)
    - 유저의 검색은 짧은 키워드(ex 발라드, 크리스마스..) 위주가 될 것이라고 생각했습니다.
    - GPT api를 통해 플레이리스트 제목에서 짧은 키워드를 추출하여 이를 학습에 사용하여 키워드 검색을 강화하였습니다.
        
        70s Hits - The Biggest Hits of the 70's → ['70s', 'hits']
        
        이후 3번 이상 등장한 2300여개의 태그를 학습에 사용하였습니다.
        

- **inference**
    - 추론시에는 플레이리스트 임베딩과 같은 공간에 존재하는 트랙 임베딩에 프로젝션 모델을 사용합니다.
    - projected embedding을 저장하여 검색 시간 단축합니다.
    - ANN search(HNSW)
    - MMR 기반으로 검색의 다양성 확보

<img width="595" height="260" alt="image" src="https://github.com/user-attachments/assets/ebf0d2e5-3a16-461c-8da5-63b9d927ea03" />

**참고 논문**

https://arxiv.org/abs/1603.04259

https://arxiv.org/abs/2002.02126

https://arxiv.org/abs/2112.08679

https://arxiv.org/pdf/2103.00020
