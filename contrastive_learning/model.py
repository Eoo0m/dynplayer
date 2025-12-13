from collections import defaultdict
import numpy as np
import torch
from torch import nn
import random


def build_graph(df):
    """
    neighbors 컬럼을 이용해 track 간 양성 그래프 생성.

    Input:
        df: 'track_id', 'neighbors' 컬럼 포함 DataFrame
    Return:
        tids: 트랙 ID 리스트
        pos_sets: 각 트랙별 양성 이웃 dict
        pos_count: 각 트랙별 양성 개수 dict
    """
    tids = df["track_id"].drop_duplicates().to_list()
    tid_set = set(tids)

    pos_sets = defaultdict(set)
    for tid, neighbor_str in zip(df["track_id"], df["neighbors"]):
        for nb in neighbor_str.split("|"):
            nb = nb.strip()
            if nb and nb != tid and nb in tid_set:
                pos_sets[tid].add(nb)
                pos_sets[nb].add(tid)

    pos_count = {tid: len(pos_sets.get(tid, set())) for tid in tid_set}
    return tids, pos_sets, pos_count


class PositiveBatcher:
    """
    앵커 + 양성(이웃)을 섞어 배치를 생성.


    Input:
        anchors: 앵커 트랙 리스트
        pos_sets: 양성 관계 dict
        pos_count: 각 트랙별 양성 개수 dict
        batch_size: 배치 크기
    Yield:
        track_id 리스트 (길이=batch_size)
    """

    def __init__(
        self, anchors, pos_sets, pos_count, batch_size=512, fill_pool=None, seed=42
    ):
        self.anchors = anchors
        self.pos_sets = pos_sets
        self.pos_count = pos_count
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.fill_pool = fill_pool or anchors

    def __iter__(self):
        # 균등하게 앵커 선택 (모든 앵커를 한 번씩)
        A = self.anchors.copy()
        self.rng.shuffle(A)
        cur = []

        for a in A:
            cur.append(a)
            cand = list(self.pos_sets.get(a, set()))
            if cand:
                # 양성 샘플 1~2개 균등하게 랜덤 선택
                n_samples = self.rng.choice([1, 1])
                if len(cand) >= n_samples:
                    selected = self.rng.sample(cand, k=n_samples)
                    cur.extend(selected)
                else:
                    cur.extend(cand)

            if len(cur) >= self.batch_size:
                yield self._finalize_batch(cur)
                cur = []
        if cur:
            yield self._finalize_batch(cur)

    def _finalize_batch(self, cur):
        uniq = list(dict.fromkeys(cur))
        self.rng.shuffle(uniq)
        if len(uniq) < self.batch_size:
            need = self.batch_size - len(uniq)
            fill = self.rng.sample(self.fill_pool, k=min(need, len(self.fill_pool)))
            uniq.extend(fill[:need])
        return uniq[: self.batch_size]


class NormalEmbedding(nn.Module):
    """
    단순 임베딩 테이블 (L2 정규화 포함)

    Input:
        idx: (B,) LongTensor (아이템 인덱스)
    Return:
        z: (B, dim) FloatTensor (L2-normalized)
    """

    def __init__(self, n_items, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor):
        z = self.emb(idx)
        return nn.functional.normalize(z, dim=1)


def build_pos_mask(batch_ids, pos_sets):
    """
    배치 내 양성 관계를 표시하는 (B,B) BoolTensor 생성.
    Input:
        batch_ids: 배치 트랙 ID 리스트
        pos_sets: 양성 관계 dict
    Return:
        mask: (B,B) torch.BoolTensor
    """
    B = len(batch_ids)
    mask = torch.zeros((B, B), dtype=torch.bool)
    idx_map = {tid: i for i, tid in enumerate(batch_ids)}

    for i, ti in enumerate(batch_ids):
        count = 0
        for nb in pos_sets.get(ti, ()):
            j = idx_map.get(nb)
            if j not in (None, i):
                mask[i, j] = True
                count += 1

    return mask


def uniformity_loss(z, t=2):
    """
    Uniformity loss: L_uni = log E[exp(-t * ||f(x) - f(y)||^2)]

    임베딩이 hypersphere 위에 균일하게 분포하도록 유도.

    Input:
        z: (B, D) L2-normalized embeddings
        t: temperature parameter (default=2)
    Return:
        uniformity: scalar loss
    """
    # Pairwise squared L2 distance: ||z_i - z_j||^2
    # For normalized vectors: ||z_i - z_j||^2 = 2 - 2 * <z_i, z_j>
    sq_dist = 2 - 2 * (z @ z.T)  # (B, B)

    # Exclude diagonal (self-pairs)
    mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    sq_dist = sq_dist[mask]

    # L_uni = log mean(exp(-t * sq_dist))
    return torch.logsumexp(-t * sq_dist, dim=0) - torch.log(
        torch.tensor(sq_dist.numel(), dtype=torch.float, device=z.device)
    )


def mp_infonce_loss(z, tau, pos_mask, uniformity_weight=0.0):
    """
    Multi-Positive InfoNCE 손실 계산 + Uniformity Loss.

    Input:
        z: (B, D) 임베딩
        tau: temperature
        pos_mask: (B, B) 양성 마스크
        uniformity_weight: uniformity loss의 가중치 (default=0.0)
    Return:
        loss: 스칼라 손실
        used: 양성이 존재한 anchor 개수
        avg_pos_sim: 평균 positive similarity
        avg_neg_sim: 평균 negative similarity
        uni_loss_val: uniformity loss 값
    """
    # temperature 적용 전 원본 similarity
    raw_sim = z @ z.T
    sim = raw_sim / tau
    sim.fill_diagonal_(-float("inf"))
    pos_logits = sim.masked_fill(~pos_mask, -float("inf"))
    denom_logits = sim
    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return sim.new_tensor(0.0), 0, 0.0, 0.0, 0.0
    pos_lse = torch.logsumexp(pos_logits[has_pos], dim=1)
    all_lse = torch.logsumexp(denom_logits[has_pos], dim=1)
    infonce_loss = (all_lse - pos_lse).mean()

    # Uniformity loss 추가
    uni_loss_val = 0.0
    if uniformity_weight > 0:
        uni_loss = uniformity_loss(z, t=2)
        uni_loss_val = uni_loss.item()
        total_loss = infonce_loss + uniformity_weight * uni_loss
    else:
        total_loss = infonce_loss

    # positive/negative similarity 계산 (temperature 적용 전 값 사용)
    raw_sim_no_diag = raw_sim.clone()
    raw_sim_no_diag.fill_diagonal_(0)

    # positive similarity: pos_mask가 True인 위치의 평균
    if pos_mask.any():
        avg_pos_sim = raw_sim_no_diag[pos_mask].mean().item()
    else:
        avg_pos_sim = 0.0

    # negative similarity: pos_mask가 False이고 대각선이 아닌 위치의 평균
    neg_mask = ~pos_mask
    neg_mask.fill_diagonal_(False)
    if neg_mask.any():
        avg_neg_sim = raw_sim_no_diag[neg_mask].mean().item()
    else:
        avg_neg_sim = 0.0

    return total_loss, int(has_pos.sum().item()), avg_pos_sim, avg_neg_sim, uni_loss_val
