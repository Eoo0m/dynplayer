import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random


# ======================================================
# Dataset - Playlist Caption ↔ Playlist Embedding
# ======================================================
class PlaylistCaptionDataset(Dataset):
    def __init__(self, playlist_ids, title_emb_dict, playlist_embs, playlist_index):
        """
        플레이리스트 캡션 ↔ 플레이리스트 임베딩 대조학습
        """
        self.playlist_ids = [
            pid
            for pid in playlist_ids
            if pid in title_emb_dict and pid in playlist_index
        ]
        self.title_emb_dict = title_emb_dict
        self.playlist_embs = playlist_embs
        self.playlist_index = playlist_index

    def __len__(self):
        return len(self.playlist_ids)

    def __getitem__(self, idx):
        playlist_id = self.playlist_ids[idx]

        # 플레이리스트 타이틀 임베딩 (GPT)
        title_emb = self.title_emb_dict[playlist_id]

        # 플레이리스트 임베딩 (LightGCN)
        playlist_emb = self.playlist_embs[self.playlist_index[playlist_id]]

        return (
            torch.tensor(title_emb, dtype=torch.float32),
            torch.tensor(playlist_emb, dtype=torch.float32),
        )


# ======================================================
# CLIP Model (Caption ↔ Playlist)
# ======================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()

        # First projection to hidden dimension
        self.proj_in = nn.Linear(in_dim, hidden_dim)

        # Residual blocks
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        # Final projection to output dimension
        self.proj_out = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Project to hidden dimension
        h = self.proj_in(x)
        h = self.activation(h)
        h = self.norm(h)

        # Residual block 1
        residual = h
        h = self.block1(h)
        h = h + residual

        # Residual block 2
        residual = h
        h = self.block2(h)
        h = h + residual

        # Project to output dimension
        h = self.proj_out(h)

        # L2 normalize
        return F.normalize(h, dim=-1)


# class ProjectionMLP(nn.Module):
#     def __init__(self, in_dim, out_dim=1024, hidden_dim=2048, heads=8):
#         super().__init__()

#         self.heads = heads
#         self.projs = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Linear(in_dim, hidden_dim),
#                     nn.GELU(),
#                     nn.LayerNorm(hidden_dim),
#                     nn.Dropout(0.1),
#                     nn.Linear(hidden_dim, out_dim),
#                 )
#                 for _ in range(heads)
#             ]
#         )

#     def forward(self, x):
#         # 각 head별 projection
#         outs = []
#         for proj in self.projs:
#             h = proj(x)  # (B, out_dim)
#             h = F.normalize(h, dim=-1)
#             outs.append(h)

#         # (heads, B, out_dim) → (B, out_dim)
#         h_final = torch.stack(outs, dim=0).mean(dim=0)
#         return F.normalize(h_final, dim=-1)


class CaptionPlaylistCLIP(nn.Module):
    def __init__(self, caption_dim, playlist_dim, out_dim=1024, temperature=0.07):
        super().__init__()
        self.caption_proj = ProjectionMLP(caption_dim, out_dim)
        self.playlist_proj = ProjectionMLP(playlist_dim, out_dim)
        self.temperature = temperature

    def forward(self, caption, playlist):
        """
        caption: (B, caption_dim) - GPT 텍스트 임베딩 (플레이리스트 타이틀)
        playlist: (B, playlist_dim) - LightGCN 플레이리스트 임베딩
        """
        # 프로젝션
        z_caption = self.caption_proj(caption)  # (B, out_dim)
        z_playlist = self.playlist_proj(playlist)  # (B, out_dim)

        # Similarity matrix (원본 코사인 유사도)
        sim_raw = torch.matmul(z_caption, z_playlist.T)  # (B, B)

        # Temperature 적용 (loss 계산용)
        sim_matrix = sim_raw / self.temperature

        # Positive는 대각선
        labels = torch.arange(len(z_caption), device=z_caption.device)

        # Cross-entropy loss (양방향)
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss1 + loss2) / 2

        # 모니터링용 (원본 코사인 유사도 사용)
        pos_sim = torch.diagonal(sim_raw).mean()
        neg_sim = (sim_raw.sum() - torch.diagonal(sim_raw).sum()) / (
            sim_raw.numel() - len(z_caption)
        )

        return loss, pos_sim.item(), neg_sim.item()
