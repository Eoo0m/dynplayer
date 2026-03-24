"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

LightGCN propagation + BPR loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN model for playlist-track recommendation

    Args:
        num_playlists: number of playlist nodes
        num_tracks: number of track nodes
        embedding_dim: dimension of embeddings
        n_layers: number of GCN propagation layers
    """

    def __init__(self, num_playlists, num_tracks, embedding_dim, n_layers=3, normalize_embeddings=False):
        super().__init__()

        self.num_playlists = num_playlists
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.normalize_embeddings = normalize_embeddings

        # Learnable embeddings
        self.playlist_embedding = nn.Embedding(num_playlists, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)

        # Initialize with Xavier
        nn.init.xavier_uniform_(self.playlist_embedding.weight)
        nn.init.xavier_uniform_(self.track_embedding.weight)

        self.Graph = None  # Will be set by setup_graph()

    def setup_graph(self, norm_adj):
        """Set up normalized adjacency matrix"""
        self.Graph = norm_adj.coalesce()

    def propagate(self, embeddings):
        """LightGCN propagation (using sparse matrix multiplication)"""
        all_embeddings = [embeddings]

        for layer in range(self.n_layers):
            # Sparse matrix multiplication
            embeddings = torch.sparse.mm(self.Graph, embeddings)
            all_embeddings.append(embeddings)

        # LightGCN: average all layers
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final_embeddings

    def forward(self):
        """
        Forward pass: returns clean embeddings

        Returns:
            playlist_emb: [num_playlists, dim]
            track_emb: [num_tracks, dim]
        """
        # Concatenate playlist and track embeddings
        all_embeddings = torch.cat([
            self.playlist_embedding.weight,
            self.track_embedding.weight
        ], dim=0)

        # Propagate
        final_embeddings = self.propagate(all_embeddings)

        playlist_emb = final_embeddings[:self.num_playlists]
        track_emb = final_embeddings[self.num_playlists:]

        # Optional L2 normalization
        if self.normalize_embeddings:
            playlist_emb = F.normalize(playlist_emb, p=2, dim=1)
            track_emb = F.normalize(track_emb, p=2, dim=1)

        return playlist_emb, track_emb

    def get_embedding(self, playlists, pos_tracks, neg_tracks):
        """
        Get embeddings for batch (for regularization)

        Returns:
            all_playlists: propagated playlist embeddings
            all_tracks: propagated track embeddings
            playlists_ego: ego playlist embeddings (before propagation)
            pos_tracks_ego: ego positive track embeddings
            neg_tracks_ego: ego negative track embeddings
        """
        all_playlists, all_tracks = self.forward()

        playlists_ego = self.playlist_embedding(playlists)
        pos_tracks_ego = self.track_embedding(pos_tracks)
        neg_tracks_ego = self.track_embedding(neg_tracks)

        return (
            all_playlists[playlists],
            all_tracks[pos_tracks],
            all_tracks[neg_tracks],
            playlists_ego,
            pos_tracks_ego,
            neg_tracks_ego,
        )


def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking loss

    Args:
        pos_scores: [batch_size] scores for positive items
        neg_scores: [batch_size] scores for negative items

    Returns:
        loss: scalar BPR loss
    """
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
    return loss
