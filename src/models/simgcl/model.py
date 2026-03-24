"""
SimGCL: Simple Graph Contrastive Learning

LightGCN propagation + noise-based augmentation + contrastive loss + BPR loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimGCL(nn.Module):
    """
    SimGCL model for playlist-track recommendation

    Args:
        num_playlists: number of playlist nodes
        num_tracks: number of track nodes
        embedding_dim: dimension of embeddings
        n_layers: number of GCN propagation layers
        noise_eps: noise scale for perturbation (default: 0.1)
        normalize_embeddings: whether to normalize embeddings to L2 norm=1 (default: False)
    """

    def __init__(self, num_playlists, num_tracks, embedding_dim, n_layers=3, noise_eps=0.1, normalize_embeddings=False):
        super().__init__()

        self.num_playlists = num_playlists
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.noise_eps = noise_eps
        self.normalize_embeddings = normalize_embeddings

        # Learnable embeddings
        self.playlist_embedding = nn.Embedding(num_playlists, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)

        # Initialize with Xavier
        nn.init.xavier_uniform_(self.playlist_embedding.weight)
        nn.init.xavier_uniform_(self.track_embedding.weight)

        self.Graph = None  # Will be set by setup_graph()

    def setup_graph(self, norm_adj):
        """Set up normalized adjacency matrix (same as LightGCN)."""
        self.Graph = norm_adj.coalesce()

    def propagate(self, embeddings, add_noise=False):
        """
        LightGCN propagation with optional noise augmentation (using sparse matrix multiplication)

        Args:
            embeddings: [num_nodes, dim] node embeddings
            add_noise: whether to add uniform noise for augmentation

        Returns:
            final_embeddings: [num_nodes, dim] aggregated embeddings
        """
        all_embeddings = [embeddings]

        for layer in range(self.n_layers):
            # Sparse matrix multiplication (same as LightGCN)
            embeddings = torch.sparse.mm(self.Graph, embeddings)

            # Add noise for augmentation (SimGCL)
            if add_noise and self.training:
                noise = torch.rand_like(embeddings).to(embeddings.device)
                noise = noise * 2 * self.noise_eps - self.noise_eps  # Uniform[-eps, eps]
                embeddings = embeddings + noise

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

        # Propagate without noise
        final_embeddings = self.propagate(all_embeddings, add_noise=False)

        playlist_emb = final_embeddings[:self.num_playlists]
        track_emb = final_embeddings[self.num_playlists:]

        # Original code (no normalization):
        # return playlist_emb, track_emb

        # New code (optional L2 normalization to unit norm):
        if self.normalize_embeddings:
            playlist_emb = F.normalize(playlist_emb, p=2, dim=1)
            track_emb = F.normalize(track_emb, p=2, dim=1)

        return playlist_emb, track_emb

    def get_augmented_views(self):
        """
        Generate two augmented views for contrastive learning

        Returns:
            view1: [num_playlists + num_tracks, dim]
            view2: [num_playlists + num_tracks, dim]
        """
        all_embeddings = torch.cat([
            self.playlist_embedding.weight,
            self.track_embedding.weight
        ], dim=0)

        view1 = self.propagate(all_embeddings, add_noise=True)
        view2 = self.propagate(all_embeddings, add_noise=True)

        # Original code (no normalization):
        # return view1, view2

        # New code (optional L2 normalization to unit norm):
        if self.normalize_embeddings:
            view1 = F.normalize(view1, p=2, dim=1)
            view2 = F.normalize(view2, p=2, dim=1)

        return view1, view2


def bpr_loss(pos_scores, neg_scores, weights=None):
    """
    Bayesian Personalized Ranking loss (optionally weighted)

    Args:
        pos_scores: [batch_size] scores for positive items
        neg_scores: [batch_size] scores for negative items
        weights: [batch_size] optional weights for each sample (default: None)

    Returns:
        loss: scalar BPR loss
    """
    if weights is None:
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    else:
        loss = (-F.logsigmoid(pos_scores - neg_scores) * weights).mean()
    return loss


def infonce_loss(view1, view2, tau=0.2, random_neg_emb=None):
    """
    InfoNCE contrastive loss between two views with optional random negatives

    Args:
        view1: [batch_size, dim]
        view2: [batch_size, dim]
        tau: temperature parameter
        random_neg_emb: [num_random_neg, dim] optional random negative embeddings

    Returns:
        loss: scalar contrastive loss
    """
    # Normalize
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)

    batch_size = view1.size(0)

    if random_neg_emb is not None:
        # Add random negatives to view2
        random_neg_emb = F.normalize(random_neg_emb, dim=1)
        view2_extended = torch.cat([view2, random_neg_emb], dim=0)

        # Similarity matrix: [batch_size, batch_size + num_random_neg]
        sim_matrix = torch.matmul(view1, view2_extended.t()) / tau
    else:
        # Similarity matrix: [batch_size, batch_size]
        sim_matrix = torch.matmul(view1, view2.t()) / tau

    # Positive pairs are on the diagonal (first batch_size columns)
    labels = torch.arange(batch_size, device=view1.device)

    # InfoNCE: cross-entropy with diagonal as positive
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


