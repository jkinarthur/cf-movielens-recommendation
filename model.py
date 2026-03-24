"""
model.py – Standalone NCF (Neural Collaborative Filtering) Model Definition
============================================================================
Mirrors the architecture used in Module 4 training so that app.py can import
this class and load saved weights without touching the training code.

Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
"""

import torch
import torch.nn as nn


class NeuralCollaborativeFiltering(nn.Module):
    """
    NCF model combining Generalised Matrix Factorisation (GMF)
    and Multi-Layer Perceptron (MLP) branches.

    Args:
        n_users       (int)  : Number of unique users in the training set.
        n_movies      (int)  : Number of unique movies in the training set.
        embedding_dim (int)  : Embedding size for the GMF branch (default 32).
        mlp_layers    (list) : Hidden layer widths for the MLP branch
                               (default [64, 32, 16]).  The first value is
                               also the concatenated input size, so each
                               embedding is half that width.
        dropout       (float): Dropout probability in MLP (default 0.2).
    """

    def __init__(
        self,
        n_users: int,
        n_movies: int,
        embedding_dim: int = 32,
        mlp_layers: list = None,
        dropout: float = 0.2,
    ):
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]
        super().__init__()

        # ── GMF branch ────────────────────────────────────────────────────────
        self.gmf_user_embedding  = nn.Embedding(n_users,  embedding_dim)
        self.gmf_movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # ── MLP branch ────────────────────────────────────────────────────────
        mlp_half = mlp_layers[0] // 2          # half the first layer width
        self.mlp_user_embedding  = nn.Embedding(n_users,  mlp_half)
        self.mlp_movie_embedding = nn.Embedding(n_movies, mlp_half)

        layers, in_dim = [], mlp_layers[0]
        for out_dim in mlp_layers[1:]:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        # ── Output layer ──────────────────────────────────────────────────────
        self.output_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for emb in (
            self.gmf_user_embedding,
            self.gmf_movie_embedding,
            self.mlp_user_embedding,
            self.mlp_movie_embedding,
        ):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids  : LongTensor of shape (B,)
            movie_ids : LongTensor of shape (B,)
        Returns:
            Tensor of shape (B,) – predicted ratings in [0, 1] (normalised).
        """
        # GMF: element-wise product of embeddings
        gmf_out = (
            self.gmf_user_embedding(user_ids) * self.gmf_movie_embedding(movie_ids)
        )

        # MLP: concatenate embeddings, feed through layers
        mlp_in  = torch.cat(
            [self.mlp_user_embedding(user_ids), self.mlp_movie_embedding(movie_ids)],
            dim=-1,
        )
        mlp_out = self.mlp(mlp_in)

        # Fuse and predict
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.sigmoid(self.output_layer(fused)).squeeze(-1)
