"""Learned projection head plus the PCA and random-projection baselines.

The three together are the ablation (spec §5.4): each produces a low-dimensional key
that feeds the same SFC index, and we compare them by downstream coarse and
end-to-end recall. The learned head optimizes a ranking loss over neighbors; PCA only
maximizes variance; random projection is the floor.

This file holds the architecture and the two baselines (data-independent, unit-tested
here). Hard-negative mining and the training loop are added once the embeddings and
exact-NN ranking are cached, since they need real neighbor data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

Projector = Callable[[np.ndarray], np.ndarray]


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, eps)).astype(np.float32)


class ProjectionHead(nn.Module):
    """MLP 384 -> hidden -> d_low with ReLU, output L2-normalized."""

    def __init__(self, in_dim: int = 384, hidden: int = 128, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def pca_projector(train_vecs: np.ndarray, d_low: int, seed: int) -> Projector:
    """Fit PCA on train_vecs; return a transform that projects to d_low and L2-normalizes."""
    pca = PCA(n_components=d_low, random_state=seed)
    pca.fit(np.asarray(train_vecs, dtype=np.float32))

    def transform(vecs: np.ndarray) -> np.ndarray:
        return _l2norm(pca.transform(np.asarray(vecs, dtype=np.float32)))

    return transform


def random_projector(in_dim: int, d_low: int, seed: int) -> Projector:
    """A fixed random Gaussian projection to d_low, L2-normalized. Deterministic by seed."""
    rng = np.random.default_rng(seed)
    matrix = (rng.standard_normal((in_dim, d_low)) / np.sqrt(in_dim)).astype(np.float32)

    def transform(vecs: np.ndarray) -> np.ndarray:
        return _l2norm(np.asarray(vecs, dtype=np.float32) @ matrix)

    return transform
