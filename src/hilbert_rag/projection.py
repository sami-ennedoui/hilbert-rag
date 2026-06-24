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


def mine_triplets(
    ranking: np.ndarray,
    n_pos: int,
    neg_lo: int,
    neg_hi: int,
    n_neg: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (anchor_row, positive, negative) triplets from an exact-NN ranking.

    `ranking` is (A, R): row a holds anchor a's neighbor positions sorted by descending
    similarity, excluding the anchor itself. Positives are the n_pos closest neighbors;
    hard negatives are sampled from ranks [neg_lo, neg_hi), the band that is near but
    not a true neighbor. Mining hard (not random) negatives is what lets the learned
    projection beat PCA, so this band is deliberate (spec §5.4, rule 2).

    Returns three equal-length arrays: anchor row indices, positive positions, negative
    positions. The trainer maps an anchor row to its vector via its own anchor list.
    """
    ranking = np.asarray(ranking)
    n_anchors, width = ranking.shape
    neg_hi = min(neg_hi, width)
    rng = np.random.default_rng(seed)
    anchors, positives, negatives = [], [], []
    for a in range(n_anchors):
        pool = ranking[a, neg_lo:neg_hi]
        if pool.size == 0:
            continue
        for p in ranking[a, :n_pos]:
            chosen = rng.choice(pool, size=n_neg, replace=pool.size < n_neg)
            for neg in chosen:
                anchors.append(a)
                positives.append(int(p))
                negatives.append(int(neg))
    return np.asarray(anchors), np.asarray(positives), np.asarray(negatives)


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
