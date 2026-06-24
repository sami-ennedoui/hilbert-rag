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
from pathlib import Path

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


def train_projection(
    a_vecs: np.ndarray,
    p_vecs: np.ndarray,
    n_vecs: np.ndarray,
    d_low: int,
    in_dim: int = 384,
    hidden: int = 128,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    margin: float = 0.2,
    seed: int = 1234,
) -> tuple[ProjectionHead, list[float]]:
    """Train the projection head with a triplet margin loss on (anchor, positive,
    negative) vectors. The loss pushes the projected anchor closer to its positive than
    to its hard negative: max(0, margin - <za,zp> + <za,zn>), keys L2-normalized so the
    dot product is cosine. Returns the trained head and the per-epoch mean loss.
    """
    torch.manual_seed(seed)
    head = ProjectionHead(in_dim, hidden, d_low)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    a = torch.from_numpy(np.ascontiguousarray(a_vecs, dtype=np.float32))
    p = torch.from_numpy(np.ascontiguousarray(p_vecs, dtype=np.float32))
    n = torch.from_numpy(np.ascontiguousarray(n_vecs, dtype=np.float32))
    n_triplets = a.shape[0]
    rng = np.random.default_rng(seed)

    history: list[float] = []
    for _ in range(epochs):
        perm = rng.permutation(n_triplets)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n_triplets, batch_size):
            idx = perm[start : start + batch_size]
            za, zp, zn = head(a[idx]), head(p[idx]), head(n[idx])
            pos_sim = (za * zp).sum(dim=1)
            neg_sim = (za * zn).sum(dim=1)
            loss = torch.relu(margin - pos_sim + neg_sim).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(batches, 1))
    return head, history


def train_projection_infonce(
    a_vecs: np.ndarray,
    p_vecs: np.ndarray,
    d_low: int,
    in_dim: int = 384,
    hidden: int = 128,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    temperature: float = 0.1,
    seed: int = 1234,
) -> tuple[ProjectionHead, list[float]]:
    """Train the projection head with an InfoNCE loss over (anchor, positive) pairs,
    using the other positives in the batch as negatives.

    Unlike the pairwise triplet loss, InfoNCE makes each anchor discriminate its true
    neighbor against many negatives at once, a far stronger signal for preserving the
    global cosine geometry the oracle uses (spec §5.4 permits triplet or InfoNCE).
    Each row of a_vecs is an anchor and the same row of p_vecs is one of its true
    nearest neighbors. Returns the trained head and per-epoch mean loss.
    """
    torch.manual_seed(seed)
    head = ProjectionHead(in_dim, hidden, d_low)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    a = torch.from_numpy(np.ascontiguousarray(a_vecs, dtype=np.float32))
    p = torch.from_numpy(np.ascontiguousarray(p_vecs, dtype=np.float32))
    n_pairs = a.shape[0]
    rng = np.random.default_rng(seed)

    history: list[float] = []
    for _ in range(epochs):
        perm = rng.permutation(n_pairs)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start : start + batch_size]
            if idx.size < 2:  # InfoNCE needs at least one negative in the batch
                continue
            za, zp = head(a[idx]), head(p[idx])
            logits = za @ zp.T / temperature           # (b, b); row i positive is column i
            labels = torch.arange(za.shape[0])
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(batches, 1))
    return head, history


def save_head(
    head: ProjectionHead, path, in_dim: int, hidden: int, out_dim: int, meta: dict | None = None
) -> None:
    """Persist a trained head: weights plus the architecture dims needed to rebuild it,
    plus optional metadata (the training config) for provenance."""
    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_dim": int(in_dim),
            "hidden": int(hidden),
            "out_dim": int(out_dim),
            "meta": meta or {},
        },
        str(path),
    )


def load_head(path) -> ProjectionHead:
    """Rebuild a head from a checkpoint written by save_head; returns it in eval mode."""
    ckpt = torch.load(str(Path(path)), map_location="cpu", weights_only=False)
    head = ProjectionHead(ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head


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
