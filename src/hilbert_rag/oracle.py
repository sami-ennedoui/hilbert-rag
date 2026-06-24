"""Exact nearest-neighbor ground truth.

The recall oracle for the whole benchmark. Vectors are L2-normalized, so cosine
similarity is a plain dot product. This is embedding-space truth, not human-judged
relevance; the README says so. Everything else is scored against these results.
"""

from __future__ import annotations

import numpy as np


def holdout_split(n: int, n_queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split positions [0, n) into a sorted query set and the sorted searchable
    index (its complement). Deterministic given the seed.

    Queries are held out of the index so a query never trivially retrieves itself;
    its true neighbors are found among the index only.
    """
    rng = np.random.default_rng(seed)
    query_idx = np.sort(rng.choice(n, size=n_queries, replace=False))
    mask = np.ones(n, dtype=bool)
    mask[query_idx] = False
    index_idx = np.flatnonzero(mask)
    return query_idx, index_idx


def _topk_from_sims(sims: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k by descending similarity for each row of a (Q, N) sim matrix."""
    q, n = sims.shape
    k = min(k, n)
    # argpartition gives the k best (unordered), then we sort just those k.
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(q)[:, None]
    part_sims = sims[rows, part]
    order = np.argsort(-part_sims, axis=1)
    idx = part[rows, order]
    out_sims = part_sims[rows, order]
    return idx, out_sims


def exact_topk(query_vecs: np.ndarray, corpus_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k over the full corpus.

    Returns (idx, sims), each shape (Q, k); idx are global corpus indices.
    """
    sims = query_vecs.astype(np.float32) @ corpus_vecs.astype(np.float32).T
    return _topk_from_sims(sims, k)


def exact_topk_masked(
    query_vecs: np.ndarray, corpus_vecs: np.ndarray, mask: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k restricted to corpus items where mask is True.

    This is the oracle for filtered retrieval: the true neighbors within the
    metadata-filtered subset. Returns global corpus indices.
    """
    sims = query_vecs.astype(np.float32) @ corpus_vecs.astype(np.float32).T
    sims = np.where(np.asarray(mask, dtype=bool)[None, :], sims, -np.inf)
    return _topk_from_sims(sims, k)


def neighbor_ranking(
    corpus_vecs: np.ndarray, anchor_pos: np.ndarray, width: int, block: int = 512
) -> np.ndarray:
    """Exact-NN ranking of each anchor's `width` nearest neighbors within the corpus,
    excluding the anchor itself, sorted by descending cosine.

    This is the training signal for hard-negative mining (projection.mine_triplets):
    positives come from the top ranks, hard negatives from a deeper band. Computed in
    anchor blocks so the (block, N) similarity matrix never blows up memory.

    Returns (A, width') int64 of corpus positions, where width' = min(width, N - 1).
    """
    corpus_vecs = np.asarray(corpus_vecs, dtype=np.float32)
    anchor_pos = np.asarray(anchor_pos, dtype=np.int64)
    n = corpus_vecs.shape[0]
    width = min(width, n - 1)
    out = np.empty((anchor_pos.shape[0], width), dtype=np.int64)
    for start in range(0, anchor_pos.shape[0], block):
        blk = anchor_pos[start : start + block]
        sims = corpus_vecs[blk] @ corpus_vecs.T          # (b, N)
        sims[np.arange(blk.shape[0]), blk] = -np.inf       # drop self
        idx, _ = _topk_from_sims(sims, width)
        out[start : start + blk.shape[0]] = idx
    return out
