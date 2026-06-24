"""Coarse-to-exact cascade.

Stage 1: the SFC index returns candidate positions from the low-dimensional key
(cheap, approximate). Stage 2: exact cosine rerank of those candidates in full
dimension returns the top-k. The candidate set is exposed alongside the reranked
result so the benchmark can report coarse recall (did the candidates contain the
true neighbors) separately from end-to-end recall@k.
"""

from __future__ import annotations

import numpy as np

from .sfc_index import SFCIndex


def exact_rerank(
    query_vec: np.ndarray, cand_pos: np.ndarray, corpus_vecs: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank candidate positions by exact cosine against the full-dimension query.

    Returns (positions, sims), each length min(k, #candidates), sorted by descending
    similarity. Vectors are L2-normalized, so cosine is a dot product.
    """
    cand_pos = np.asarray(cand_pos, dtype=np.int64)
    if cand_pos.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    sims = corpus_vecs[cand_pos].astype(np.float32) @ np.asarray(query_vec, dtype=np.float32)
    k = min(k, cand_pos.size)
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return cand_pos[top], sims[top]


class CascadeRetriever:
    def __init__(self, sfc: SFCIndex, corpus_vecs: np.ndarray):
        self.sfc = sfc
        self.corpus_vecs = np.asarray(corpus_vecs, dtype=np.float32)

    def search(self, query_low: np.ndarray, query_full: np.ndarray, window: int, k: int) -> dict:
        """Return {candidates, topk, scores}: the SFC candidate positions, the
        reranked top-k positions, and their similarities."""
        candidates = self.sfc.query(query_low, window)
        topk, scores = exact_rerank(query_full, candidates, self.corpus_vecs, k)
        return {"candidates": candidates, "topk": topk, "scores": scores}
