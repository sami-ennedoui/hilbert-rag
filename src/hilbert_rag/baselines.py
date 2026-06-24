"""FAISS baselines: Flat (exact) and HNSW (approximate).

Both expose the same build/search interface as the cascade, so the harness runs
them through the same metrics. Vectors are L2-normalized, so inner product is cosine.
FaissFlat is the exact recall ceiling and a latency reference; FaissHNSW is the real
approximate competitor that the SFC index is measured against.
"""

from __future__ import annotations

import faiss
import numpy as np


class FaissFlat:
    def build(self, vecs: np.ndarray) -> "FaissFlat":
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        return self

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions (Q,k), sims (Q,k))."""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        sims, pos = self.index.search(q, k)
        return pos, sims


class FaissHNSW:
    def __init__(self, M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build(self, vecs: np.ndarray) -> "FaissHNSW":
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.index = faiss.IndexHNSWFlat(vecs.shape[1], self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(vecs)
        self.index.hnsw.efSearch = self.ef_search
        return self

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions (Q,k), sims (Q,k)). Approximate."""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        sims, pos = self.index.search(q, k)
        return pos, sims
