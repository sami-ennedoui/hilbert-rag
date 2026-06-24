import numpy as np

from hilbert_rag import cascade
from hilbert_rag.sfc_index import SFCIndex


def _norm(a):
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def test_exact_rerank_picks_best_candidates():
    corpus = _norm(np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32))
    q = _norm(np.array([[1, 0.1]], dtype=np.float32))[0]
    cand = np.array([1, 2, 3])                 # exclude the true best (idx 0)
    pos, sims = cascade.exact_rerank(q, cand, corpus, k=2)
    assert list(pos) == [2, 1]                 # diagonal then e_y
    assert sims[0] >= sims[1]


def test_exact_rerank_empty_candidates():
    corpus = _norm(np.array([[1, 0], [0, 1]], dtype=np.float32))
    pos, sims = cascade.exact_rerank(corpus[0], np.array([], dtype=int), corpus, k=3)
    assert len(pos) == 0 and len(sims) == 0


def test_cascade_search_reranks_within_window():
    rng = np.random.default_rng(0)
    pts = _norm(rng.standard_normal((30, 2)).astype(np.float32))
    retr = cascade.CascadeRetriever(SFCIndex(bits=6).build(pts), pts)
    out = retr.search(pts[7], pts[7], window=6, k=3)
    assert len(out["topk"]) == 3
    assert set(out["topk"].tolist()).issubset(set(out["candidates"].tolist()))
    assert out["topk"][0] == 7                 # identical query vector -> itself is top-1
