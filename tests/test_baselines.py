import numpy as np

from hilbert_rag import baselines


def _norm(a):
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def test_faiss_flat_is_exact():
    corpus = _norm(np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32))
    idx = baselines.FaissFlat().build(corpus)
    pos, sims = idx.search(corpus[:3], k=5)
    assert pos.shape == (3, 5) and sims.shape == (3, 5)
    assert list(pos[:, 0]) == [0, 1, 2]              # each query's nearest is itself
    assert np.allclose(sims[:, 0], 1.0, atol=1e-4)   # cosine with itself is 1


def test_faiss_hnsw_runs_and_mostly_finds_self():
    corpus = _norm(np.random.default_rng(1).standard_normal((200, 8)).astype(np.float32))
    idx = baselines.FaissHNSW().build(corpus)
    pos, sims = idx.search(corpus[:10], k=10)
    assert pos.shape == (10, 10)
    assert (pos[:, 0] == np.arange(10)).mean() >= 0.8   # approximate, but self is easy
