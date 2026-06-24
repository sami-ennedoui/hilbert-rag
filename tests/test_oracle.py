import numpy as np

from hilbert_rag import oracle


def _norm(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_exact_topk_known_answer():
    corpus = _norm(np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32))
    q = _norm(np.array([[1, 0.1]], dtype=np.float32))
    idx, sims = oracle.exact_topk(q, corpus, k=2)
    assert list(idx[0]) == [0, 2]            # nearest is e_x, then the diagonal
    assert sims[0, 0] >= sims[0, 1]


def test_holdout_split_disjoint_deterministic_sorted():
    qi, ii = oracle.holdout_split(100, 10, seed=1234)
    assert len(qi) == 10 and len(ii) == 90
    assert set(qi).isdisjoint(set(ii))
    assert list(qi) == sorted(qi) and list(ii) == sorted(ii)
    assert set(qi) | set(ii) == set(range(100))
    qi2, _ = oracle.holdout_split(100, 10, seed=1234)
    assert list(qi) == list(qi2)           # deterministic given the seed


def test_exact_topk_masked_excludes_filtered():
    corpus = _norm(np.array([[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32))
    q = _norm(np.array([[1, 0.1]], dtype=np.float32))
    mask = np.array([False, True, True, True])   # drop the true nearest (idx 0)
    idx, sims = oracle.exact_topk_masked(q, corpus, mask, k=1)
    assert idx[0, 0] == 2                     # best remaining is the diagonal
