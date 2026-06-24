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


def test_neighbor_ranking_excludes_self_and_orders_by_similarity():
    # v0 and v1 nearly identical; v2, v3 are the other basis directions.
    v = _norm(np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.14, 0.0],   # closest to v0
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32))
    rank = oracle.neighbor_ranking(v, anchor_pos=np.array([0, 2]), width=2, block=1)
    assert rank.shape == (2, 2)
    assert 0 not in rank[0].tolist()          # anchor excluded from its own ranking
    assert rank[0, 0] == 1                     # nearest neighbor of v0 is v1
    assert 2 not in rank[1].tolist()           # self excluded for the second anchor


def test_neighbor_ranking_caps_width_and_is_blocked_consistent():
    rng = np.random.default_rng(0)
    v = _norm(rng.standard_normal((20, 5)).astype(np.float32))
    pos = np.array([3, 7, 11, 15])
    r_one = oracle.neighbor_ranking(v, pos, width=100, block=1)   # width > N-1 -> capped
    r_big = oracle.neighbor_ranking(v, pos, width=100, block=4)   # block size must not matter
    assert r_one.shape == (4, 19)                                  # 20 corpus - 1 self
    assert np.array_equal(r_one, r_big)
