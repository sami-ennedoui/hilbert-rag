import numpy as np
import torch

from hilbert_rag import projection


def test_projection_head_shape_and_unit_norm():
    torch.manual_seed(0)
    head = projection.ProjectionHead(in_dim=384, hidden=128, out_dim=8)
    y = head(torch.randn(5, 384))
    assert y.shape == (5, 8)
    assert np.allclose(y.norm(dim=1).detach().numpy(), 1.0, atol=1e-5)


def test_train_projection_loss_decreases_on_separable_data():
    rng = np.random.default_rng(0)
    d = 32

    def _n(x):
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    a = _n(rng.standard_normal((200, d)))
    p = _n(a + 0.01 * rng.standard_normal((200, d)))     # positives near their anchor
    neg = _n(rng.standard_normal((200, d)))              # negatives unrelated
    head, history = projection.train_projection(
        a, p, neg, d_low=8, in_dim=d, epochs=15, lr=1e-2, seed=0
    )
    assert history[-1] < history[0]                      # the ranking loss went down
    assert isinstance(head, projection.ProjectionHead)


def test_train_projection_infonce_decreases_and_separates():
    rng = np.random.default_rng(0)
    d = 32

    def _n(x):
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    a = _n(rng.standard_normal((256, d)))
    p = _n(a + 0.01 * rng.standard_normal((256, d)))     # each anchor's true neighbor
    head, history = projection.train_projection_infonce(
        a, p, d_low=8, in_dim=d, epochs=20, lr=1e-2, batch_size=64, seed=0
    )
    assert history[-1] < history[0]
    # after training, projected anchor is closer to its own positive than to a random other
    import torch
    with torch.no_grad():
        za = head(torch.from_numpy(a))
        zp = head(torch.from_numpy(p))
    own = (za * zp).sum(1).mean().item()
    other = (za * torch.roll(zp, 1, 0)).sum(1).mean().item()
    assert own > other


def test_pca_projector_shape_and_norm():
    vecs = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float32)
    proj = projection.pca_projector(vecs, d_low=4, seed=1234)
    keys = proj(vecs[:10])
    assert keys.shape == (10, 4)
    assert np.allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-5)


def test_mine_triplets_picks_right_ranks_and_shapes():
    # 2 anchors, 10 ranked neighbors each (values are arbitrary corpus positions).
    ranking = np.array([
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        [200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
    ])
    a, p, n = projection.mine_triplets(ranking, n_pos=1, neg_lo=3, neg_hi=8, n_neg=2, seed=0)
    assert len(a) == len(p) == len(n) == 2 * 1 * 2          # A * n_pos * n_neg
    assert set(p.tolist()) == {100, 200}                    # positive = closest neighbor
    assert set(a.tolist()) == {0, 1}                        # anchor row indices
    # negatives come only from ranks [3, 8)
    assert set(n.tolist()).issubset(
        {103, 104, 105, 106, 107, 203, 204, 205, 206, 207}
    )


def test_mine_triplets_negatives_exclude_positives():
    ranking = np.tile(np.arange(20), (3, 1))
    _, p, n = projection.mine_triplets(ranking, n_pos=2, neg_lo=5, neg_hi=15, n_neg=3, seed=1)
    assert set(p.tolist()).isdisjoint(set(n.tolist()))      # positives (ranks 0-1) never negatives


def test_save_and_load_head_roundtrip(tmp_path):
    import torch

    torch.manual_seed(0)
    head = projection.ProjectionHead(in_dim=32, hidden=16, out_dim=8)
    x = torch.randn(4, 32)
    before = head(x).detach().numpy()
    path = tmp_path / "head.pt"
    projection.save_head(head, path, in_dim=32, hidden=16, out_dim=8, meta={"note": "test"})
    loaded = projection.load_head(path)
    after = loaded(x).detach().numpy()
    assert np.allclose(before, after, atol=1e-6)


def test_random_projector_deterministic_and_normed():
    v = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
    k1 = projection.random_projector(16, 4, seed=1234)(v)
    k2 = projection.random_projector(16, 4, seed=1234)(v)
    assert k1.shape == (5, 4)
    assert np.allclose(k1, k2)                                   # deterministic given seed
    assert np.allclose(np.linalg.norm(k1, axis=1), 1.0, atol=1e-5)
