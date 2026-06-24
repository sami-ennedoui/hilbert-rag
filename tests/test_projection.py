import numpy as np
import torch

from hilbert_rag import projection


def test_projection_head_shape_and_unit_norm():
    torch.manual_seed(0)
    head = projection.ProjectionHead(in_dim=384, hidden=128, out_dim=8)
    y = head(torch.randn(5, 384))
    assert y.shape == (5, 8)
    assert np.allclose(y.norm(dim=1).detach().numpy(), 1.0, atol=1e-5)


def test_pca_projector_shape_and_norm():
    vecs = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float32)
    proj = projection.pca_projector(vecs, d_low=4, seed=1234)
    keys = proj(vecs[:10])
    assert keys.shape == (10, 4)
    assert np.allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-5)


def test_random_projector_deterministic_and_normed():
    v = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
    k1 = projection.random_projector(16, 4, seed=1234)(v)
    k2 = projection.random_projector(16, 4, seed=1234)(v)
    assert k1.shape == (5, 4)
    assert np.allclose(k1, k2)                                   # deterministic given seed
    assert np.allclose(np.linalg.norm(k1, axis=1), 1.0, atol=1e-5)
