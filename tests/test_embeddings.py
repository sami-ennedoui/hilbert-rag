import numpy as np
import pandas as pd

from hilbert_rag import embeddings


def test_embeddings_shape_norm_and_cache_roundtrip(tmp_path):
    vecs = embeddings.embed_texts(["hello world", "space filling curves"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    df = pd.DataFrame({"chunk_id": ["a", "b"], "text": ["hello world", "space filling curves"]})
    embeddings.cache_embeddings(df, tmp_path)
    ids, loaded = embeddings.load_embeddings(tmp_path)
    assert list(ids) == ["a", "b"]
    assert np.allclose(loaded, vecs, atol=1e-5)
