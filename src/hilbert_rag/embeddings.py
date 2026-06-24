"""Embed the corpus with a frozen sentence-transformer and cache to disk.

The model is loaded once and reused. Outputs are L2-normalized float32, so cosine
similarity downstream is a plain dot product. Embedding the full corpus is a
one-time cost; everything else loads the cached vectors.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config

_MODEL: SentenceTransformer | None = None


def _model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(config.MODEL_ID)
    return _MODEL


def embed_texts(texts: Iterable[str], batch_size: int = 256) -> np.ndarray:
    """Return an (N, 384) float32 array of L2-normalized embeddings."""
    vecs = _model().encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vecs, dtype=np.float32)


def cache_embeddings(df: pd.DataFrame, out_dir: str | Path) -> np.ndarray:
    """Embed df['text'] and write embeddings.npy + ids.npy (from df['chunk_id'])."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vecs = embed_texts(df["text"].tolist())
    np.save(out_dir / "embeddings.npy", vecs)
    # Build a fixed-width unicode array (dtype '<U...'), not an object array, so
    # the file is plain and loads without allow_pickle.
    ids = np.array(df["chunk_id"].astype(str).tolist())
    np.save(out_dir / "ids.npy", ids)
    return vecs


def load_embeddings(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (ids, vecs) from a directory written by cache_embeddings."""
    data_dir = Path(data_dir)
    vecs = np.load(data_dir / "embeddings.npy")
    ids = np.load(data_dir / "ids.npy")
    return ids, vecs
