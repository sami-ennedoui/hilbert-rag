"""Embed the cached corpus and write data/embeddings.npy + data/ids.npy.

One-time CPU embedding pass over corpus.parquet. Pass HF_HUB_OFFLINE=1 so the
already-cached model loads without a hub round-trip.

Run: HF_HUB_OFFLINE=1 .venv/bin/python scripts/embed_corpus.py
"""

from __future__ import annotations

import time

import pandas as pd

from hilbert_rag import config, embeddings


def main() -> None:
    df = pd.read_parquet(config.DATA_DIR / "corpus.parquet")
    print(f"embedding {len(df)} chunks with {config.MODEL_ID} ...", flush=True)
    t0 = time.time()
    vecs = embeddings.cache_embeddings(df, config.DATA_DIR)
    print(f"wrote embeddings.npy {vecs.shape} + ids.npy in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
