"""Build the held-out query set and the exact-NN unfiltered oracle.

Holds out N_QUERIES chunks as queries (deterministic), computes each query's exact
cosine top-K_ORACLE neighbors among the remaining index, and caches the split, the
oracle, and the query metadata. The oracle stores full-corpus row positions (into
ids/embeddings), all of which lie in the index subset by construction.

Run: .venv/bin/python scripts/build_oracle.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from hilbert_rag import config, embeddings, oracle


def main() -> None:
    ids, vecs = embeddings.load_embeddings(config.DATA_DIR)
    df = pd.read_parquet(config.DATA_DIR / "corpus.parquet")
    assert (df["chunk_id"].astype(str).to_numpy() == ids).all(), "corpus/embeddings id order mismatch"

    n = len(ids)
    query_idx, index_idx = oracle.holdout_split(n, config.N_QUERIES, config.SEED)
    index_mask = np.zeros(n, dtype=bool)
    index_mask[index_idx] = True

    t0 = time.time()
    oracle_idx, _ = oracle.exact_topk_masked(vecs[query_idx], vecs, index_mask, config.K_ORACLE)
    print(f"oracle {oracle_idx.shape} computed in {time.time() - t0:.1f}s", flush=True)

    np.savez(config.DATA_DIR / "split.npz", query_idx=query_idx, index_idx=index_idx)
    np.save(config.DATA_DIR / "oracle_unfiltered.npy", oracle_idx)

    qdf = df.iloc[query_idx][["chunk_id", "text", "primary_category", "year", "published"]].copy()
    qdf.insert(0, "corpus_pos", query_idx)
    qdf.to_parquet(config.DATA_DIR / "queries.parquet", index=False)

    print(
        f"queries={len(qdf)} index={len(index_idx)}; "
        f"wrote split.npz, oracle_unfiltered.npy, queries.parquet",
        flush=True,
    )


if __name__ == "__main__":
    main()
