"""Phase 2: unfiltered recall and latency, SFC cascade vs FAISS Flat and HNSW.

The honest head-to-head on plain (unfiltered) retrieval. FAISS Flat is the exact
recall ceiling (and a latency reference); HNSW is the real approximate competitor.
The SFC cascade is swept over candidate budgets (window), HNSW over efSearch, so each
is a recall/latency curve, not a single point. Every method is scored against the same
exact-NN oracle, on the same 300 held-out queries, in full-corpus position space.

Latency is measured single-query, single-thread (faiss threads pinned to 1) so the
numbers are comparable. Expectation, stated up front: HNSW dominates the SFC index on
unfiltered recall-per-millisecond. This script quantifies that gap honestly; the SFC
index's case is made on filtered retrieval (Phase 3), not here.

Outputs: results/recall_unfiltered.csv
Run:  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1 \
      .venv/bin/python scripts/run_recall_unfiltered.py
"""

from __future__ import annotations

import csv
import time

import faiss
import numpy as np
import torch

from hilbert_rag import benchmark, config, projection
from hilbert_rag.baselines import FaissFlat, FaissHNSW
from hilbert_rag.cascade import CascadeRetriever
from hilbert_rag.sfc_index import SFCIndex

SFC_WINDOWS = (100, 250, 500, 1000, 2000)
HNSW_EF = (16, 32, 64, 128, 256)
BITS = 10
K_RERANK = config.K_ORACLE          # search/rerank depth; >= max(K_VALUES)
N_WARMUP = 5


def _map_full(pos: np.ndarray, index_idx: np.ndarray) -> np.ndarray:
    """Map local index positions to full-corpus positions; keep -1 (FAISS miss) as -1."""
    pos = np.asarray(pos, dtype=np.int64)
    valid = pos >= 0
    return np.where(valid, index_idx[np.clip(pos, 0, index_idx.shape[0] - 1)], -1)


def _recall_rows(method, param, retrieved, oracle_full, lat, build_s, coarse=None, cand=None):
    rows = []
    for k in config.K_VALUES:
        rows.append(
            {
                "method": method,
                "param": param,
                "k": k,
                "recall_at_k": round(benchmark.recall_at_k(retrieved, oracle_full, k), 4),
                "coarse_recall_at_k": round(coarse[k], 4) if coarse else "",
                "mean_cand_size": cand if cand is not None else "",
                "lat_p50_ms": round(lat["p50"], 3),
                "lat_p95_ms": round(lat["p95"], 3),
                "build_s": round(build_s, 3),
            }
        )
    return rows


def main() -> None:
    faiss.omp_set_num_threads(1)
    emb = np.load(config.DATA_DIR / "embeddings.npy")
    split = np.load(config.DATA_DIR / "split.npz")
    query_idx, index_idx = split["query_idx"], split["index_idx"]
    oracle_full = np.load(config.DATA_DIR / "oracle_unfiltered.npy")
    index_vecs = emb[index_idx]
    query_vecs = emb[query_idx]
    nq = query_vecs.shape[0]
    print(f"index={index_vecs.shape[0]} queries={nq}")

    head = projection.load_head(config.DATA_DIR / "projection_head.pt")
    with torch.no_grad():
        index_keys = head(torch.from_numpy(np.ascontiguousarray(index_vecs))).numpy()
        query_keys = head(torch.from_numpy(np.ascontiguousarray(query_vecs))).numpy()

    rows: list[dict] = []

    # --- SFC cascade: one build, sweep the candidate window ---
    t = time.time()
    sfc = SFCIndex(bits=BITS).build(index_keys)
    sfc_build = time.time() - t
    retr = CascadeRetriever(sfc, index_vecs)
    for w in SFC_WINDOWS:
        for i in range(min(N_WARMUP, nq)):
            retr.search(query_keys[i], query_vecs[i], window=w, k=K_RERANK)
        retrieved = np.full((nq, K_RERANK), -1, dtype=np.int64)
        cand_sets, sizes, times = [], [], []
        for i in range(nq):
            t0 = time.perf_counter()
            res = retr.search(query_keys[i], query_vecs[i], window=w, k=K_RERANK)
            times.append((time.perf_counter() - t0) * 1e3)
            tk = _map_full(res["topk"], index_idx)
            retrieved[i, : tk.shape[0]] = tk
            cs = _map_full(res["candidates"], index_idx)
            cand_sets.append(cs)
            sizes.append(cs.shape[0])
        coarse = {k: benchmark.coarse_recall_at_k(cand_sets, oracle_full, k) for k in config.K_VALUES}
        lat = benchmark.summarize_latency(times)
        rows += _recall_rows("sfc", w, retrieved, oracle_full, lat, sfc_build, coarse, int(np.mean(sizes)))
        print(f"sfc w={w:5d} cand~{int(np.mean(sizes)):5d} r@10={rows[-2]['recall_at_k']:.3f} p50={lat['p50']:.2f}ms")

    # --- FAISS Flat: exact ceiling + latency reference ---
    t = time.time()
    flat = FaissFlat().build(index_vecs)
    flat_build = time.time() - t
    for i in range(min(N_WARMUP, nq)):
        flat.search(query_vecs[i : i + 1], K_RERANK)
    times = []
    pos_all = np.full((nq, K_RERANK), -1, dtype=np.int64)
    for i in range(nq):
        t0 = time.perf_counter()
        pos, _ = flat.search(query_vecs[i : i + 1], K_RERANK)
        times.append((time.perf_counter() - t0) * 1e3)
        pos_all[i] = pos[0]
    retrieved = _map_full(pos_all, index_idx)
    lat = benchmark.summarize_latency(times)
    rows += _recall_rows("faiss_flat", "", retrieved, oracle_full, lat, flat_build)
    print(f"faiss_flat r@10={rows[-2]['recall_at_k']:.3f} p50={lat['p50']:.2f}ms (recall@k must be ~1.0)")

    # --- FAISS HNSW: one build, sweep efSearch ---
    t = time.time()
    hnsw = FaissHNSW(M=32, ef_construction=200).build(index_vecs)
    hnsw_build = time.time() - t
    for ef in HNSW_EF:
        hnsw.index.hnsw.efSearch = ef
        for i in range(min(N_WARMUP, nq)):
            hnsw.search(query_vecs[i : i + 1], K_RERANK)
        times = []
        pos_all = np.full((nq, K_RERANK), -1, dtype=np.int64)
        for i in range(nq):
            t0 = time.perf_counter()
            pos, _ = hnsw.search(query_vecs[i : i + 1], K_RERANK)
            times.append((time.perf_counter() - t0) * 1e3)
            pos_all[i] = pos[0]
        retrieved = _map_full(pos_all, index_idx)
        lat = benchmark.summarize_latency(times)
        rows += _recall_rows("faiss_hnsw", ef, retrieved, oracle_full, lat, hnsw_build)
        print(f"hnsw ef={ef:4d} r@10={rows[-2]['recall_at_k']:.3f} p50={lat['p50']:.2f}ms")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "recall_unfiltered.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
