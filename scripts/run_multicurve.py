"""Salvage experiment: multiple shifted Hilbert curves vs a single curve.

The single-curve result loses most neighbor structure because the curve splits some
spatial neighbors across a recursion boundary. The standard fix is several curves with
different shifts whose windows are unioned (Leutenegger & Mokbel; HD-Index). This script
measures how much of the gap that recovers, at a FIXED TOTAL candidate budget so the
comparison is fair: C curves each scan a window of total/(2*C), then the union is reranked.

It sweeps C in {1,2,4,8} at two total budgets and reports coarse recall (the candidate-set
ceiling) and end-to-end recall@k, the mean unique candidate count actually scanned, and
single-query latency. C=1 reproduces the plain single-curve number as a sanity check. One
FAISS HNSW row is measured in the same run as the reference the curve is trying to reach;
the honest expectation is that multi-curve narrows the gap but does not close it.

Outputs: results/multicurve.csv
Run:  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
      .venv/bin/python scripts/run_multicurve.py
"""

from __future__ import annotations

import csv
import time

import faiss
import numpy as np
import torch

from hilbert_rag import benchmark, config, projection
from hilbert_rag.baselines import FaissHNSW
from hilbert_rag.cascade import CascadeRetriever
from hilbert_rag.sfc_index import MultiCurveSFCIndex

CURVES = (1, 2, 4, 8)
BUDGETS = (1000, 10000)     # total candidate budget, split as window = budget // (2*C)
HNSW_EF = 64               # the reference operating point from Phase 2
BITS = 10
K_RERANK = config.K_ORACLE
N_WARMUP = 5


def _map_full(pos: np.ndarray, index_idx: np.ndarray) -> np.ndarray:
    """Map local index positions to full-corpus positions; keep -1 as -1."""
    pos = np.asarray(pos, dtype=np.int64)
    valid = pos >= 0
    return np.where(valid, index_idx[np.clip(pos, 0, index_idx.shape[0] - 1)], -1)


def _rows(method, n_curves, budget, retrieved, oracle_full, lat, build_s, coarse=None, cand=None):
    out = []
    for k in config.K_VALUES:
        out.append(
            {
                "method": method,
                "n_curves": n_curves,
                "total_budget": budget,
                "k": k,
                "recall_at_k": round(benchmark.recall_at_k(retrieved, oracle_full, k), 4),
                "coarse_recall_at_k": round(coarse[k], 4) if coarse else "",
                "mean_cand_size": cand if cand is not None else "",
                "lat_p50_ms": round(lat["p50"], 3),
                "lat_p95_ms": round(lat["p95"], 3),
                "build_s": round(build_s, 3),
            }
        )
    return out


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

    for c in CURVES:
        t = time.time()
        sfc = MultiCurveSFCIndex(bits=BITS, n_curves=c).build(index_keys)
        build_s = time.time() - t
        retr = CascadeRetriever(sfc, index_vecs)
        for budget in BUDGETS:
            window = max(1, budget // (2 * c))
            for i in range(min(N_WARMUP, nq)):
                retr.search(query_keys[i], query_vecs[i], window=window, k=K_RERANK)
            retrieved = np.full((nq, K_RERANK), -1, dtype=np.int64)
            cand_sets, sizes, times = [], [], []
            for i in range(nq):
                t0 = time.perf_counter()
                res = retr.search(query_keys[i], query_vecs[i], window=window, k=K_RERANK)
                times.append((time.perf_counter() - t0) * 1e3)
                tk = _map_full(res["topk"], index_idx)
                retrieved[i, : tk.shape[0]] = tk
                cs = _map_full(res["candidates"], index_idx)
                cand_sets.append(cs)
                sizes.append(cs.shape[0])
            coarse = {k: benchmark.coarse_recall_at_k(cand_sets, oracle_full, k) for k in config.K_VALUES}
            lat = benchmark.summarize_latency(times)
            rows += _rows("sfc_multicurve", c, budget, retrieved, oracle_full, lat, build_s, coarse, int(np.mean(sizes)))
            r10 = next(r for r in rows[-3:] if r["k"] == 10)
            print(f"C={c} budget={budget:5d} window={window:4d} cand~{int(np.mean(sizes)):5d} "
                  f"coarse@10={coarse[10]:.3f} e2e@10={r10['recall_at_k']:.3f} p50={lat['p50']:.2f}ms")

    # HNSW reference, same run, same machine state
    t = time.time()
    hnsw = FaissHNSW(M=32, ef_construction=200).build(index_vecs)
    hnsw_build = time.time() - t
    hnsw.index.hnsw.efSearch = HNSW_EF
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
    rows += _rows(f"faiss_hnsw_ef{HNSW_EF}", "", "", retrieved, oracle_full, lat, hnsw_build)
    r10 = next(r for r in rows[-3:] if r["k"] == 10)
    print(f"hnsw ef={HNSW_EF} e2e@10={r10['recall_at_k']:.3f} p50={lat['p50']:.2f}ms (reference)")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "multicurve.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
