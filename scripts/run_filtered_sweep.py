"""Phase 3: filtered retrieval, the headline. Recall and latency vs selectivity.

A filtered query asks for the top-k among the subset passing a metadata predicate.
Selectivity is the fraction of the corpus that passes. To isolate selectivity as the
single variable, the sweep uses seeded random masks of the target size; a real
metadata predicate (filtered.build_mask) is what the service uses, but its behavior
here would depend on its particular selectivity, so the controlled mask is the clean
comparison. Three strategies, each scaling its effort as the filter tightens:

  exact_prefilter   mask, then exact search over the subset. Recall 1.0 by definition;
                    the latency baseline. At low selectivity the subset is tiny, so it
                    is both exact and cheap, the bar the approximate methods must clear.
  hnsw_postfilter   HNSW over-retrieves top-M (M ~ k/selectivity), drops non-matches,
                    keeps top-k. Expected to degrade as selectivity falls.
  sfc_filter        SFC scans a curve window (~ k/selectivity), intersects the mask,
                    reranks the survivors exactly.

All in local index-position space against the per-filter exact oracle. Single-thread
for comparable latency.

Outputs: results/filtered_sweep.csv
Run:  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
      .venv/bin/python scripts/run_filtered_sweep.py
"""

from __future__ import annotations

import csv
import math
import time

import faiss
import numpy as np
import torch

from hilbert_rag import benchmark, config, oracle, projection
from hilbert_rag.baselines import FaissHNSW
from hilbert_rag.cascade import exact_rerank
from hilbert_rag.sfc_index import SFCIndex

BITS = 10
K_RERANK = config.K_ORACLE
N_WARMUP = 3


def _budget(selectivity: float, n: int) -> int:
    """Retrieval budget that targets ~K_RERANK survivors after filtering."""
    return int(min(n, math.ceil(K_RERANK / max(selectivity, 1.0 / n))))


def main() -> None:
    faiss.omp_set_num_threads(1)
    emb = np.load(config.DATA_DIR / "embeddings.npy")
    split = np.load(config.DATA_DIR / "split.npz")
    query_idx, index_idx = split["query_idx"], split["index_idx"]
    index_vecs = emb[index_idx]
    query_vecs = emb[query_idx]
    n_index, nq = index_vecs.shape[0], query_vecs.shape[0]
    print(f"index={n_index} queries={nq}")

    head = projection.load_head(config.DATA_DIR / "projection_head.pt")
    with torch.no_grad():
        index_keys = head(torch.from_numpy(np.ascontiguousarray(index_vecs))).numpy()
        query_keys = head(torch.from_numpy(np.ascontiguousarray(query_vecs))).numpy()

    sfc = SFCIndex(bits=BITS).build(index_keys)
    hnsw = FaissHNSW(M=32, ef_construction=200).build(index_vecs)
    rng = np.random.default_rng(config.SEED)
    rows: list[dict] = []

    for s in config.SELECTIVITY_GRID:
        n_keep = max(K_RERANK, int(round(s * n_index)))
        keep = np.sort(rng.choice(n_index, size=n_keep, replace=False))
        mask = np.zeros(n_index, dtype=bool)
        mask[keep] = True
        actual_s = float(mask.mean())
        budget = _budget(actual_s, n_index)

        # per-filter exact oracle (also the exact_prefilter result)
        ora, _ = oracle.exact_topk_masked(query_vecs, index_vecs, mask, K_RERANK)

        # exact_prefilter: subset once, then per-query exact search over it
        sub_vecs = index_vecs[keep]
        for i in range(N_WARMUP):
            _ = sub_vecs @ query_vecs[i]
        times = []
        for i in range(nq):
            t0 = time.perf_counter()
            sims = sub_vecs @ query_vecs[i]
            kk = min(K_RERANK, sims.shape[0])
            np.argpartition(-sims, kk - 1)[:kk]
            times.append((time.perf_counter() - t0) * 1e3)
        lat = benchmark.summarize_latency(times)
        _emit(rows, "exact_prefilter", actual_s, ora, ora, lat, budget, n_keep)

        # hnsw_postfilter: over-retrieve, drop non-matches, keep top-k
        hnsw.index.hnsw.efSearch = int(min(n_index, max(64, budget)))
        retrieved = np.full((nq, K_RERANK), -1, dtype=np.int64)
        times = []
        for i in range(N_WARMUP):
            hnsw.search(query_vecs[i : i + 1], budget)
        for i in range(nq):
            t0 = time.perf_counter()
            pos, _ = hnsw.search(query_vecs[i : i + 1], budget)
            surv = pos[0][mask[pos[0]]][:K_RERANK]
            times.append((time.perf_counter() - t0) * 1e3)
            retrieved[i, : surv.shape[0]] = surv
        lat = benchmark.summarize_latency(times)
        _emit(rows, "hnsw_postfilter", actual_s, retrieved, ora, lat, budget, n_keep)

        # sfc_filter: curve window, intersect mask, rerank survivors
        retrieved = np.full((nq, K_RERANK), -1, dtype=np.int64)
        times = []
        for i in range(N_WARMUP):
            sfc.query(query_keys[i], budget // 2)
        for i in range(nq):
            t0 = time.perf_counter()
            cand = sfc.query(query_keys[i], budget // 2)
            surv = cand[mask[cand]]
            tk, _ = exact_rerank(query_vecs[i], surv, index_vecs, K_RERANK)
            times.append((time.perf_counter() - t0) * 1e3)
            retrieved[i, : tk.shape[0]] = tk
        lat = benchmark.summarize_latency(times)
        _emit(rows, "sfc_filter", actual_s, retrieved, ora, lat, budget, n_keep)

        print(
            f"s={actual_s:.3f} budget={budget:5d}  "
            f"exact p50={_p50(rows,'exact_prefilter',actual_s):.2f}ms  "
            f"hnsw r@10={_r10(rows,'hnsw_postfilter',actual_s):.3f}/{_p50(rows,'hnsw_postfilter',actual_s):.2f}ms  "
            f"sfc r@10={_r10(rows,'sfc_filter',actual_s):.3f}/{_p50(rows,'sfc_filter',actual_s):.2f}ms"
        )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "filtered_sweep.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def _emit(rows, method, s, retrieved, ora, lat, budget, subset):
    for k in config.K_VALUES:
        rows.append(
            {
                "method": method,
                "selectivity": round(s, 4),
                "k": k,
                "recall_at_k": round(benchmark.recall_at_k(retrieved, ora, k), 4),
                "lat_p50_ms": round(lat["p50"], 3),
                "lat_p95_ms": round(lat["p95"], 3),
                "budget": budget,
                "subset_size": subset,
            }
        )


def _row(rows, method, s, k=10):
    for r in rows:
        if r["method"] == method and abs(r["selectivity"] - round(s, 4)) < 1e-9 and r["k"] == k:
            return r
    return {"recall_at_k": float("nan"), "lat_p50_ms": float("nan")}


def _r10(rows, method, s):
    return _row(rows, method, s)["recall_at_k"]


def _p50(rows, method, s):
    return _row(rows, method, s)["lat_p50_ms"]


if __name__ == "__main__":
    main()
