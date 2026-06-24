# Filtered retrieval sweep: reading of the result

Source: `results/filtered_sweep.csv`. Seed 1234, 39,700-vector index, 300 queries,
controlled random masks at each selectivity, per-filter exact-NN oracle. Single thread.
Each approximate method scales its retrieval budget as ~k / selectivity. recall@10 / p50:

| selectivity | exact_prefilter | hnsw_postfilter | sfc_filter |
|-------------|-----------------|-----------------|------------|
| 1.00        | 1.000 / 5.61 ms | 0.999 / 0.49 ms | 0.163 / 0.21 ms |
| 0.50        | 1.000 / 2.73 ms | 0.999 / 0.92 ms | 0.198 / 0.21 ms |
| 0.25        | 1.000 / 1.19 ms | 1.000 / 1.59 ms | 0.228 / 0.21 ms |
| 0.10        | 1.000 / 0.32 ms | 1.000 / 3.93 ms | 0.271 / 0.21 ms |
| 0.05        | 1.000 / 0.14 ms | 1.000 / 8.81 ms | 0.328 / 0.21 ms |
| 0.01        | 1.000 / 0.05 ms | 1.000 / 112.49 ms | 0.507 / 0.21 ms |

## What it says

1. **Exact pre-filter wins at low selectivity, and it is not close.** As the filter
   tightens, the subset shrinks, so exact brute-force over the subset gets faster and
   stays exact. At 1% selectivity it answers in 0.05 ms with recall 1.0. The correct
   engineering choice for a selective filter is: apply the filter, then brute-force the
   small remainder.

2. **HNSW post-filter degrades as predicted.** To return k matches when only 1% of the
   corpus passes, it must over-retrieve thousands of candidates; latency rises from
   0.49 ms at full selectivity to 112 ms at 1%. Post-filtering is the wrong strategy
   exactly where a filter is most useful.

3. **The SFC index has flat, very low latency but recall too low to use.** A curve query
   is one Hilbert-distance computation plus a slice, so latency (~0.21 ms) is
   independent of selectivity and budget. But recall@10 ranges 0.16 to 0.51 and never
   approaches the exact answer. It is fast and wrong; there is no selectivity where it
   offers a recall/latency trade-off that beats both exact pre-filter and HNSW.

## Honest conclusion across Phases 1-3

- The learned InfoNCE projection genuinely beats PCA at feeding the curve (Phase 1).
- The Hilbert space-filling-curve index is Pareto-dominated by HNSW on unfiltered recall
  (Phase 2) and has no winning niche under filtering (Phase 3); exact pre-filter is the
  right tool when the filter is selective, HNSW when it is not.
- This is a negative result, reported as one. The space-filling-curve approach does not
  compete with HNSW or with exact pre-filtering for high-dimensional dense retrieval.
  The value of the work is the rigor: a trained projection that beats PCA, leakage-safe
  evaluation, and a clear, measured account of why the curve fails and what to use
  instead.
