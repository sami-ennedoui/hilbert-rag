# Hilbert-RAG

A CPU-native RAG retrieval service. Its differentiator is a Hilbert space-filling-curve index built for filtered retrieval, with a trained PyTorch projection that makes the curve viable on semantic embeddings.

Status: under construction. Build state lives in `BUILD-PLAN.md`.

## What this is, honestly

This is a leakage-safe benchmark of a space-filling-curve index for dense retrieval, and the headline result is negative. On unfiltered recall FAISS HNSW wins and it is not close. Under a selective metadata filter, exact search over the filtered subset wins on both recall and latency. At this corpus size the curve does not win a retrieval regime, and the benchmark says so plainly.

The value of the project is what it measures and how honestly it does it.

- A trained PyTorch projection feeds the curve measurably better than PCA once the objective is right. An InfoNCE head beats PCA on the candidate-set recall the system actually depends on.
- The standard multiple-shifted-curve fix is implemented and measured. Eight shifted curves recover a large part of the single-curve deficit while scanning fewer candidates, and they still do not catch HNSW. The benchmark shows exactly how much the fix buys and where it stops.
- The failure is decomposed, not guessed. The projection preserves most of the neighbor structure, and the one-dimensional curve order then discards it. Those two losses are measured separately.
- The index is small and dependency-light, about a hundred lines you can read end to end.

This is a reproduction and extension of a published line of work, HD-Index and Leutenegger and Mokbel, not novel research. Learned projection into a one-dimensional index is also known, for instance LIDER. Saying so is part of what makes the benchmark credible.

The retrieval quality oracle is exact nearest neighbors in embedding space, not human-judged relevance. Every number in the results comes from a measurement saved to disk, never a rounded or hoped-for figure.

## Layout

```
src/hilbert_rag/
  embeddings.py   embed the corpus and cache it
  oracle.py       exact nearest-neighbor ground truth, filtered and unfiltered
  sfc_index.py    Hilbert key, sorted array, window query
  cascade.py      coarse candidates then exact rerank
  projection.py   PyTorch MLP head, contrastive loss, hard-negative mining
  baselines.py    FAISS Flat and HNSW wrappers
  filtered.py     metadata predicate and the selectivity sweep
  benchmark.py    metrics and plots
  service.py      FastAPI /search and /ask
tests/            pytest suite
results/          CSV tables and figures, committed
data/             corpus and cached embeddings, gitignored, rebuilt from scripts
```

## Setup

CPU only, Python 3.12, podman for the container build.

```bash
bash scripts/setup-env.sh      # creates .venv (py3.12), installs CPU torch + the rest
source .venv/bin/activate
```

## Results

All numbers below are measured and saved under `results/`. The full reading is in `results/ablation_findings.md` and `results/filtered_findings.md`.

Unfiltered recall@10 and median latency, single thread:

| method | recall@10 | p50 |
|--------|-----------|-----|
| FAISS HNSW, ef=64 | 0.999 | 0.36 ms |
| FAISS Flat, exact | 1.000 | 5.51 ms |
| single-curve SFC, about 10k candidates | 0.667 | 10.6 ms |
| 8-curve SFC, about 10k candidates | 0.830 | 7.65 ms |

HNSW wins on both axes. The eight-curve fix narrows the gap and is the correct way to use the curve, but it does not close the gap.

Under filtering, exact search over the filtered subset reaches recall 1.0 and is the fastest method once the filter is selective, down to 0.05 ms at one percent selectivity. The curve's recall under filtering stays too low to compete. The sweep is in `results/filtered_sweep.csv`.

Projection ablation at the curve's operating point, d=8 and about 1k candidates: the InfoNCE head reaches coarse recall@10 of 0.353, against 0.322 for PCA and 0.151 for a random projection. The learned projection is genuinely the best key for the curve.
