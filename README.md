# Hilbert-RAG

A CPU-native RAG retrieval service. Its differentiator is a Hilbert space-filling-curve index built for filtered retrieval, with a trained PyTorch projection that makes the curve viable on semantic embeddings.

Status: under construction. Build state lives in `BUILD-PLAN.md`.

## What this is, honestly

On pure unfiltered recall, FAISS HNSW wins, and this project says so plainly. The space-filling-curve index earns its place in three specific regimes:

- Filtered retrieval, where a metadata predicate composes natively with the curve while graph indexes degrade at low selectivity.
- Cheap streaming inserts into a sorted structure, versus a graph relink.
- A small, dependency-light index, about a hundred lines, whose behavior you can read end to end.

This is a reproduction and extension of a published line of work, HD-Index and Leutenegger and Mokbel, not novel research. Stating that is part of what makes the benchmark credible.

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

To come. See `BUILD-PLAN.md` for what is built so far.
