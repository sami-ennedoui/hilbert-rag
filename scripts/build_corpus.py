"""Build the arXiv corpus and cache it to data/corpus.parquet.

Downloads the snapshot parquet shards once (authenticated, resumable), filters
each shard locally to the configured categories and id-derived submission-date
window, concatenates, takes a deterministic sample, chunks one row per abstract,
and writes the parquet plus a provenance record. One-time; the benchmark runs
offline from the parquet afterward. Robust against the mid-scan connection drops
that the streaming path suffered.

Run: HF_TOKEN=... .venv/bin/python scripts/build_corpus.py
"""

from __future__ import annotations

import glob
import json
import os
import time

import pandas as pd
from huggingface_hub import snapshot_download

from hilbert_rag import config, data


def _counts(series, *, as_year=False):
    vc = series.value_counts()
    items = ((str(k), int(v)) for k, v in vc.items())
    return dict(sorted(items)) if as_year else dict(sorted(items, key=lambda kv: -kv[1]))


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    local = snapshot_download(
        config.ARXIV_DATASET,
        repo_type="dataset",
        allow_patterns="*.parquet",
        token=os.getenv("HF_TOKEN"),
    )
    files = sorted(glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True))
    print(f"downloaded {len(files)} parquet shards in {time.time() - t0:.0f}s", flush=True)

    parts = []
    for i, f in enumerate(files, 1):
        shard = pd.read_parquet(f, columns=["id", "categories", "abstract", "title"])
        keep = data.filter_frame(shard, config.ARXIV_CATEGORIES, config.DATE_MIN, config.DATE_MAX)
        parts.append(keep)
        print(f"  shard {i}/{len(files)}: {len(shard)} rows -> {len(keep)} matches", flush=True)
        del shard

    matches = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("id")
        .sort_values("id")
        .reset_index(drop=True)
    )
    if len(matches) > config.TARGET_CORPUS_SIZE:
        matches = (
            matches.sample(n=config.TARGET_CORPUS_SIZE, random_state=config.SEED)
            .sort_values("id")
            .reset_index(drop=True)
        )
    chunks = data.chunk_corpus(matches)

    out = config.DATA_DIR / "corpus.parquet"
    chunks.to_parquet(out, index=False)

    provenance = {
        "source": config.ARXIV_DATASET,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "snapshot_download parquet shards, per-shard vectorized filter",
        "categories": list(config.ARXIV_CATEGORIES),
        "date_field": "submission_date_from_arxiv_id (YYYY-MM-01)",
        "date_min": config.DATE_MIN,
        "date_max": config.DATE_MAX,
        "target_n": config.TARGET_CORPUS_SIZE,
        "seed": config.SEED,
        "total_matches_before_sample": int(sum(len(p) for p in parts)),
        "n_chunks": int(len(chunks)),
        "by_year": _counts(chunks["year"], as_year=True),
        "by_category": _counts(chunks["primary_category"]),
        "build_seconds": round(time.time() - t0, 1),
    }
    (config.DATA_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"wrote {out} with {len(chunks)} chunks in {provenance['build_seconds']}s", flush=True)
    print("by_year:", provenance["by_year"], flush=True)
    print("by_category:", provenance["by_category"], flush=True)


if __name__ == "__main__":
    main()
