"""Build the arXiv corpus and cache it to data/corpus.parquet.

Streams the HF snapshot once, derives the submission date from each arXiv id,
filters to the configured categories and date window, takes a deterministic
sample, chunks one row per abstract, and writes the parquet plus a provenance
record. One-time: the benchmark runs offline from the parquet afterward.

Run: .venv/bin/python scripts/build_corpus.py
"""

from __future__ import annotations

import json
import os
import sys
import time

from datasets import load_dataset

from hilbert_rag import config, data


def _with_submission_date(ds):
    """Yield each record with update_date replaced by the id-derived submission
    date, skipping legacy ids that carry no parseable date."""
    for r in ds:
        published = data.submission_date_from_id(r.get("id", ""))
        if published is None:
            continue
        r = dict(r)
        r["update_date"] = published
        yield r


def _counts(series, *, as_year=False):
    vc = series.value_counts()
    items = ((str(k), int(v)) for k, v in vc.items())
    return dict(sorted(items)) if as_year else dict(sorted(items, key=lambda kv: -kv[1]))


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = load_dataset(config.ARXIV_DATASET, split="train", streaming=True)
    df = data.filter_and_sample(
        _with_submission_date(ds),
        categories=config.ARXIV_CATEGORIES,
        date_min=config.DATE_MIN,
        date_max=config.DATE_MAX,
        target_n=config.TARGET_CORPUS_SIZE,
        seed=config.SEED,
    )
    chunks = data.chunk_corpus(df)

    out = config.DATA_DIR / "corpus.parquet"
    chunks.to_parquet(out, index=False)

    provenance = {
        "source": config.ARXIV_DATASET,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "categories": list(config.ARXIV_CATEGORIES),
        "date_field": "submission_date_from_arxiv_id (YYYY-MM-01)",
        "date_min": config.DATE_MIN,
        "date_max": config.DATE_MAX,
        "target_n": config.TARGET_CORPUS_SIZE,
        "seed": config.SEED,
        "n_chunks": int(len(chunks)),
        "by_year": _counts(chunks["year"], as_year=True),
        "by_category": _counts(chunks["primary_category"]),
        "scan_seconds": round(time.time() - t0, 1),
    }
    (config.DATA_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"wrote {out} with {len(chunks)} chunks in {provenance['scan_seconds']}s")
    print("by_year:", provenance["by_year"])
    print("by_category:", provenance["by_category"])
    sys.stdout.flush()
    # HF streaming leaves prefetch threads that crash at interpreter teardown;
    # the parquet is already flushed, so exit hard to avoid a spurious core dump.
    os._exit(0)


if __name__ == "__main__":
    main()
