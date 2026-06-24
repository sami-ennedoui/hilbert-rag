"""Build the arXiv corpus and cache it to data/corpus.parquet.

Downloads the snapshot parquet shards one at a time (authenticated, with retries and
a per-read timeout), filters each shard locally to the configured categories and
id-derived submission-date window right after it lands, then concatenates, samples,
chunks one row per abstract, and writes the parquet plus a provenance record.

Sequential single-connection downloads are deliberate: parallel multi-connection
downloads stalled repeatedly on this link. A shard that keeps failing is skipped and
recorded, rather than aborting the whole build. One-time; the benchmark then runs
offline from the parquet.

Run: HF_HUB_DOWNLOAD_TIMEOUT=30 .venv/bin/python scripts/build_corpus.py
(uses the HF token stored by `huggingface_hub.login`; no token env needed)
"""

from __future__ import annotations

import json
import time

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from hilbert_rag import config, data

_RETRIES = 6


def _download_shard(filename: str) -> str | None:
    """Download one shard with retries and exponential backoff. Returns the local
    path, or None if it keeps failing."""
    for attempt in range(1, _RETRIES + 1):
        try:
            return hf_hub_download(config.ARXIV_DATASET, filename, repo_type="dataset")
        except Exception as e:  # noqa: BLE001 - report and retry any download error
            wait = min(30, 2**attempt)
            print(f"  retry {attempt}/{_RETRIES} for {filename}: {type(e).__name__} (wait {wait}s)", flush=True)
            time.sleep(wait)
    return None


def _counts(series, *, as_year=False):
    vc = series.value_counts()
    items = ((str(k), int(v)) for k, v in vc.items())
    return dict(sorted(items)) if as_year else dict(sorted(items, key=lambda kv: -kv[1]))


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    shards = sorted(
        f for f in list_repo_files(config.ARXIV_DATASET, repo_type="dataset") if f.endswith(".parquet")
    )
    print(f"{len(shards)} shards to fetch", flush=True)

    parts, skipped = [], []
    for i, name in enumerate(shards, 1):
        path = _download_shard(name)
        if path is None:
            print(f"  shard {i}/{len(shards)} SKIPPED (download failed): {name}", flush=True)
            skipped.append(name)
            continue
        shard = pd.read_parquet(path, columns=["id", "categories", "abstract", "title"])
        keep = data.filter_frame(shard, config.ARXIV_CATEGORIES, config.DATE_MIN, config.DATE_MAX)
        parts.append(keep)
        print(f"  shard {i}/{len(shards)}: {len(shard)} rows -> {len(keep)} matches "
              f"[{time.time() - t0:.0f}s]", flush=True)
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
        "method": "hf_hub_download per shard (sequential, retries), per-shard vectorized filter",
        "categories": list(config.ARXIV_CATEGORIES),
        "date_field": "submission_date_from_arxiv_id (YYYY-MM-01)",
        "date_min": config.DATE_MIN,
        "date_max": config.DATE_MAX,
        "target_n": config.TARGET_CORPUS_SIZE,
        "seed": config.SEED,
        "shards_total": len(shards),
        "shards_skipped": skipped,
        "total_matches_before_sample": int(sum(len(p) for p in parts)),
        "n_chunks": int(len(chunks)),
        "by_year": _counts(chunks["year"], as_year=True),
        "by_category": _counts(chunks["primary_category"]),
        "build_seconds": round(time.time() - t0, 1),
    }
    (config.DATA_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"wrote {out} with {len(chunks)} chunks in {provenance['build_seconds']}s "
          f"(skipped {len(skipped)} shards)", flush=True)
    print("by_year:", provenance["by_year"], flush=True)
    print("by_category:", provenance["by_category"], flush=True)


if __name__ == "__main__":
    main()
