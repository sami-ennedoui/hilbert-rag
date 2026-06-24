"""arXiv corpus ingestion and chunking.

These functions are pure and unit-tested. The download lives in
scripts/build_corpus.py, which fetches the parquet shards once, filters each one
with filter_frame, then concatenates, samples, chunks, and writes corpus.parquet.
Keeping the predicate here means the filtering logic is tested without the network.
"""

from __future__ import annotations

import pandas as pd

_COLUMNS = ["id", "text", "primary_category", "published", "year", "title"]


def submission_date_from_id(arxiv_id: str) -> str | None:
    """Submission date 'YYYY-MM-01' parsed from a modern arXiv id 'YYMM.NNNNN'.

    The id encodes the original submission month, a cleaner 'published' date than
    the update_date field, which records the last modification. Returns None for
    legacy ids (no YYMM prefix) or an invalid month.
    """
    if "." not in arxiv_id:
        return None
    head = arxiv_id.split(".")[0]
    if len(head) != 4 or not head.isdigit():
        return None
    yy, mm = int(head[:2]), int(head[2:4])
    if not 1 <= mm <= 12:
        return None
    year = 2000 + yy if yy < 90 else 1900 + yy
    return f"{year:04d}-{mm:02d}-01"


def filter_frame(
    df: pd.DataFrame,
    categories: tuple[str, ...],
    date_min: str,
    date_max: str,
) -> pd.DataFrame:
    """Keep rows whose primary category is in `categories` and whose id-derived
    submission date is within [date_min, date_max], with normalized text.

    Vectorized so it runs on a full parquet shard at once. The date is parsed from
    the arXiv id (submission month), not the update_date field. Returns the columns
    in `_COLUMNS`; no sampling here, the caller samples after concatenating shards.
    """
    cats = set(categories)
    d = df.copy()
    d["primary_category"] = d["categories"].fillna("").astype(str).str.split().str[0]
    d = d[d["primary_category"].isin(cats)]
    d["published"] = d["id"].astype(str).map(submission_date_from_id)
    d = d[d["published"].notna()]
    d = d[(d["published"] >= date_min) & (d["published"] <= date_max)]
    d["text"] = d["abstract"].fillna("").astype(str).str.split().str.join(" ")
    d = d[d["text"].str.len() > 0]
    d["year"] = d["published"].str.slice(0, 4).astype(int)
    d["title"] = d["title"].fillna("").astype(str).str.split().str.join(" ")
    d["id"] = d["id"].astype(str)
    return d[_COLUMNS].reset_index(drop=True)


def chunk_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """One chunk per abstract, with a stable chunk_id of the form '<id>#0'.

    Abstracts are short, so a single chunk per document is the documented choice.
    Longer documents would use fixed-size overlapping windows instead.
    """
    out = df.copy()
    out["chunk_id"] = out["id"].astype(str) + "#0"
    ordered = ["chunk_id"] + [c for c in out.columns if c != "chunk_id"]
    return out[ordered].reset_index(drop=True)
