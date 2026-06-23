"""arXiv corpus ingestion and chunking.

The two functions here are pure and unit-tested. The download itself lives in
scripts/build_corpus.py, which streams the snapshot, hands records to
filter_and_sample, then chunks and writes corpus.parquet. Keeping the predicate
here means the filtering logic is tested without touching the network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

_COLUMNS = ["id", "text", "primary_category", "published", "year", "title"]


def _primary_category(raw: str) -> str:
    parts = (raw or "").split()
    return parts[0] if parts else ""


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


def filter_and_sample(
    records: Iterable[Mapping],
    categories: tuple[str, ...],
    date_min: str,
    date_max: str,
    target_n: int,
    seed: int,
) -> pd.DataFrame:
    """Keep records whose primary category is in `categories` and whose date is
    within [date_min, date_max], normalize text, then take a deterministic sample
    of at most `target_n` rows.

    Records are filtered row by row, so passing a streaming dataset only holds the
    matching rows in memory. Dates are ISO strings (YYYY-MM-DD), compared lexically.
    """
    cats = set(categories)
    rows = []
    for r in records:
        primary = _primary_category(r.get("categories", ""))
        if primary not in cats:
            continue
        published = r.get("update_date", "") or ""
        if not (date_min <= published <= date_max):
            continue
        text = " ".join((r.get("abstract", "") or "").split())
        if not text:
            continue
        year = int(published[:4]) if published[:4].isdigit() else 0
        rows.append(
            {
                "id": str(r["id"]),
                "text": text,
                "primary_category": primary,
                "published": published,
                "year": year,
                "title": " ".join((r.get("title", "") or "").split()),
            }
        )

    df = pd.DataFrame(rows, columns=_COLUMNS)
    df = df.drop_duplicates(subset="id").sort_values("id").reset_index(drop=True)
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=seed).sort_values("id").reset_index(drop=True)
    return df


def chunk_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """One chunk per abstract, with a stable chunk_id of the form '<id>#0'.

    Abstracts are short, so a single chunk per document is the documented choice.
    Longer documents would use fixed-size overlapping windows instead.
    """
    out = df.copy()
    out["chunk_id"] = out["id"].astype(str) + "#0"
    ordered = ["chunk_id"] + [c for c in out.columns if c != "chunk_id"]
    return out[ordered].reset_index(drop=True)
