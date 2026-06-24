"""Metadata predicate and selectivity for filtered retrieval (the headline).

A filter is a category set and/or a year range. build_mask turns it into a boolean
mask over the corpus; the SFC path restricts its candidate scan to True positions, and
the filtered oracle (oracle.exact_topk_masked) uses the same mask, so recall-under-
filter is measured honestly. selectivity is |filtered| / |corpus|, the x-axis of the
sweep where HNSW post-filtering is expected to degrade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_mask(
    meta: pd.DataFrame,
    categories: tuple[str, ...] | None = None,
    year_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Boolean mask over rows of `meta` satisfying the predicate. No predicate -> all True."""
    mask = np.ones(len(meta), dtype=bool)
    if categories is not None:
        mask &= meta["primary_category"].isin(set(categories)).to_numpy()
    if year_range is not None:
        lo, hi = year_range
        mask &= ((meta["year"] >= lo) & (meta["year"] <= hi)).to_numpy()
    return mask


def selectivity(mask: np.ndarray) -> float:
    """Fraction of the corpus passing the filter."""
    return float(np.asarray(mask, dtype=bool).mean())
