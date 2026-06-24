"""Benchmark metrics and plots.

The metrics here are pure and unit-tested. Recall is measured against the exact-NN
oracle (see oracle.py). Coarse recall (the candidate set before rerank) is reported
separately from end-to-end recall@k, because the candidate set is the hard ceiling:
if a true neighbor is not in the candidates, the exact rerank cannot recover it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(retrieved: np.ndarray, oracle: np.ndarray, k: int) -> float:
    """Mean over queries of |retrieved top-k ∩ true top-k| / k.

    retrieved and oracle are (Q, >=k) arrays of item positions; only the first k
    columns of each are used. Order within the top-k does not matter.
    """
    retrieved = np.asarray(retrieved)
    oracle = np.asarray(oracle)
    q = retrieved.shape[0]
    total = 0.0
    for i in range(q):
        truth = set(oracle[i, :k].tolist())
        got = set(retrieved[i, :k].tolist())
        total += len(truth & got) / k
    return total / q


def coarse_recall_at_k(candidate_sets: Sequence[np.ndarray], oracle: np.ndarray, k: int) -> float:
    """Mean over queries of |candidate set ∩ true top-k| / k.

    candidate_sets is a per-query sequence of variable-length position arrays (the
    SFC candidates before rerank); oracle is (Q, >=k).
    """
    oracle = np.asarray(oracle)
    q = len(candidate_sets)
    total = 0.0
    for i in range(q):
        truth = set(oracle[i, :k].tolist())
        cand = set(np.asarray(candidate_sets[i]).tolist())
        total += len(truth & cand) / k
    return total / q
