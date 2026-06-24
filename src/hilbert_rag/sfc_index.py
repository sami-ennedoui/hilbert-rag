"""A Hilbert space-filling-curve index over low-dimensional keys.

Each key is quantized per coordinate to `bits` bits, mapped to a single Hilbert
distance, and stored in an array sorted by that distance. A query maps to its own
Hilbert distance, binary-searches its position, and returns a window of +/- W
neighbors in curve order as the candidate set. Small and dependency-light: the only
non-stdlib pieces are numpy and the Hilbert point<->distance mapping.

The locality argument (why neighbors on the curve tend to be neighbors in space, and
where it breaks in high dimension) is the part to defend; see the README.
"""

from __future__ import annotations

import bisect

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


class SFCIndex:
    def __init__(self, bits: int = 10):
        self.bits = bits

    def _quantize(self, keys: np.ndarray) -> np.ndarray:
        """Map float keys to integer grid coordinates in [0, 2**bits - 1] using the
        per-dimension range fitted at build time; out-of-range queries are clamped."""
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        scaled = (np.asarray(keys, dtype=np.float64) - self.lo) / span
        maxv = (1 << self.bits) - 1
        return np.clip(np.rint(scaled * maxv).astype(np.int64), 0, maxv)

    def build(self, keys: np.ndarray) -> "SFCIndex":
        keys = np.asarray(keys, dtype=np.float64)
        self.n, self.d = keys.shape
        self.lo = keys.min(axis=0)
        self.hi = keys.max(axis=0)
        self.hc = HilbertCurve(self.bits, self.d)
        dists = self.hc.distances_from_points(self._quantize(keys).tolist())
        order = sorted(range(self.n), key=dists.__getitem__)
        self.order = np.asarray(order, dtype=np.int64)
        self.sorted_dists = [dists[i] for i in order]
        return self

    def query(self, key: np.ndarray, window: int) -> np.ndarray:
        """Return candidate positions: a +/- window slice of the curve order around
        the query's Hilbert position. Positions index the array passed to build()."""
        q = self._quantize(np.asarray(key, dtype=np.float64).reshape(1, -1))[0]
        d = self.hc.distance_from_point(q.tolist())
        pos = bisect.bisect_left(self.sorted_dists, d)
        lo = max(0, pos - window)
        hi = min(self.n, pos + window)
        return self.order[lo:hi]
