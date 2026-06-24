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


def _quantize_to_grid(keys: np.ndarray, lo: np.ndarray, hi: np.ndarray, bits: int) -> np.ndarray:
    """Map float keys to integer grid coordinates in [0, 2**bits - 1] using the
    per-dimension range [lo, hi] fitted at build time; out-of-range values are clamped."""
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (np.asarray(keys, dtype=np.float64) - lo) / span
    maxv = (1 << bits) - 1
    return np.clip(np.rint(scaled * maxv).astype(np.int64), 0, maxv)


class SFCIndex:
    def __init__(self, bits: int = 10):
        self.bits = bits

    def _quantize(self, keys: np.ndarray) -> np.ndarray:
        return _quantize_to_grid(keys, self.lo, self.hi, self.bits)

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


class MultiCurveSFCIndex:
    """C shifted Hilbert curves over one shared grid; union of per-curve windows.

    A single space-filling curve splits some spatial neighbors across a recursion
    boundary, which is the main reason single-curve recall collapses in high dimension.
    Each curve here applies a distinct diagonal offset to the quantized keys, wrapped
    modulo the grid size, so a point on a boundary in one curve is interior in another.
    The union of the per-curve windows recovers neighbors any one curve drops. This is
    the standard multiple-shifted-curve fix (Leutenegger & Mokbel; HD-Index). With
    n_curves == 1 the offset is 0 and the candidate set equals a plain SFCIndex.

    `query(key, window)` uses `window` as the per-curve half-width, so the union budget
    is at most ``n_curves * 2 * window`` before de-duplication; the benchmark fixes the
    total budget by setting window = total // (2 * n_curves).
    """

    def __init__(self, bits: int = 10, n_curves: int = 4):
        if n_curves < 1:
            raise ValueError("n_curves must be >= 1")
        self.bits = bits
        self.n_curves = n_curves

    def _offsets(self) -> list[int]:
        """Distinct diagonal offsets evenly spaced along the wrapped grid; first is 0."""
        m = 1 << self.bits
        return [int(round(c / self.n_curves * m)) % m for c in range(self.n_curves)]

    def build(self, keys: np.ndarray) -> "MultiCurveSFCIndex":
        keys = np.asarray(keys, dtype=np.float64)
        self.n, self.d = keys.shape
        self.lo = keys.min(axis=0)
        self.hi = keys.max(axis=0)
        self.m = 1 << self.bits
        self.hc = HilbertCurve(self.bits, self.d)
        q = _quantize_to_grid(keys, self.lo, self.hi, self.bits)
        self.orders: list[np.ndarray] = []
        self.sorted_dists: list[list[int]] = []
        for off in self._offsets():
            shifted = (q + off) % self.m
            dists = self.hc.distances_from_points(shifted.tolist())
            order = sorted(range(self.n), key=dists.__getitem__)
            self.orders.append(np.asarray(order, dtype=np.int64))
            self.sorted_dists.append([dists[i] for i in order])
        return self

    def query(self, key: np.ndarray, window: int) -> np.ndarray:
        """Union of the +/- window curve slices across all curves. Positions index the
        array passed to build(); the result is sorted and de-duplicated."""
        q = _quantize_to_grid(np.asarray(key, dtype=np.float64).reshape(1, -1), self.lo, self.hi, self.bits)[0]
        parts: list[np.ndarray] = []
        for off, order, sdists in zip(self._offsets(), self.orders, self.sorted_dists):
            shifted = ((q + off) % self.m).tolist()
            d = self.hc.distance_from_point(shifted)
            pos = bisect.bisect_left(sdists, d)
            lo = max(0, pos - window)
            hi = min(self.n, pos + window)
            parts.append(order[lo:hi])
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))
