import numpy as np

from hilbert_rag.sfc_index import MultiCurveSFCIndex, SFCIndex


def _grid_points():
    # 2-D points on a small integer grid, as floats.
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [3, 3], [3, 2]], dtype=np.float32)


def test_build_orders_by_hilbert_distance():
    pts = _grid_points()
    idx = SFCIndex(bits=2).build(pts)
    assert idx.sorted_dists == sorted(idx.sorted_dists)          # ascending
    assert sorted(idx.order.tolist()) == list(range(len(pts)))   # a permutation


def test_query_window_is_contiguous_band():
    pts = _grid_points()
    idx = SFCIndex(bits=2).build(pts)
    cand = idx.query(pts[0], window=2).tolist()
    assert 1 <= len(cand) <= 4                                   # up to 2*window
    order = idx.order.tolist()
    start = order.index(cand[0])
    assert order[start:start + len(cand)] == cand                # a contiguous slice


def test_identical_point_retrieves_itself():
    pts = _grid_points()
    idx = SFCIndex(bits=4).build(pts)
    assert 4 in idx.query(pts[4], window=2).tolist()             # point [3,3]


def test_quantize_clamps_out_of_range_queries():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    idx = SFCIndex(bits=3).build(pts)
    q = idx._quantize(np.array([[2.0, -1.0]], dtype=np.float32))[0]
    assert q.min() >= 0 and q.max() <= (2**3 - 1)


def test_multicurve_offsets_are_distinct_and_first_is_zero():
    idx = MultiCurveSFCIndex(bits=10, n_curves=4)
    offs = idx._offsets()
    assert len(offs) == 4
    assert offs[0] == 0                       # first curve is the unshifted one
    assert len(set(offs)) == 4                # all distinct


def test_multicurve_c1_matches_single_curve_as_a_set():
    pts = _grid_points()
    single = SFCIndex(bits=4).build(pts)
    multi = MultiCurveSFCIndex(bits=4, n_curves=1).build(pts)
    for w in (1, 2, 3):
        assert set(multi.query(pts[0], window=w).tolist()) == set(
            single.query(pts[0], window=w).tolist()
        )


def test_multicurve_union_is_superset_of_one_curve_and_finds_self():
    pts = _grid_points()
    single = SFCIndex(bits=4).build(pts)
    multi = MultiCurveSFCIndex(bits=4, n_curves=4).build(pts)
    one = set(single.query(pts[4], window=1).tolist())
    union = set(multi.query(pts[4], window=1).tolist())
    assert 4 in union                         # point [3,3] retrieves itself
    assert one <= union                       # the unshifted curve is one of the C curves


def test_multicurve_rejects_zero_curves():
    try:
        MultiCurveSFCIndex(bits=4, n_curves=0)
    except ValueError:
        return
    raise AssertionError("n_curves=0 should raise ValueError")
