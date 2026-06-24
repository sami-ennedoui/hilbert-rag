import numpy as np

from hilbert_rag import benchmark


def test_recall_at_k_basic():
    oracle = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    retrieved = np.array([[1, 2, 9, 4], [6, 5, 0, 0]])
    # @4: q0 {1,2,4} of {1,2,3,4}=3/4; q1 {5,6}=2/4; avg=(0.75+0.5)/2=0.625
    assert abs(benchmark.recall_at_k(retrieved, oracle, k=4) - 0.625) < 1e-9
    # @2: q0 {1,2}=2/2; q1 {6,5}=2/2; avg=1.0
    assert benchmark.recall_at_k(retrieved, oracle, k=2) == 1.0


def test_coarse_recall_at_k_with_variable_candidate_sets():
    oracle = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    cands = [np.array([1, 2, 3, 99, 98]), np.array([5, 0])]
    # q0 {1,2,3}=3/4; q1 {5}=1/4; avg=0.5
    assert abs(benchmark.coarse_recall_at_k(cands, oracle, k=4) - 0.5) < 1e-9


def test_recall_is_one_when_retrieved_covers_truth():
    oracle = np.array([[10, 11, 12]])
    retrieved = np.array([[12, 11, 10]])     # order-insensitive
    assert benchmark.recall_at_k(retrieved, oracle, k=3) == 1.0


def test_summarize_latency_reports_percentiles():
    times = [float(t) for t in range(1, 101)]   # 1..100 ms
    s = benchmark.summarize_latency(times)
    assert set(s) == {"p50", "p95", "mean", "n"}
    assert s["n"] == 100
    assert abs(s["mean"] - 50.5) < 1e-9
    assert abs(s["p50"] - np.percentile(times, 50)) < 1e-9
    assert abs(s["p95"] - np.percentile(times, 95)) < 1e-9
    assert s["p95"] >= s["p50"]
