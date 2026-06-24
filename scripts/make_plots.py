"""Regenerate the result figures from the committed CSVs. No recomputation: every
number comes from results/*.csv, so the plots always match the tables.

Produces, in results/:
  plot_recall_latency.png      unfiltered recall@10 vs p50 latency (SFC window sweep,
                               HNSW efSearch sweep, exact Flat point). Shows HNSW dominating.
  plot_filtered_selectivity.png  recall@10 and p50 latency vs selectivity for exact
                               pre-filter, HNSW post-filter, and the SFC filter.
  plot_multicurve.png          coarse recall@10 vs number of shifted curves, at two
                               candidate budgets, against the HNSW reference line.

Run:  .venv/bin/python scripts/make_plots.py
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hilbert_rag import config


def _load(name: str) -> list[dict]:
    path = config.RESULTS_DIR / name
    with path.open() as f:
        return list(csv.DictReader(f))


def _at_k(rows: list[dict], k: int = 10) -> list[dict]:
    return [r for r in rows if int(r["k"]) == k]


def plot_recall_latency() -> None:
    rows = _at_k(_load("recall_unfiltered.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))

    def series(method):
        pts = sorted(
            ((float(r["lat_p50_ms"]), float(r["recall_at_k"]), r["param"]) for r in rows if r["method"] == method),
            key=lambda t: t[0],
        )
        return pts

    for method, label, style in [
        ("sfc", "SFC cascade (window sweep)", "o-"),
        ("faiss_hnsw", "FAISS HNSW (efSearch sweep)", "s-"),
    ]:
        pts = series(method)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=label)
    flat = series("faiss_flat")
    if flat:
        ax.plot(flat[0][0], flat[0][1], "D", color="black", label="FAISS Flat (exact)")

    ax.set_xlabel("p50 latency (ms, single thread)")
    ax.set_ylabel("recall@10")
    ax.set_title("Unfiltered retrieval: recall vs latency\nHNSW dominates the SFC cascade on both axes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = config.RESULTS_DIR / "plot_recall_latency.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_filtered_selectivity() -> None:
    rows = _at_k(_load("filtered_sweep.csv"))
    methods = [
        ("exact_prefilter", "exact pre-filter"),
        ("hnsw_postfilter", "HNSW post-filter"),
        ("sfc_filter", "SFC filter"),
    ]
    fig, (axr, axl) = plt.subplots(1, 2, figsize=(12, 5))
    for method, label in methods:
        pts = sorted(
            ((float(r["selectivity"]), float(r["recall_at_k"]), float(r["lat_p50_ms"])) for r in rows if r["method"] == method),
            key=lambda t: t[0],
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        axr.plot(xs, [p[1] for p in pts], "o-", label=label)
        axl.plot(xs, [p[2] for p in pts], "o-", label=label)
    for ax in (axr, axl):
        ax.set_xscale("log")
        ax.set_xlabel("selectivity (fraction of corpus passing the filter)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axr.set_ylabel("recall@10")
    axr.set_title("Recall under filter")
    axl.set_yscale("log")
    axl.set_ylabel("p50 latency (ms)")
    axl.set_title("Latency under filter")
    fig.suptitle("Filtered retrieval: exact pre-filter wins when the filter is selective")
    fig.tight_layout()
    out = config.RESULTS_DIR / "plot_filtered_selectivity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_multicurve() -> None:
    rows = _at_k(_load("multicurve.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    budgets = sorted({int(r["total_budget"]) for r in rows if r["method"] == "sfc_multicurve"})
    for budget in budgets:
        pts = sorted(
            ((int(r["n_curves"]), float(r["coarse_recall_at_k"])) for r in rows if r["method"] == "sfc_multicurve" and int(r["total_budget"]) == budget),
            key=lambda t: t[0],
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"~{budget} candidate budget")
    hnsw = [float(r["recall_at_k"]) for r in rows if r["method"].startswith("faiss_hnsw")]
    if hnsw:
        ax.axhline(hnsw[0], ls="--", color="gray", label=f"HNSW reference ({hnsw[0]:.3f})")
    ax.set_xlabel("number of shifted Hilbert curves (C)")
    ax.set_ylabel("coarse recall@10")
    ax.set_title("Multi-curve recovers part of the single-curve deficit\nbut does not close the gap to HNSW")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = config.RESULTS_DIR / "plot_multicurve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def plot_learning_curve() -> None:
    path = config.RESULTS_DIR / "learning_curve.csv"
    if not path.exists():
        print("(skip learning curve: results/learning_curve.csv not found; run scripts/run_learning_curve.py)")
        return
    rows = _load("learning_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, obj, title in [(axes[0], "infonce", "InfoNCE (the head used)"), (axes[1], "triplet", "Triplet (underperformed PCA)")]:
        for d in sorted({int(r["d_low"]) for r in rows if r["objective"] == obj}):
            pts = sorted(
                ((int(r["epoch"]), float(r["loss"])) for r in rows if r["objective"] == obj and int(r["d_low"]) == d),
                key=lambda t: t[0],
            )
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"d_low={d}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{title} training loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Projection training: both objectives converge, but only InfoNCE beat PCA downstream")
    fig.tight_layout()
    out = config.RESULTS_DIR / "plot_learning_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_recall_latency()
    plot_filtered_selectivity()
    plot_multicurve()
    plot_learning_curve()


if __name__ == "__main__":
    main()
