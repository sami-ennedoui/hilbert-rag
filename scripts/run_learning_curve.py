"""Recompute the projection training curves (InfoNCE and triplet) per d_low.

This is the training-only slice of the Phase 1 ablation: same anchors, exact-NN ranking,
hard-negative band, InfoNCE pairs, seed, and epoch count as run_ablation.py, so the curves
match the committed ablation result, but without the SFC build and cascade eval. It writes
results/learning_curve.csv; the figure is rendered by scripts/make_plots.py from that CSV.

Run offline:  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 .venv/bin/python scripts/run_learning_curve.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

from hilbert_rag import config, oracle, projection

# Mirror run_ablation.py exactly so the curves correspond to the ablation's trained heads.
N_ANCHOR = 4000
N_POS = 10
NEG_LO, NEG_HI = 50, 500
N_NEG = 4
EPOCHS = 15


def main() -> None:
    t0 = time.time()
    emb = np.load(config.DATA_DIR / "embeddings.npy")
    split = np.load(config.DATA_DIR / "split.npz")
    index_vecs = emb[split["index_idx"]]
    n_index = index_vecs.shape[0]
    print(f"index={n_index} dim={emb.shape[1]}")

    rng = np.random.default_rng(config.SEED)
    anchors = np.sort(rng.choice(n_index, size=min(N_ANCHOR, n_index), replace=False))
    t = time.time()
    ranking = oracle.neighbor_ranking(index_vecs, anchors, width=NEG_HI, block=512)
    print(f"anchor ranking {ranking.shape} in {time.time() - t:.1f}s")

    a_rows, p_local, n_local = projection.mine_triplets(
        ranking, n_pos=N_POS, neg_lo=NEG_LO, neg_hi=NEG_HI, n_neg=N_NEG, seed=config.SEED
    )
    a_vecs, p_vecs, n_vecs = index_vecs[anchors[a_rows]], index_vecs[p_local], index_vecs[n_local]
    nce_a = index_vecs[np.repeat(anchors, N_POS)]
    nce_p = index_vecs[ranking[:, :N_POS].reshape(-1)]

    curves: list[tuple[str, int, int, float]] = []
    for d_low in config.D_LOW_OPTIONS:
        _, h_triplet = projection.train_projection(a_vecs, p_vecs, n_vecs, d_low=d_low, epochs=EPOCHS, seed=config.SEED)
        _, h_infonce = projection.train_projection_infonce(nce_a, nce_p, d_low=d_low, epochs=EPOCHS, seed=config.SEED)
        curves += [("triplet", d_low, e, loss) for e, loss in enumerate(h_triplet)]
        curves += [("infonce", d_low, e, loss) for e, loss in enumerate(h_infonce)]
        print(f"d_low={d_low} triplet {h_triplet[0]:.4f}->{h_triplet[-1]:.4f}  infonce {h_infonce[0]:.4f}->{h_infonce[-1]:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "learning_curve.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["objective", "d_low", "epoch", "loss"])
        w.writerows(curves)
    print(f"wrote {out} ({len(curves)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
