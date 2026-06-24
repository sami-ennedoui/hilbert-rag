"""Train and persist the chosen projection head (InfoNCE, d_low=8).

The ablation (scripts/run_ablation.py, results/ablation_findings.md) selected InfoNCE
at d_low=8 as the projection that best feeds the SFC index. This script trains exactly
that head and saves it to data/projection_head.pt so the recall benchmark, the filtered
sweep, and the service all reuse one frozen projection instead of retraining.

Run offline:  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 .venv/bin/python scripts/train_projection.py
"""

from __future__ import annotations

import numpy as np

from hilbert_rag import config, oracle, projection

D_LOW = 8
N_ANCHOR = 4000
N_POS = 10
NEG_HI = 500     # ranking width retained (the InfoNCE path uses only the top N_POS)
EPOCHS = 15


def main() -> None:
    emb = np.load(config.DATA_DIR / "embeddings.npy")
    split = np.load(config.DATA_DIR / "split.npz")
    index_vecs = emb[split["index_idx"]]
    n_index = index_vecs.shape[0]

    rng = np.random.default_rng(config.SEED)
    anchors = np.sort(rng.choice(n_index, size=min(N_ANCHOR, n_index), replace=False))
    ranking = oracle.neighbor_ranking(index_vecs, anchors, width=NEG_HI, block=512)

    nce_a = index_vecs[np.repeat(anchors, N_POS)]
    nce_p = index_vecs[ranking[:, :N_POS].reshape(-1)]
    head, history = projection.train_projection_infonce(
        nce_a, nce_p, d_low=D_LOW, in_dim=emb.shape[1], epochs=EPOCHS, seed=config.SEED
    )
    print(f"infonce d_low={D_LOW} loss {history[0]:.4f} -> {history[-1]:.4f}")

    out = config.DATA_DIR / "projection_head.pt"
    projection.save_head(
        head,
        out,
        in_dim=emb.shape[1],
        hidden=128,
        out_dim=D_LOW,
        meta={
            "objective": "infonce",
            "d_low": D_LOW,
            "n_anchor": N_ANCHOR,
            "n_pos": N_POS,
            "epochs": EPOCHS,
            "seed": config.SEED,
            "final_loss": round(history[-1], 4),
        },
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
