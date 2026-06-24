"""Phase 1 ablation: learned vs PCA vs random projection through the SFC index.

The headline of the projection work (spec §5.4, rule 2). Each projector turns the
384-D embedding into a low-dimensional key; the same Hilbert index and the same
candidate budget (window) are used for all three, so any difference in coarse or
end-to-end recall is attributable to the projection alone, not the index.

Train/eval hygiene: the projection is trained only on anchors drawn from the index
(searchable) set, never the held-out queries. Hard negatives come from the exact-NN
ranking band [NEG_LO, NEG_HI). Recall is scored against the precomputed exact-NN
oracle in full-corpus position space.

Outputs:
  results/ablation_projection.csv   one row per (key_type, d_low, k)
  results/learning_curve.csv        per-epoch training loss for each learned d_low
  results/learning_curve.png        the same, rendered (best effort)
Run offline:  HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 .venv/bin/python scripts/run_ablation.py
"""

from __future__ import annotations

import csv
import time

import numpy as np
import torch

from hilbert_rag import config, oracle, projection
from hilbert_rag.benchmark import coarse_recall_at_k, recall_at_k
from hilbert_rag.cascade import CascadeRetriever
from hilbert_rag.oracle import exact_topk
from hilbert_rag.sfc_index import SFCIndex

# --- ablation knobs (documented, fixed by SEED) ---
N_ANCHOR = 4000          # training anchors sampled from the index set
N_POS = 10               # positives per anchor (closest exact neighbors)
NEG_LO, NEG_HI = 50, 500  # hard-negative band (ranks), spec §5.4
N_NEG = 4                # hard negatives sampled per (anchor, positive)
EPOCHS = 15
WINDOW = 500             # +/- curve neighbors -> ~2*WINDOW candidates (equal budget for all)
BITS = 10                # quantization resolution per key dimension
K_RERANK = config.K_ORACLE  # rerank depth; must be >= max(K_VALUES)


def _learned_transform(head: projection.ProjectionHead):
    head.eval()

    def transform(vecs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(vecs, dtype=np.float32))
            return head(t).numpy()

    return transform


def main() -> None:
    t0 = time.time()
    emb = np.load(config.DATA_DIR / "embeddings.npy")
    split = np.load(config.DATA_DIR / "split.npz")
    query_idx, index_idx = split["query_idx"], split["index_idx"]
    oracle_full = np.load(config.DATA_DIR / "oracle_unfiltered.npy")  # (Q, K_ORACLE), full-corpus pos

    index_vecs = emb[index_idx]      # (N_index, 384) searchable corpus, local space
    query_vecs = emb[query_idx]      # (Q, 384)
    n_index = index_vecs.shape[0]
    print(f"index={n_index} queries={query_vecs.shape[0]} dim={emb.shape[1]}")

    # --- training signal: anchor exact-NN ranking, then hard-negative triplets ---
    rng = np.random.default_rng(config.SEED)
    anchors_local = np.sort(rng.choice(n_index, size=min(N_ANCHOR, n_index), replace=False))
    t = time.time()
    ranking = oracle.neighbor_ranking(index_vecs, anchors_local, width=NEG_HI, block=512)
    print(f"anchor ranking {ranking.shape} in {time.time() - t:.1f}s")
    a_rows, p_local, n_local = projection.mine_triplets(
        ranking, n_pos=N_POS, neg_lo=NEG_LO, neg_hi=NEG_HI, n_neg=N_NEG, seed=config.SEED
    )
    a_vecs = index_vecs[anchors_local[a_rows]]
    p_vecs = index_vecs[p_local]
    n_vecs = index_vecs[n_local]
    print(f"triplets: {len(a_rows)}")

    # InfoNCE pairs: each anchor with each of its top-N_POS true neighbors.
    nce_a = index_vecs[np.repeat(anchors_local, N_POS)]
    nce_p = index_vecs[ranking[:, :N_POS].reshape(-1)]

    # --- build the projectors per d_low ---
    learn_curves: list[tuple[str, int, int, float]] = []   # (objective, d_low, epoch, loss)
    final_loss: dict[int, float] = {}
    rows: list[dict] = []

    for d_low in config.D_LOW_OPTIONS:
        head, history = projection.train_projection(
            a_vecs, p_vecs, n_vecs, d_low=d_low, epochs=EPOCHS, seed=config.SEED
        )
        head_nce, history_nce = projection.train_projection_infonce(
            nce_a, nce_p, d_low=d_low, epochs=EPOCHS, seed=config.SEED
        )
        final_loss[d_low] = history[-1]
        learn_curves += [("triplet", d_low, e, loss) for e, loss in enumerate(history)]
        learn_curves += [("infonce", d_low, e, loss) for e, loss in enumerate(history_nce)]
        print(
            f"d_low={d_low} triplet {history[0]:.4f}->{history[-1]:.4f}  "
            f"infonce {history_nce[0]:.4f}->{history_nce[-1]:.4f}"
        )

        projectors = {
            "learned": _learned_transform(head),
            "learned_nce": _learned_transform(head_nce),
            "pca": projection.pca_projector(index_vecs, d_low, config.SEED),
            "random": projection.random_projector(emb.shape[1], d_low, config.SEED),
        }

        for key_type, proj in projectors.items():
            index_keys = proj(index_vecs)
            query_keys = proj(query_vecs)

            # curve-free ceiling: exact cosine-NN in the low-dim space at the same
            # candidate budget. Isolates projection quality from curve degradation:
            # if this is high but curve recall is low, the Hilbert curve is the loss.
            budget = 2 * WINDOW
            low_nn_local, _ = exact_topk(query_keys, index_keys, budget)
            low_nn_sets = [index_idx[low_nn_local[i]] for i in range(low_nn_local.shape[0])]

            tb = time.time()
            sfc = SFCIndex(bits=BITS).build(index_keys)
            build_s = time.time() - tb
            retriever = CascadeRetriever(sfc, index_vecs)

            retrieved = np.full((query_keys.shape[0], K_RERANK), -1, dtype=np.int64)
            cand_sets: list[np.ndarray] = []
            cand_sizes = []
            for i in range(query_keys.shape[0]):
                res = retriever.search(query_keys[i], query_vecs[i], window=WINDOW, k=K_RERANK)
                topk_full = index_idx[res["topk"]]
                retrieved[i, : topk_full.shape[0]] = topk_full
                cand_full = index_idx[res["candidates"]]
                cand_sets.append(cand_full)
                cand_sizes.append(cand_full.shape[0])

            for k in config.K_VALUES:
                rows.append(
                    {
                        "key_type": key_type,
                        "d_low": d_low,
                        "k": k,
                        "recall_at_k": round(recall_at_k(retrieved, oracle_full, k), 4),
                        "coarse_recall_at_k": round(coarse_recall_at_k(cand_sets, oracle_full, k), 4),
                        "lowdim_nn_coarse": round(coarse_recall_at_k(low_nn_sets, oracle_full, k), 4),
                        "mean_cand_size": int(np.mean(cand_sizes)),
                        "window": WINDOW,
                        "build_s": round(build_s, 2),
                        "train_loss_final": round(final_loss[d_low], 4) if key_type == "learned" else "",
                    }
                )
            print(
                f"  {key_type:7s} d={d_low:2d} cand~{int(np.mean(cand_sizes)):5d} "
                f"coarse@10={rows[-2]['coarse_recall_at_k']:.3f} "
                f"e2e@10={rows[-2]['recall_at_k']:.3f}"
            )

    # --- write results ---
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "ablation_projection.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lc = config.RESULTS_DIR / "learning_curve.csv"
    with lc.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["objective", "d_low", "epoch", "loss"])
        w.writerows(learn_curves)
    _plot_learning_curve(learn_curves)
    print(f"wrote {out} ({len(rows)} rows) and {lc} in total {time.time() - t0:.1f}s")


def _plot_learning_curve(curves: list[tuple[str, int, int, float]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping learning-curve plot: {e})")
        return
    objectives = ["triplet", "infonce"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, obj in zip(axes, objectives):
        rows = [(d, e, loss) for (o, d, e, loss) in curves if o == obj]
        arr = np.array(rows, dtype=float)
        for d in sorted(set(arr[:, 0])):
            m = arr[:, 0] == d
            ax.plot(arr[m, 1], arr[m, 2], marker="o", label=f"d_low={int(d)}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{obj} training loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "learning_curve.png", dpi=120)
    print("wrote results/learning_curve.png")


if __name__ == "__main__":
    main()
