# Projection ablation: reading of the result

Source: `results/ablation_projection.csv` (24 rows). Seed 1234, 39,700-vector index,
300 held-out queries, exact cosine-NN oracle. Candidate budget fixed at ~1,000
(window 500) so the only thing that varies is the projection.

Two metrics per projector and `d_low`:
- **curve coarse recall@k**: fraction of the true top-k found in the SFC candidate set.
- **curve-free ceiling@k** (`lowdim_nn_coarse`): exact cosine-NN in the low-dim space
  at the same budget. The best the curve could do if its 1-D order lost nothing.

## Numbers at k=10

| projector          | curve d=8 | curve d=16 | ceiling d=8 | ceiling d=16 |
|--------------------|-----------|------------|-------------|--------------|
| random             | 0.151     | 0.126      | 0.354       | 0.598        |
| PCA                | 0.322     | 0.319      | 0.836       | 0.955        |
| learned (triplet)  | 0.191     | 0.156      | 0.520       | 0.735        |
| learned (InfoNCE)  | 0.353     | 0.247      | 0.890       | 0.962        |

## What it says

1. **The triplet head failed.** It trained (loss fell to near zero) but produced a
   worse projection than linear PCA. A margin between mined pairs does not preserve
   the global cosine geometry the oracle scores against.

2. **InfoNCE fixed it.** With in-batch negatives the learned projection has the best
   curve-free ceiling at both dimensions (0.890 vs PCA 0.836 at d=8; 0.962 vs 0.955
   at d=16). The trained model is genuinely the best projection, so the claim that a
   learned head helps is honest, once the objective is right.

3. **At d=8 the learned head also wins on the system metric** (SFC coarse recall@10
   0.353 vs PCA 0.322). d=8 is the better operating point for the curve, and that is
   the configuration carried into the rest of the benchmark.

4. **At d=16 PCA wins on the curve despite a worse ceiling.** PCA concentrates
   variance in the early axes, which the Hilbert order handles well. Neighbor
   fidelity and curve-friendliness are not the same property. The learned head is not
   trained to be axis-aligned, so its better neighbor structure does not transfer to
   the curve at higher dimension.

5. **The curve is the bottleneck, not the projection.** Even PCA's 0.955 ceiling
   collapses to 0.319 through the Hilbert order at a 2.5% scan budget. The 1-D curve
   discards most of the neighbor structure regardless of projection. This is the known
   high-dimensional limit of space-filling-curve search and it means unfiltered recall
   will lose to HNSW. That comparison is Phase 2; "beats FAISS" on unfiltered recall is
   not a claim this project makes.

## Carried forward

Operating configuration for Phase 2 and Phase 3: **InfoNCE projection, d_low = 8.**
