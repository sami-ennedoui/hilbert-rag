"""Frozen project configuration: constants, paths, and the experiment grid.

Every other module imports its knobs from here so runs are reproducible and a
single edit changes the whole pipeline. Nothing here has side effects.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
SEED = 1234

# --- Embeddings ------------------------------------------------------------
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384  # output dimension of all-MiniLM-L6-v2

# --- Projection / SFC key --------------------------------------------------
D_LOW_OPTIONS = (8, 16)  # low-dimensional key fed to the Hilbert curve

# --- Evaluation ------------------------------------------------------------
K_VALUES = (1, 10, 100)            # recall@k reported at these k
K_ORACLE = 100                     # neighbors precomputed by the exact oracle
SELECTIVITY_GRID = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01)  # |filtered| / |corpus|

# --- Corpus (arXiv) --------------------------------------------------------
# librarian-bots/arxiv-metadata-snapshot, filtered to these primary categories
# and date window, then deterministically sampled.
ARXIV_DATASET = "librarian-bots/arxiv-metadata-snapshot"
# Populous ML / stats / signal / optimization categories. The date field is the
# submission month parsed from the arXiv id, not the update_date.
ARXIV_CATEGORIES = (
    "cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML",
    "cs.NE", "cs.IR", "cs.RO", "eess.SP", "math.OC",
)
DATE_MIN = "2018-01-01"
DATE_MAX = "2024-12-31"
TARGET_CORPUS_SIZE = 40_000
N_QUERIES = 300

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
