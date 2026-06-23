from hilbert_rag import config


def test_core_constants():
    assert config.EMB_DIM == 384
    assert config.SEED == 1234
    assert config.K_ORACLE >= max(config.K_VALUES)
    assert config.SELECTIVITY_GRID[0] == 1.0
    assert 1.0 > config.SELECTIVITY_GRID[-1] > 0
    assert len(config.ARXIV_CATEGORIES) >= 4
    assert config.DATA_DIR.name == "data"
    assert config.RESULTS_DIR.name == "results"
