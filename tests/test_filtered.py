import pandas as pd

from hilbert_rag import filtered


def _meta():
    return pd.DataFrame(
        {
            "primary_category": ["cs.LG", "cs.CV", "cs.LG", "stat.ML"],
            "year": [2018, 2020, 2024, 2019],
        }
    )


def test_build_mask_category_and_year():
    m = filtered.build_mask(_meta(), categories=("cs.LG",), year_range=(2018, 2020))
    assert list(m) == [True, False, False, False]   # cs.LG AND 2018<=year<=2020 -> row 0 only


def test_build_mask_year_only_and_selectivity():
    m = filtered.build_mask(_meta(), year_range=(2019, 2024))
    assert list(m) == [False, True, True, True]
    assert filtered.selectivity(m) == 0.75


def test_build_mask_no_predicate_is_all_true():
    m = filtered.build_mask(_meta())
    assert m.all() and len(m) == 4
    assert filtered.selectivity(m) == 1.0
