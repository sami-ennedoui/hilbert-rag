import pandas as pd

from hilbert_rag import data

RECORDS = [
    {"id": "1", "abstract": "  a b ", "categories": "cs.LG x", "update_date": "2023-05-01", "title": "T1"},
    {"id": "2", "abstract": "c",      "categories": "math.PR",  "update_date": "2019-01-01", "title": "T2"},
    {"id": "3", "abstract": "d",      "categories": "cs.LG",    "update_date": "2024-06-01", "title": "T3"},
]


def test_filter_and_sample_respects_category_date_and_is_deterministic():
    df1 = data.filter_and_sample(RECORDS, categories=("cs.LG",),
                                 date_min="2022-01-01", date_max="2025-01-01",
                                 target_n=2, seed=1234)
    assert set(df1["primary_category"]) == {"cs.LG"}
    assert set(df1["id"]) == {"1", "3"}            # id 2 dropped by category
    assert (df1["text"] == df1["text"].str.strip()).all()
    df2 = data.filter_and_sample(RECORDS, categories=("cs.LG",),
                                 date_min="2022-01-01", date_max="2025-01-01",
                                 target_n=2, seed=1234)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_chunk_corpus_one_per_abstract_with_stable_ids():
    df = pd.DataFrame({"id": ["1", "3"], "text": ["a b", "d"],
                       "primary_category": ["cs.LG", "cs.LG"],
                       "published": ["2023-05-01", "2024-06-01"], "year": [2023, 2024]})
    out = data.chunk_corpus(df)
    assert len(out) == 2
    assert out["chunk_id"].is_unique
    assert list(out["chunk_id"]) == list(data.chunk_corpus(df)["chunk_id"])
