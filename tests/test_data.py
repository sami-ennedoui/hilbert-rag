import pandas as pd

from hilbert_rag import data


def test_filter_frame_filters_by_id_date_and_category():
    raw = pd.DataFrame(
        {
            "id":         ["1801.00001", "1901.00002", "9901.00003", "2012.00004", "1805.00005"],
            "abstract":   ["  a b ",     "c",          "d",          "e",          ""],
            "categories": ["cs.LG x",    "stat.ML",    "cs.LG",      "math.OC",    "cs.LG"],
            "title":      ["T1",         "T2",         "T3",         "T4",         "T5"],
        }
    )
    out = data.filter_frame(
        raw, categories=("cs.LG", "stat.ML"), date_min="2018-01-01", date_max="2024-12-31"
    )
    # 1801 cs.LG 2018 keep; 1901 stat.ML 2019 keep; 9901 -> 1999 out of window;
    # 2012 math.OC wrong category; 1805 cs.LG but empty abstract dropped.
    assert set(out["id"]) == {"1801.00001", "1901.00002"}
    assert list(out.columns) == ["id", "text", "primary_category", "published", "year", "title"]
    assert (out["text"] == out["text"].str.strip()).all()
    assert out.loc[out["id"] == "1801.00001", "year"].iloc[0] == 2018


def test_filter_frame_empty_when_nothing_matches():
    raw = pd.DataFrame({"id": ["math/0309136"], "abstract": ["x"], "categories": ["cs.LG"], "title": ["T"]})
    out = data.filter_frame(raw, categories=("cs.LG",), date_min="2018-01-01", date_max="2024-12-31")
    assert len(out) == 0
    assert list(out.columns) == ["id", "text", "primary_category", "published", "year", "title"]


def test_submission_date_from_arxiv_id():
    # Modern ids encode YYMM as the submission month; legacy ids return None.
    assert data.submission_date_from_id("1406.0214") == "2014-06-01"
    assert data.submission_date_from_id("2401.12345") == "2024-01-01"
    assert data.submission_date_from_id("0704.0001") == "2007-04-01"
    assert data.submission_date_from_id("hep-th/9901001") is None
    assert data.submission_date_from_id("math/0309136") is None
    assert data.submission_date_from_id("1413.0001") is None   # month 13 is invalid


def test_chunk_corpus_one_per_abstract_with_stable_ids():
    df = pd.DataFrame({"id": ["1", "3"], "text": ["a b", "d"],
                       "primary_category": ["cs.LG", "cs.LG"],
                       "published": ["2023-05-01", "2024-06-01"], "year": [2023, 2024]})
    out = data.chunk_corpus(df)
    assert len(out) == 2
    assert out["chunk_id"].is_unique
    assert list(out["chunk_id"]) == list(data.chunk_corpus(df)["chunk_id"])
