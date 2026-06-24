"""Service tests, fully hermetic: a tiny synthetic engine and a stub LLM, no model
download, no data on disk, no network. The fake embedder hashes a string to a fixed
unit vector, and the corpus is embedded with that same fake embedder, so querying a
corpus text returns that exact item as the top hit. That lets us assert real retrieval
behavior (top hit, filter honored, both backends) without a real embedding model.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hilbert_rag.service import RetrievalEngine, create_app

D = 8


def _fake_embed(texts):
    """Deterministic unit vectors keyed by text content. No model involved."""
    out = np.zeros((len(texts), D), dtype=np.float32)
    for i, t in enumerate(texts):
        seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little")
        v = np.random.default_rng(seed).standard_normal(D).astype(np.float32)
        out[i] = v / max(np.linalg.norm(v), 1e-12)
    return out


def _build_engine():
    n = 12
    texts = [f"abstract number {i} about retrieval and curves" for i in range(n)]
    titles = [f"Title {i}" for i in range(n)]
    ids = [f"p{i}#0" for i in range(n)]
    categories = ["cs.LG" if i % 2 == 0 else "cs.CL" for i in range(n)]
    years = [2018 + (i % 6) for i in range(n)]
    vecs = _fake_embed(texts)  # corpus embedded with the same fake embedder
    return RetrievalEngine(
        vecs=vecs,
        ids=ids,
        categories=categories,
        years=years,
        texts=texts,
        titles=titles,
        embed_fn=_fake_embed,
        project_fn=lambda v: np.ascontiguousarray(np.asarray(v)[:, :2], dtype=np.float64),
        n_curves=2,
        sfc_bits=4,
        sfc_window=n,  # scan everything so the sfc path is deterministic in the test
    ), texts, ids, categories


class _StubLLM:
    """Stand-in for the HF router client; records the prompt, returns a fixed answer."""

    def __init__(self):
        self.last_user = None

    def complete(self, system: str, user: str) -> str:
        self.last_user = user
        return "STUB_ANSWER grounded in the provided sources."


@pytest.fixture
def client_and_data():
    engine, texts, ids, cats = _build_engine()
    llm = _StubLLM()
    app = create_app(engine, llm=llm)
    return TestClient(app), texts, ids, cats, llm


def test_healthz_ok(client_and_data):
    client = client_and_data[0]
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_search_default_backend_returns_ids_and_scores(client_and_data):
    client, texts, ids, _, _ = client_and_data
    r = client.post("/search", json={"query": texts[3], "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "faiss_hnsw"
    results = body["results"]
    assert 1 <= len(results) <= 3
    for hit in results:
        assert "id" in hit and "score" in hit
    assert results[0]["id"] == ids[3]  # querying a corpus text returns that item first


def test_search_sfc_backend_works(client_and_data):
    client, texts, ids, _, _ = client_and_data
    r = client.post("/search", json={"query": texts[5], "k": 3, "backend": "sfc"})
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "sfc"
    assert body["results"][0]["id"] == ids[5]


def test_search_filter_is_honored_end_to_end(client_and_data):
    client, texts, _, _, _ = client_and_data
    r = client.post(
        "/search",
        json={"query": texts[0], "k": 6, "filter": {"categories": ["cs.LG"]}},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) >= 1
    assert all(hit["category"] == "cs.LG" for hit in results)  # every result passes the filter


def test_search_rejects_unknown_backend(client_and_data):
    client, texts, _, _, _ = client_and_data
    r = client.post("/search", json={"query": texts[0], "k": 3, "backend": "nope"})
    assert r.status_code in (400, 422)


def test_ask_returns_answer_with_cited_ids(client_and_data):
    client, texts, ids, _, llm = client_and_data
    r = client.post("/ask", json={"query": texts[2], "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "STUB_ANSWER" in body["answer"]
    assert len(body["cited_ids"]) >= 1
    assert set(body["cited_ids"]).issubset(set(ids))  # citations are real corpus ids
    assert llm.last_user is not None and texts[2] in llm.last_user  # query reached the LLM


def test_ask_without_llm_returns_503():
    engine, _, _, _ = _build_engine()
    client = TestClient(create_app(engine, llm=None))
    r = client.post("/ask", json={"query": "anything"})
    assert r.status_code == 503


def test_search_faiss_flat_backend_is_exact_and_honors_filter(client_and_data):
    client, texts, ids, _, _ = client_and_data
    r = client.post("/search", json={"query": texts[7], "k": 3, "backend": "faiss_flat"})
    assert r.status_code == 200
    assert r.json()["backend"] == "faiss_flat"
    assert r.json()["results"][0]["id"] == ids[7]  # exact backend returns the queried item first
    r2 = client.post(
        "/search",
        json={"query": texts[0], "k": 6, "backend": "faiss_flat", "filter": {"categories": ["cs.CL"]}},
    )
    assert r2.status_code == 200
    assert all(h["category"] == "cs.CL" for h in r2.json()["results"])


def test_ask_grounds_only_on_filtered_chunks(client_and_data):
    client, texts, ids, cats, _ = client_and_data
    id_to_cat = dict(zip(ids, cats))
    r = client.post("/ask", json={"query": texts[1], "k": 4, "filter": {"categories": ["cs.LG"]}})
    assert r.status_code == 200
    cited = r.json()["cited_ids"]
    assert len(cited) >= 1
    assert all(id_to_cat[cid] == "cs.LG" for cid in cited)  # every grounding chunk passes the filter
