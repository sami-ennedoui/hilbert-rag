"""FastAPI service: /search and /ask.

`/search` runs the selected backend over the corpus and honors an optional metadata
filter. The default backend is `faiss_hnsw`, the production-correct choice; `sfc` (the
studied space-filling-curve cascade) and `faiss_flat` (exact) are selectable. `/ask`
does one retrieval and one LLM call and returns an answer with the cited chunk ids; the
LLM client is injected so it is stubbed in tests and the real HF router is wired only in
production.

The app is built by a factory so importing this module never loads a model or touches
disk. `build_default_app()` serves a small synthetic corpus in demo mode (the default in
the container, so `/search` responds with no data or token), or the cached real artifacts
when they are present. Run it with: uvicorn hilbert_rag.service:build_default_app --factory
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .baselines import FaissFlat, FaissHNSW
from .cascade import exact_rerank
from .sfc_index import MultiCurveSFCIndex

Backend = Literal["faiss_hnsw", "sfc", "faiss_flat"]


@dataclass
class Hit:
    id: str
    score: float
    title: str
    category: str
    year: int
    text: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class RetrievalEngine:
    """Holds the corpus, the built backends, and the query path. Constructed from
    in-memory arrays so it is equally easy to build from disk or from a test fixture."""

    def __init__(
        self,
        vecs: np.ndarray,
        ids: list[str],
        categories: list[str],
        years: list[int],
        texts: list[str],
        titles: list[str],
        embed_fn,
        project_fn=None,
        n_curves: int = 8,
        sfc_bits: int = 10,
        sfc_window: int = 125,
        hnsw_overfetch: int = 10,
    ):
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.n = self.vecs.shape[0]
        self.ids = list(ids)
        self.categories = np.asarray(categories)
        self.years = np.asarray(years, dtype=np.int64)
        self.texts = list(texts)
        self.titles = list(titles)
        self.embed_fn = embed_fn
        self.project_fn = project_fn
        self.sfc_window = sfc_window
        self.hnsw_overfetch = hnsw_overfetch

        self.flat = FaissFlat().build(self.vecs)
        self.hnsw = FaissHNSW(M=32, ef_construction=200, ef_search=64).build(self.vecs)
        if project_fn is not None:
            keys = np.asarray(project_fn(self.vecs))
            self.sfc = MultiCurveSFCIndex(bits=sfc_bits, n_curves=n_curves).build(keys)
        else:
            self.sfc = None

    def _mask(self, flt: "Filter | None") -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if flt is None:
            return mask
        if flt.categories:
            mask &= np.isin(self.categories, list(flt.categories))
        if flt.year_min is not None:
            mask &= self.years >= flt.year_min
        if flt.year_max is not None:
            mask &= self.years <= flt.year_max
        return mask

    def _exact(self, qvec: np.ndarray, mask: np.ndarray, k: int):
        pos = np.flatnonzero(mask)
        if pos.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        sims = self.vecs[pos] @ qvec
        kk = min(k, pos.size)
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        return pos[top], sims[top]

    def _hnsw(self, qvec: np.ndarray, mask: np.ndarray, k: int):
        fetch = int(min(self.n, max(k, k * self.hnsw_overfetch)))
        pos, sims = self.hnsw.search(qvec[None, :], fetch)
        pos, sims = pos[0], sims[0]
        valid = pos >= 0
        pos, sims = pos[valid], sims[valid]
        keep = mask[pos]
        return pos[keep][:k], sims[keep][:k]

    def _sfc(self, qvec: np.ndarray, mask: np.ndarray, k: int):
        if self.sfc is None or self.project_fn is None:
            raise HTTPException(status_code=400, detail="sfc backend not available (no projection)")
        qlow = np.asarray(self.project_fn(qvec[None, :]))[0]
        cand = self.sfc.query(qlow, self.sfc_window)
        cand = cand[mask[cand]]
        return exact_rerank(qvec, cand, self.vecs, k)

    def search(self, query: str, k: int, flt: "Filter | None", backend: Backend) -> tuple[list[Hit], float]:
        qvec = np.ascontiguousarray(self.embed_fn([query])[0], dtype=np.float32)
        mask = self._mask(flt)
        if backend == "faiss_flat":
            pos, sims = self._exact(qvec, mask, k)
        elif backend == "sfc":
            pos, sims = self._sfc(qvec, mask, k)
        else:
            pos, sims = self._hnsw(qvec, mask, k)
        hits = [
            Hit(
                id=self.ids[p],
                score=float(s),
                title=self.titles[p],
                category=str(self.categories[p]),
                year=int(self.years[p]),
                text=self.texts[p],
            )
            for p, s in zip(pos.tolist(), sims.tolist())
        ]
        return hits, float(mask.mean())


class Filter(BaseModel):
    categories: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    filter: Filter | None = None
    backend: Backend = "faiss_hnsw"


class AskRequest(BaseModel):
    query: str
    k: int = 5
    filter: Filter | None = None


def _build_prompt(query: str, hits: list[Hit]) -> tuple[str, str]:
    system = (
        "You answer strictly from the provided sources. Cite the bracketed source ids you "
        "use, like [id]. If the sources do not contain the answer, say you do not know."
    )
    sources = "\n\n".join(f"[{h.id}] {h.title}\n{h.text}" for h in hits)
    user = f"Question: {query}\n\nSources:\n{sources}\n\nAnswer with citations like [id]."
    return system, user


def create_app(engine: RetrievalEngine, llm: LLMClient | None = None) -> FastAPI:
    app = FastAPI(title="Hilbert-RAG", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "corpus_size": engine.n, "sfc_available": engine.sfc is not None}

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        hits, sel = engine.search(req.query, req.k, req.filter, req.backend)
        return {
            "backend": req.backend,
            "k": req.k,
            "selectivity": round(sel, 4),
            "results": [
                {"id": h.id, "score": round(h.score, 6), "title": h.title, "category": h.category, "year": h.year}
                for h in hits
            ],
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        if llm is None:
            raise HTTPException(status_code=503, detail="LLM not configured (set HF_TOKEN)")
        hits, _ = engine.search(req.query, req.k, req.filter, backend="faiss_hnsw")
        system, user = _build_prompt(req.query, hits)
        answer = llm.complete(system, user)
        return {
            "answer": answer,
            "cited_ids": [h.id for h in hits],
            "contexts": [{"id": h.id, "title": h.title} for h in hits],
        }

    return app


# --- Production / demo wiring (only touched by build_default_app, never at import) ---


def _hash_embed(texts, dim: int) -> np.ndarray:
    """Deterministic unit vectors keyed by text. Demo only, no model involved."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        seed = int.from_bytes(hashlib.sha256(str(t).encode()).digest()[:4], "little")
        v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
        out[i] = v / max(float(np.linalg.norm(v)), 1e-12)
    return out


class _EchoLLM:
    """Offline stand-in so /ask responds in demo mode without a token."""

    def complete(self, system: str, user: str) -> str:
        q = user.split("\n", 1)[0].replace("Question: ", "")
        return f"[demo mode, no live LLM] Retrieved sources for: {q}. See cited_ids for the grounding chunks."


def _demo_engine(n: int = 256, dim: int = config.EMB_DIM) -> RetrievalEngine:
    rng = np.random.default_rng(config.SEED)
    texts = [f"synthetic abstract {i} on retrieval, indexing and space-filling curves" for i in range(n)]
    titles = [f"Demo paper {i}" for i in range(n)]
    ids = [f"demo{i}#0" for i in range(n)]
    cats = [config.ARXIV_CATEGORIES[i % len(config.ARXIV_CATEGORIES)] for i in range(n)]
    years = [2018 + (i % 7) for i in range(n)]
    vecs = _hash_embed(texts, dim)
    proj = rng.standard_normal((dim, 8)).astype(np.float32) / np.sqrt(dim)
    return RetrievalEngine(
        vecs=vecs, ids=ids, categories=cats, years=years, texts=texts, titles=titles,
        embed_fn=lambda ts: _hash_embed(ts, dim),
        project_fn=lambda v: np.ascontiguousarray(np.asarray(v) @ proj, dtype=np.float64),
        n_curves=4, sfc_bits=10, sfc_window=64,
    )


def _engine_from_disk() -> RetrievalEngine:
    import pandas as pd

    from . import embeddings, projection

    ids_arr, vecs = embeddings.load_embeddings(config.DATA_DIR)
    df = pd.read_parquet(config.DATA_DIR / "corpus.parquet").set_index("chunk_id")
    df = df.loc[list(ids_arr)]  # align metadata to embedding order
    head = projection.load_head(config.DATA_DIR / "projection_head.pt")

    import torch

    def project_fn(v):
        with torch.no_grad():
            return head(torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))).numpy()

    return RetrievalEngine(
        vecs=vecs,
        ids=list(ids_arr),
        categories=df["primary_category"].tolist(),
        years=df["year"].tolist(),
        texts=df["text"].tolist(),
        titles=df["title"].tolist(),
        embed_fn=embeddings.embed_texts,
        project_fn=project_fn,
        n_curves=8, sfc_bits=10, sfc_window=125,
    )


def _llm_from_env() -> LLMClient | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    base_url = os.environ.get("HILBERT_RAG_LLM_BASE_URL", "https://router.huggingface.co/v1")
    model = os.environ.get("HILBERT_RAG_LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=token)

    class _HFRouterLLM:
        def complete(self, system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""

    return _HFRouterLLM()


def build_default_app() -> FastAPI:
    """uvicorn factory. Demo mode (synthetic corpus, offline) unless real artifacts exist
    and HILBERT_RAG_DEMO is not set; demo is the container default so /search always responds."""
    demo = os.environ.get("HILBERT_RAG_DEMO", "").lower() in ("1", "true", "yes")
    have_data = (config.DATA_DIR / "embeddings.npy").exists()
    if demo or not have_data:
        return create_app(_demo_engine(), llm=_EchoLLM())
    return create_app(_engine_from_disk(), llm=_llm_from_env())
