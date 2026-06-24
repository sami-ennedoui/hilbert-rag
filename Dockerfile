# CPU-only image for the Hilbert-RAG service. Demo mode is the default, so the
# container serves /search and /ask with a synthetic corpus and no token or data.
# For real retrieval, run with -e HILBERT_RAG_DEMO=0, the data directory mounted at
# /app/data, and HF_TOKEN set.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HILBERT_RAG_DEMO=1 \
    OMP_NUM_THREADS=4

WORKDIR /app

# Install the CPU torch wheel first so pip does not pull the large CUDA build.
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the package and its dependencies. README and pyproject are needed by the build.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "hilbert_rag.service:build_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
