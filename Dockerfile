# The cra-rag API in a container.
#   docker build --tag cra-rag .
#   docker run --publish 8000:8000 cra-rag        -> http://localhost:8000/docs
# Alone, it answers /health and /search with BM25. For semantic search and
# answers it needs Ollama: see compose.yaml.

FROM python:3.12-slim

# uv, from its official image, pinned to a version
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# 1. Dependencies only. This layer is rebuilt only when uv.lock changes, so
#    changing the code does not download every library again.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 2. The project and its corpus.
COPY README.md LICENSE ./
COPY src ./src
COPY corpus ./corpus
RUN uv sync --locked --no-dev

# Run as an ordinary user, not root. data/ receives the vector index (a volume).
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app:app /app/data
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    CRA_RAG_OLLAMA_URL="http://ollama:11434"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "cra_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
