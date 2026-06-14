FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY api ./api
COPY agents ./agents
COPY shared ./shared
COPY frontend ./frontend

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]
