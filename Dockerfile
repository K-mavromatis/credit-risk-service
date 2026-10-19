# =============================================================================
# Credit Risk API - production image (Sprint 5, Card 1)
#
# Two stages:
#   builder - has gcc + libpq-dev, compiles/installs Python deps into /opt/venv
#   runtime - slim image with only the runtime shared libraries + the venv + src/
#
# Build:  docker build -t credit-api .
# Run:    docker run --rm -p 8000:8000 --env-file .env credit-api
# =============================================================================

# ---------- Stage 1: builder -------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build toolchain for any package that compiles from source
# (psycopg2 needs gcc + libpq headers if a binary wheel is not available).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv: the whole dependency tree is copied to the runtime
# stage as one directory.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Layer caching: copy ONLY requirements.txt first. This layer (and the slow
# pip install below) is rebuilt only when requirements.txt changes, not on
# every source-code edit.
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------- Stage 2: runtime -------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Runtime shared libraries only (no compiler):
#   libpq5   - PostgreSQL client library used by psycopg2
#   libgomp1 - OpenMP runtime required by LightGBM (import fails without it)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user: the API never needs root.
RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code goes in LAST, so code changes invalidate only these layers.
# src/ includes src/ml_model/scoring_pipeline.pkl (loaded at startup);
# raw data, caches and secrets are excluded by .dockerignore.
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic/ ./alembic/
COPY --chown=app:app src/ ./src/

USER app

EXPOSE 8000

# Liveness probe against GET /health (touches neither the model nor the DB).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
