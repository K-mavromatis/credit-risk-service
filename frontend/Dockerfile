# =============================================================================
# Credit Risk UI - Streamlit image (Sprint 5, Card 2)
#
# Build context is ./frontend (see docker-compose.yml): app.py is
# self-contained and talks to the API over HTTP via API_BASE_URL.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Unprivileged user; Streamlit writes a small ~/.streamlit, so it gets a home.
RUN groupadd --system app \
    && useradd --system --gid app --create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app.py .

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
