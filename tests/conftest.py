"""Shared pytest configuration."""

import os

# src.db.database builds its engine at import time and needs DATABASE_URL.
# This dummy value lets the ORM be imported in tests without a real database
# (create_engine never connects until first use). Because it is set before
# load_dotenv runs, a developer's real .env can never be hit by the test suite.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg2://test:test@localhost:5432/test_db"
)
