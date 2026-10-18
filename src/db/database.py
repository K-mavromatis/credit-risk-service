"""PostgreSQL connection setup for the credit scoring service.

Exposes:
    engine        -- the SQLAlchemy Engine (one per process, holds the pool)
    SessionLocal  -- factory that creates new ORM Session objects
    Base          -- declarative base class every ORM model inherits from
    get_db        -- FastAPI dependency that yields a session per request

The connection URI is read from the ``DATABASE_URL`` variable in the project
root ``.env`` file (or the real environment), so no credentials live in code.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# src/db/database.py -> parents[2] is the repository root (where .env lives).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class DatabaseSettings(BaseSettings):
    """Database configuration loaded from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # other keys in .env (API, model paths...) are fine
    )

    # repr=False keeps the password out of logs and tracebacks.
    database_url: PostgresDsn = Field(repr=False)
    db_echo: bool = False  # True -> log every SQL statement (debug only)
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)


settings: DatabaseSettings = DatabaseSettings()

# The engine manages the connection pool. It does not connect until first use.
engine: Engine = create_engine(
    str(settings.database_url),
    echo=settings.db_echo,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,  # drop dead connections (e.g. after a DB restart)
)

# Each call to SessionLocal() opens a new, independent unit of work.
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)

# Predictable constraint names make Alembic migrations stable and readable.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base class for all ORM models (SQLAlchemy 2.0 style)."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and always close it afterwards.

    Usage in FastAPI::

        @router.post("/score")
        def score(..., db: Session = Depends(get_db)) -> ...:
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection() -> bool:
    """Return True if PostgreSQL answers a trivial ``SELECT 1`` query."""
    with engine.connect() as connection:
        result: int = connection.execute(text("SELECT 1")).scalar_one()
        return result == 1


if __name__ == "__main__":
    # Smoke test:  python src/db/database.py
    print(f"Connecting to: {engine.url!r}")  # SQLAlchemy masks the password
    print("Connection OK" if check_connection() else "Connection FAILED")
