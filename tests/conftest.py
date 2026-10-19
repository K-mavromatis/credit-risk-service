"""Shared pytest configuration and fixtures.

DB fixtures (Sprint 4, Card 4):
    db_engine   -- one in-memory SQLite engine per test run; the schema is built
                   from ``Base.metadata`` directly, Alembic is never invoked.
    db_session  -- a Session per test, wrapped in a transaction that is rolled
                   back afterwards, so no row leaks from one test into another.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import ConnectionPoolEntry, StaticPool

# src.db.database builds its engine at import time and needs DATABASE_URL.
# This dummy value lets the ORM be imported in tests without a real database
# (create_engine never connects until first use). Because it is set before
# load_dotenv runs, a developer's real .env can never be hit by the test suite.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg2://test:test@localhost:5432/test_db"
)

# Importing the models module registers every table on Base.metadata, so
# create_all() below builds the full schema. Both imports need DATABASE_URL.
import src.db.models  # noqa: F401
from src.db.database import Base

TEST_DATABASE_URL = "sqlite:///:memory:"


def _enable_sqlite_savepoints(engine: Engine) -> None:
    """Make pysqlite honour BEGIN / SAVEPOINT / ROLLBACK as SQLAlchemy sends them.

    By default the sqlite3 driver delays BEGIN until the first INSERT, so a
    SAVEPOINT would open (and RELEASE would commit) the real transaction and the
    final rollback in ``db_session`` would undo nothing. This is the recipe from
    the SQLAlchemy SQLite dialect docs ("Serializable isolation / Savepoints").

    Args:
        engine: The SQLite engine to patch; must not have connected yet.
    """

    @event.listens_for(engine, "connect")
    def _disable_driver_begin(
        dbapi_connection: sqlite3.Connection, _record: ConnectionPoolEntry
    ) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _emit_begin(connection: Connection) -> None:
        connection.exec_driver_sql("BEGIN")


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, None, None]:
    """In-memory SQLite engine with all ORM tables, shared by the test run."""
    engine = create_engine(
        TEST_DATABASE_URL,
        # One shared connection: every new connection to ":memory:" would
        # otherwise get its own empty database without our tables.
        poolclass=StaticPool,
        # Lets FastAPI's TestClient (worker threads) reuse this engine later.
        connect_args={"check_same_thread": False},
    )
    _enable_sqlite_savepoints(engine)
    Base.metadata.create_all(bind=engine)  # no Alembic: straight from the models
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Session whose changes are always rolled back after the test.

    The outer transaction is opened on the connection here. The Session joins
    it with ``create_savepoint``, so ``db.commit()`` inside the CRUD functions
    only releases a SAVEPOINT; nothing becomes permanent, and the rollback in
    teardown leaves the tables empty for the next test.

    Yields:
        A ``Session`` configured like ``SessionLocal`` (``expire_on_commit=False``).
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(
        bind=connection,
        join_transaction_mode="create_savepoint",
        autoflush=False,
        expire_on_commit=False,
    )
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        transaction.rollback()
        connection.close()
