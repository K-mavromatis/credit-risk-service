"""Alembic migration environment for the credit scoring service.

The database URL is never stored in ``alembic.ini``. It comes from the same
validated ``DatabaseSettings`` the application uses (``DATABASE_URL`` in the
project-root ``.env``), so the app and the migrations target one database.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import MetaData, create_engine, pool
from sqlalchemy.engine import Connection, Engine

from alembic import context
from alembic.config import Config
from src.db.database import settings

# Import Base from *models*, not from database: importing models.py registers
# every ORM class (CreditApplication, ...) on Base.metadata. Without it,
# autogenerate sees an empty schema and produces an empty migration.
from src.db.models import Base

config: Config = context.config

# Set up Python logging from the [loggers] section of alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate compares this metadata against the live database.
target_metadata: MetaData = Base.metadata

DATABASE_URL: str = str(settings.database_url)


def run_migrations_offline() -> None:
    """Render migrations as SQL without a DB connection (``--sql`` mode)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure the migration context on an open connection and run it."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # also detect column type changes, e.g. String(16)->(32)
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to PostgreSQL and apply migrations inside a transaction."""
    # NullPool: a short-lived CLI process shouldn't keep pooled connections,
    # so we don't reuse the API's pooled engine from src.db.database.
    connectable: Engine = create_engine(DATABASE_URL, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        do_run_migrations(connection)
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
