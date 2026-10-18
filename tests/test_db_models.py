"""Schema tests for src.db.models; they never touch PostgreSQL."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime

import pytest
from sqlalchemy import Table, create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from src.db.models import MODEL_FEATURES, CreditApplication, FeatureValue

TABLE: Table = CreditApplication.__table__  # type: ignore[assignment]

VALID_FEATURES: dict[str, FeatureValue] = {
    "loan_amnt": 12_000.0,
    "term": "36 months",
    "purpose": "debt_consolidation",
    "annual_inc": 65_000.0,
    "dti": 18.5,
    "emp_length": 4,
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "fico_range_low": 690,
    "open_acc": 9,
    "pub_rec": 0,
    "revol_util": 47.3,
    "mort_acc": None,
}


@pytest.fixture
def session() -> Generator[Session, None, None]:
    """In-memory SQLite session holding only the credit_applications table."""
    engine = create_engine("sqlite://")
    TABLE.create(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def test_model_features_match_feature_whitelist() -> None:
    assert len(MODEL_FEATURES) == 13
    assert set(VALID_FEATURES) == set(MODEL_FEATURES)


def test_table_has_exactly_the_expected_columns() -> None:
    expected = {"id", *MODEL_FEATURES, "default_probability", "created_at"}
    assert set(TABLE.columns.keys()) == expected
    assert [col.name for col in TABLE.primary_key.columns] == ["id"]


def test_only_informative_missing_features_are_nullable() -> None:
    nullable = {col.name for col in TABLE.columns if col.nullable}
    assert nullable == {"emp_length", "mort_acc", "default_probability"}


def test_postgres_ddl_compiles() -> None:
    ddl = str(CreateTable(TABLE).compile(dialect=postgresql.dialect()))
    assert "created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL" in ddl
    assert "CONSTRAINT pk_credit_applications PRIMARY KEY (id)" in ddl
    assert "ck_credit_applications_default_probability_range" in ddl


def test_insert_sets_id_and_created_at(session: Session) -> None:
    application = CreditApplication(**VALID_FEATURES, default_probability=0.12)
    session.add(application)
    session.commit()

    assert application.id is not None
    assert isinstance(application.created_at, datetime)
    assert application.to_features() == VALID_FEATURES


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_outside_unit_interval_is_rejected(
    session: Session, probability: float
) -> None:
    session.add(CreditApplication(**VALID_FEATURES, default_probability=probability))
    with pytest.raises(IntegrityError):
        session.commit()


def test_unscored_application_is_allowed(session: Session) -> None:
    session.add(CreditApplication(**VALID_FEATURES))
    session.commit()
