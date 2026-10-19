"""Tests for src.db.crud; they run on in-memory SQLite, never on PostgreSQL."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import Table, create_engine, func, select
from sqlalchemy.orm import Session

from src.db.crud import ApplicationCreate, create_application, get_recent_applications
from src.db.models import MODEL_FEATURES, CreditApplication

TABLE: Table = CreditApplication.__table__  # type: ignore[assignment]

VALID_PAYLOAD: dict[str, Any] = {
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
    "default_probability": 0.12,
}


@pytest.fixture
def session() -> Generator[Session, None, None]:
    """In-memory SQLite session holding only the credit_applications table."""
    engine = create_engine("sqlite://")
    TABLE.create(engine)
    with Session(engine, expire_on_commit=False) as db:
        yield db
    engine.dispose()


def make_payload(**overrides: Any) -> ApplicationCreate:
    return ApplicationCreate(**{**VALID_PAYLOAD, **overrides})


def test_schema_fields_match_table_columns() -> None:
    expected = {*MODEL_FEATURES, "default_probability"}
    assert set(ApplicationCreate.model_fields) == expected


def test_create_application_persists_row(session: Session) -> None:
    application = create_application(session, make_payload())

    assert application.id is not None
    assert application.created_at is not None
    assert application.default_probability == pytest.approx(0.12)
    stored = session.get(CreditApplication, application.id)
    assert stored is not None
    assert stored.to_features() == {
        name: VALID_PAYLOAD[name] for name in MODEL_FEATURES
    }


def test_create_application_allows_unscored(session: Session) -> None:
    application = create_application(session, make_payload(default_probability=None))
    assert application.default_probability is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_probability": 1.5},
        {"loan_amnt": 0},
        {"annual_inc": -1},
        {"emp_length": 11},
        {"dti": float("nan")},
        {"term": "x" * 17},
        {"unknown_field": 1},
    ],
)
def test_invalid_payload_is_rejected_before_db(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_payload(**overrides)


def test_default_probability_is_required() -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "default_probability"}
    with pytest.raises(ValidationError):
        ApplicationCreate(**payload)


def test_get_recent_applications_returns_newest_first(session: Session) -> None:
    created = [
        create_application(session, make_payload(loan_amnt=1_000.0 * (i + 1)))
        for i in range(5)
    ]

    recent = get_recent_applications(session, limit=3)

    assert [app.id for app in recent] == [app.id for app in reversed(created)][:3]


def test_get_recent_applications_default_limit(session: Session) -> None:
    for _ in range(12):
        create_application(session, make_payload())

    assert len(get_recent_applications(session)) == 10
    total = session.scalar(select(func.count()).select_from(CreditApplication))
    assert total == 12


def test_get_recent_applications_empty_table(session: Session) -> None:
    assert get_recent_applications(session) == []


@pytest.mark.parametrize("limit", [0, -5])
def test_get_recent_applications_rejects_bad_limit(
    session: Session, limit: int
) -> None:
    with pytest.raises(ValueError):
        get_recent_applications(session, limit=limit)
