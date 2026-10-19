"""Isolated tests for the DB layer (Sprint 4, Card 4).

Every test uses the ``db_session`` fixture from ``tests/conftest.py``: an
in-memory SQLite database built from ``Base.metadata`` (no Alembic, never
PostgreSQL), wrapped in a transaction that is rolled back after each test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Final

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.crud import ApplicationCreate, create_application, get_recent_applications
from src.db.models import MODEL_FEATURES, CreditApplication

CLIENT_FEATURES: Final[dict[str, Any]] = {
    "loan_amnt": 15_000.0,
    "term": "60 months",
    "purpose": "credit_card",
    "annual_inc": 82_000.0,
    "dti": 21.4,
    "emp_length": 7,
    "home_ownership": "MORTGAGE",
    "verification_status": "Source Verified",
    "fico_range_low": 715,
    "open_acc": 11,
    "pub_rec": 0,
    "revol_util": 38.9,
    "mort_acc": 2,
}
SCORE: Final[float] = 0.27

# SQLite drops tzinfo, so the test timestamps are naive to compare exactly.
BASE_TIME: Final[datetime] = datetime(2026, 10, 1, 12, 0, 0)
# Minutes after BASE_TIME for rows inserted in this order (ids 1..5). The time
# order deliberately differs from the insertion order, so sorting by id instead
# of created_at would fail the ordering tests.
CREATED_AT_OFFSETS_MIN: Final[tuple[int, ...]] = (30, 10, 50, 20, 40)


def make_application(**overrides: Any) -> ApplicationCreate:
    return ApplicationCreate(
        **{**CLIENT_FEATURES, "default_probability": SCORE, **overrides}
    )


def count_applications(session: Session) -> int:
    total = session.scalar(select(func.count()).select_from(CreditApplication))
    return total or 0


# --- create_application -------------------------------------------------------


def test_create_application_saves_features_and_score(db_session: Session) -> None:
    application = create_application(db_session, make_application())

    db_session.expire_all()  # read back from the database, not the identity map
    stored = db_session.get(CreditApplication, application.id)

    assert stored is not None
    assert stored.to_features() == {
        name: CLIENT_FEATURES[name] for name in MODEL_FEATURES
    }
    assert stored.default_probability == pytest.approx(SCORE)
    assert isinstance(stored.created_at, datetime)
    assert count_applications(db_session) == 1


def test_create_application_assigns_autoincrement_id(db_session: Session) -> None:
    applications = [
        create_application(db_session, make_application(loan_amnt=amount))
        for amount in (5_000.0, 7_500.0, 10_000.0)
    ]

    ids = [application.id for application in applications]
    assert all(isinstance(id_, int) and id_ >= 1 for id_ in ids)
    assert ids == list(range(ids[0], ids[0] + len(ids)))  # consecutive, increasing


# --- get_recent_applications --------------------------------------------------


@pytest.fixture
def seeded_applications(db_session: Session) -> list[CreditApplication]:
    """Five stored applications with distinct, out-of-order ``created_at``."""
    applications: list[CreditApplication] = []
    for index, minutes in enumerate(CREATED_AT_OFFSETS_MIN):
        application = create_application(
            db_session, make_application(loan_amnt=1_000.0 * (index + 1))
        )
        # created_at is a server default (one value per second on SQLite), so
        # set explicit timestamps to make the expected order unambiguous.
        application.created_at = BASE_TIME + timedelta(minutes=minutes)
        applications.append(application)
    db_session.commit()
    db_session.expire_all()
    return applications


def expected_newest_first(applications: list[CreditApplication]) -> list[int]:
    """Ids ordered by the known created_at offsets, newest first."""
    order = sorted(
        range(len(applications)),
        key=lambda i: CREATED_AT_OFFSETS_MIN[i],
        reverse=True,
    )
    return [applications[i].id for i in order]


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_get_recent_applications_respects_limit_and_order(
    db_session: Session, seeded_applications: list[CreditApplication], limit: int
) -> None:
    recent = get_recent_applications(db_session, limit=limit)

    assert len(recent) == limit
    assert [app.id for app in recent] == expected_newest_first(seeded_applications)[
        :limit
    ]
    timestamps = [app.created_at for app in recent]
    assert timestamps == sorted(timestamps, reverse=True)
    assert recent[0].created_at == BASE_TIME + timedelta(
        minutes=max(CREATED_AT_OFFSETS_MIN)
    )


def test_get_recent_applications_limit_above_row_count_returns_all(
    db_session: Session, seeded_applications: list[CreditApplication]
) -> None:
    recent = get_recent_applications(db_session, limit=100)

    assert [app.id for app in recent] == expected_newest_first(seeded_applications)


# --- fixture isolation --------------------------------------------------------


@pytest.mark.parametrize("run", [1, 2])
def test_each_test_starts_with_an_empty_table(db_session: Session, run: int) -> None:
    # Both runs insert and commit a row; the second would see two rows if the
    # rollback in db_session did not undo the first run's commit.
    assert count_applications(db_session) == 0
    create_application(db_session, make_application())
    assert count_applications(db_session) == 1
