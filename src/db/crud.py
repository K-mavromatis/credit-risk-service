"""CRUD operations for scored credit applications (Sprint 2, Card 4).

This module is the only place where the service runs INSERT / SELECT statements
against ``credit_applications``. FastAPI endpoints call these functions with the
session from ``src.db.database.get_db``; they never build queries themselves.

Exposes:
    ApplicationCreate        -- validated input for one scored application
    create_application       -- INSERT one row and return the stored ORM object
    get_recent_applications  -- SELECT the newest rows for the analytics dashboard
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.db.models import CreditApplication


class ApplicationCreate(BaseModel):
    """Client features plus the model's score, as they will be stored.

    Field names match the ``CreditApplication`` columns exactly, so the API layer
    can build this from its request schema and the ``predict_proba`` result.
    The constraints mirror the table: CHECK constraints, VARCHAR lengths and the
    nullable columns, so bad input fails here with a clear Pydantic error instead
    of a database error in the middle of a transaction.
    """

    model_config = ConfigDict(
        extra="forbid",  # a typo in a field name must fail, not be dropped
        frozen=True,
        allow_inf_nan=False,  # NaN/inf are never valid features or scores
        str_strip_whitespace=True,
    )

    # --- Loan terms ---
    loan_amnt: float = Field(gt=0)
    term: str = Field(min_length=1, max_length=16)
    purpose: str = Field(min_length=1, max_length=32)

    # --- Financial profile and employment ---
    annual_inc: float = Field(gt=0)
    dti: float
    emp_length: int | None = Field(default=None, ge=0, le=10)
    home_ownership: str = Field(min_length=1, max_length=16)
    verification_status: str = Field(min_length=1, max_length=32)

    # --- Credit history ---
    fico_range_low: int = Field(ge=300, le=850)
    open_acc: int = Field(ge=0)
    pub_rec: int = Field(ge=0)
    revol_util: float = Field(ge=0)
    mort_acc: int | None = Field(default=None, ge=0)

    # --- Model output ---
    # Required, but may be None explicitly for an application not scored yet.
    default_probability: float | None = Field(ge=0, le=1)


def create_application(db: Session, data: ApplicationCreate) -> CreditApplication:
    """Insert one scored application and return it with ``id`` and ``created_at``.

    The function commits its own transaction: one call = one stored application.
    On any database error the session is rolled back (so it stays usable) and
    the error is re-raised for the API layer to turn into an HTTP response.
    """
    application = CreditApplication(**data.model_dump())
    db.add(application)
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    # created_at is filled by PostgreSQL (server_default), so reload the row to
    # have it available even after the request's session is closed.
    db.refresh(application)
    return application


def get_recent_applications(db: Session, limit: int = 10) -> list[CreditApplication]:
    """Return the ``limit`` most recent applications, newest first.

    ``id`` breaks ties: rows inserted in the same transaction share the same
    ``now()`` timestamp, and the order of the dashboard must be stable.
    """
    if limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")

    statement = (
        select(CreditApplication)
        .order_by(CreditApplication.created_at.desc(), CreditApplication.id.desc())
        .limit(limit)
    )
    rows: Sequence[CreditApplication] = db.scalars(statement).all()
    return list(rows)
