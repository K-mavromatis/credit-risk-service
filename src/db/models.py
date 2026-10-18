"""ORM models for the credit scoring service (Sprint 2, Card 2).

``CreditApplication`` stores one scored loan application: the 13 raw
LendingClub features the scoring pipeline consumes, the model's predicted
probability of default, and the time the application was received.

Column names match the pipeline's feature names exactly (see
``notebooks/01_data_preparation.ipynb`` and ``frontend/app.py``), so a stored
row can be fed back into ``CreditRiskPredictor.predict_proba`` without any
renaming, e.g. for re-scoring with a new model or monitoring drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base

# The exact input columns of the trained scoring pipeline, in training order.
NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "loan_amnt",
    "emp_length",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "pub_rec",
    "revol_util",
    "mort_acc",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
)
MODEL_FEATURES: Final[tuple[str, ...]] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

FeatureValue = float | int | str | None


class CreditApplication(Base):
    """One loan application together with the model's default prediction."""

    __tablename__ = "credit_applications"
    __table_args__ = (
        # NULL passes a CHECK, so an application that is not scored yet is allowed.
        CheckConstraint(
            "default_probability >= 0 AND default_probability <= 1",
            name="default_probability_range",
        ),
        CheckConstraint("loan_amnt > 0", name="loan_amnt_positive"),
        CheckConstraint("annual_inc > 0", name="annual_inc_positive"),
    )

    # --- Identity ---------------------------------------------------------------
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # --- Loan terms -------------------------------------------------------------
    loan_amnt: Mapped[float] = mapped_column(Float, comment="Requested amount, USD")
    term: Mapped[str] = mapped_column(String(16), comment="'36 months'/'60 months'")
    purpose: Mapped[str] = mapped_column(String(32), comment="Stated loan purpose")

    # --- Financial profile and employment ---------------------------------------
    annual_inc: Mapped[float] = mapped_column(Float, comment="Annual income, USD")
    dti: Mapped[float] = mapped_column(Float, comment="Debt-to-income ratio, %")
    emp_length: Mapped[int | None] = mapped_column(
        Integer, comment="Years employed, 0-10 (10 = 10+); NULL = unknown"
    )
    home_ownership: Mapped[str] = mapped_column(
        String(16), comment="MORTGAGE / RENT / OWN / OTHER"
    )
    verification_status: Mapped[str] = mapped_column(
        String(32), comment="Income verification status"
    )

    # --- Credit history at application time -------------------------------------
    fico_range_low: Mapped[int] = mapped_column(Integer, comment="Lower FICO bound")
    open_acc: Mapped[int] = mapped_column(Integer, comment="Open credit lines")
    pub_rec: Mapped[int] = mapped_column(Integer, comment="Derogatory public records")
    revol_util: Mapped[float] = mapped_column(
        Float, comment="Revolving utilisation, % (can exceed 100)"
    )
    mort_acc: Mapped[int | None] = mapped_column(
        Integer, comment="Mortgage accounts; NULL = unknown"
    )

    # --- Model output -----------------------------------------------------------
    default_probability: Mapped[float | None] = mapped_column(
        Float, comment="Predicted P(default) in [0, 1]; NULL = not scored yet"
    )

    # --- Audit ------------------------------------------------------------------
    # server_default: PostgreSQL fills the timestamp itself, in UTC-aware form,
    # even for rows inserted outside the ORM. Indexed for time-window analytics.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        comment="When the application was received",
    )

    def to_features(self) -> dict[str, FeatureValue]:
        """Return the raw model inputs, ready for ``CreditRiskPredictor``."""
        return {name: getattr(self, name) for name in MODEL_FEATURES}

    def __repr__(self) -> str:
        # Deliberately no applicant data here: reprs end up in logs.
        return (
            f"CreditApplication(id={self.id!r}, "
            f"default_probability={self.default_probability!r}, "
            f"created_at={self.created_at!r})"
        )
