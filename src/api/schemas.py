"""Pydantic v2 data contract of the scoring API (Sprint 3, Card 1).

These schemas are the API's first line of defence: every request body is
validated here *before* it reaches ``CreditRiskPredictor`` or the database, so a
wrong type, an unknown category or an out-of-range value becomes a clear
``422 Unprocessable Entity`` instead of a crash inside the sklearn pipeline.

Exposes:
    ApplicationBase      -- the 13 raw LendingClub features (shared fields)
    ApplicationCreate    -- POST request body: strict input validation
    ApplicationResponse  -- response body: features + score + id + created_at
                            + ``decision``, readable straight from a
                            ``CreditApplication`` row
    HealthResponse       -- body of ``GET /health`` (liveness check)
    decide               -- the approve / reject rule behind ``decision``

Field names match ``src.db.models.MODEL_FEATURES`` and the ``CreditApplication``
columns exactly, so data moves between the API, the model and the database
without any renaming. Category levels match ``frontend/app.py`` (the training
data levels from notebook 02); ``tests/test_api_schemas.py`` checks both.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, computed_field

# --- Category levels the model was trained on ------------------------------------
# Same values and order (by frequency) as the *_OPTIONS tuples in frontend/app.py.
# A Literal rejects any other value, so the OneHotEncoder never sees an unknown level.
Term: TypeAlias = Literal["36 months", "60 months"]
HomeOwnership: TypeAlias = Literal["MORTGAGE", "RENT", "OWN", "OTHER"]
VerificationStatus: TypeAlias = Literal["Source Verified", "Verified", "Not Verified"]
Purpose: TypeAlias = Literal[
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "moving",
    "vacation",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
]

# --- Business rules (same as frontend/app.py) --------------------------------------
MIN_LOAN_AMOUNT: float = 500.0
MAX_LOAN_AMOUNT: float = 40_000.0

# --- Credit decision (Sprint 4, Card 3) --------------------------------------------
# Same rule and the same RISK_THRESHOLD setting as ``is_high_risk`` in
# frontend/app.py, so the API decision and the UI banner always agree.
# 0.5 is a placeholder until a tuned threshold is chosen in notebook 03; changing
# it needs evaluation metrics (see CONTRIBUTING.md).
Decision: TypeAlias = Literal["approved", "rejected"]
DEFAULT_RISK_THRESHOLD: Final[float] = 0.5


def parse_risk_threshold(raw: str | None) -> float:
    """Parse a ``RISK_THRESHOLD`` value. ``None`` or blank gives the default.

    Args:
        raw: Raw environment value, or None if unset.

    Returns:
        The threshold, strictly between 0 and 1.

    Raises:
        ValueError: If the value is not a number strictly between 0 and 1.
    """
    if raw is None or not raw.strip():
        return DEFAULT_RISK_THRESHOLD
    try:
        value: float = float(raw)
    except ValueError as exc:
        raise ValueError(f"RISK_THRESHOLD must be a number, got {raw!r}.") from exc
    if not 0.0 < value < 1.0:  # also rejects NaN
        raise ValueError(f"RISK_THRESHOLD must be between 0 and 1, got {value}.")
    return value


def get_risk_threshold() -> float:
    """Return the current decision threshold from the ``RISK_THRESHOLD`` setting.

    Read at call time, not at import time: this module is imported before
    ``src.db.database`` loads ``.env``, so an import-time read would miss it.
    The API calls it once at startup so a bad value stops the service early.
    """
    return parse_risk_threshold(os.getenv("RISK_THRESHOLD"))


def decide(default_probability: float, threshold: float | None = None) -> Decision:
    """Turn a probability of default into a credit decision.

    Args:
        default_probability: Model output in [0, 1].
        threshold: Probability at or above which the application is rejected.
            ``None`` uses ``get_risk_threshold()``.

    Returns:
        ``"rejected"`` if the probability is at or above the threshold,
        otherwise ``"approved"``.
    """
    limit: float = get_risk_threshold() if threshold is None else threshold
    return "rejected" if default_probability >= limit else "approved"


_EXAMPLE_APPLICATION: dict[str, float | int | str | None] = {
    "loan_amnt": 10_000.0,
    "term": "36 months",
    "purpose": "debt_consolidation",
    "annual_inc": 60_000.0,
    "dti": 18.0,
    "emp_length": 5,
    "home_ownership": "MORTGAGE",
    "verification_status": "Source Verified",
    "fico_range_low": 700,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_util": 50.0,
    "mort_acc": 1,
}


class ApplicationBase(BaseModel):
    """The 13 raw model features, with types and value ranges.

    Shared by the request and the response, so the two can never drift apart.
    Not used directly as an endpoint schema.
    """

    # NaN / inf pass a plain ``float`` check but would poison the model's output.
    model_config = ConfigDict(allow_inf_nan=False)

    # --- Loan terms ---
    loan_amnt: float = Field(
        ge=MIN_LOAN_AMOUNT, le=MAX_LOAN_AMOUNT, description="Requested amount, USD"
    )
    term: Term = Field(description="Loan term")
    purpose: Purpose = Field(description="Stated loan purpose")

    # --- Financial profile and employment ---
    annual_inc: float = Field(gt=0, description="Self-reported annual income, USD")
    dti: float = Field(ge=0, description="Debt-to-income ratio, %")
    emp_length: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Years employed, 0-10 (10 = 10+). null = unknown",
    )
    home_ownership: HomeOwnership = Field(description="Home ownership status")
    verification_status: VerificationStatus = Field(
        description="Income verification status"
    )

    # --- Credit history at application time ---
    fico_range_low: int = Field(ge=300, le=850, description="Lower FICO bound")
    open_acc: int = Field(ge=0, description="Number of open credit lines")
    pub_rec: int = Field(ge=0, description="Number of derogatory public records")
    revol_util: float = Field(
        ge=0, description="Revolving line utilisation, % (can exceed 100)"
    )
    mort_acc: int | None = Field(
        default=None, ge=0, description="Number of mortgage accounts. null = unknown"
    )


class ApplicationCreate(ApplicationBase):
    """Request body of ``POST /applications``: one application to score.

    Strict mode: no silent type coercion. ``"10000"`` for a number, ``700.5``
    for an integer or ``true`` for a count are rejected instead of being
    converted, so a frontend bug shows up as a 422, not as a wrong score.
    An int is still accepted for a float field (``10000`` -> ``10000.0``).
    """

    model_config = ConfigDict(
        strict=True,
        extra="forbid",  # a misspelled field name must fail, not be silently dropped
        frozen=True,  # the request is a fact: nobody should mutate it downstream
        json_schema_extra={"examples": [_EXAMPLE_APPLICATION]},
    )


class ApplicationResponse(ApplicationBase):
    """Response body: the stored application together with its score.

    ``from_attributes=True`` lets FastAPI build it straight from the ORM object
    returned by ``crud.create_application``:
    ``ApplicationResponse.model_validate(credit_application)``.

    ``decision`` is a computed field: it is derived from ``default_probability``
    on output and is not stored in the database, so it never appears in
    ``model_fields`` and the ORM row needs no extra column.
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    **_EXAMPLE_APPLICATION,
                    "id": 42,
                    "default_probability": 0.137,
                    "decision": "approved",
                    "created_at": "2026-10-19T09:15:00Z",
                }
            ]
        },
    )

    id: int = Field(description="Primary key of the stored application")
    default_probability: float = Field(
        ge=0, le=1, description="Predicted probability of default, in [0, 1]"
    )
    created_at: datetime = Field(description="When the application was received")

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Credit decision: 'rejected' if default_probability is at or above "
            "the RISK_THRESHOLD setting (default 0.5), otherwise 'approved'"
        )
    )
    @property
    def decision(self) -> Decision:
        """Approve or reject the application based on its default probability."""
        return decide(self.default_probability)


class HealthResponse(BaseModel):
    """Body of ``GET /health``: the service process is up and answering.

    A liveness check only: it touches neither the database nor the model, so
    it stays fast and cannot fail because a dependency is slow.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = Field(default="ok", description="Always 'ok' when up")
    version: str = Field(description="API version")
    model_loaded: bool = Field(
        description="Whether the scoring pipeline is loaded in this process"
    )
