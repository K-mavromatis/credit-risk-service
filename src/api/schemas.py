"""Pydantic v2 data contract of the scoring API (Sprint 3, Card 1).

These schemas are the API's first line of defence: every request body is
validated here *before* it reaches ``CreditRiskPredictor`` or the database, so a
wrong type, an unknown category or an out-of-range value becomes a clear
``422 Unprocessable Entity`` instead of a crash inside the sklearn pipeline.

Exposes:
    ApplicationBase      -- the 13 raw LendingClub features (shared fields)
    ApplicationCreate    -- POST request body: strict input validation
    ApplicationResponse  -- response body: features + score + id + created_at,
                            readable straight from a ``CreditApplication`` row

Field names match ``src.db.models.MODEL_FEATURES`` and the ``CreditApplication``
columns exactly, so data moves between the API, the model and the database
without any renaming. Category levels match ``frontend/app.py`` (the training
data levels from notebook 02); ``tests/test_api_schemas.py`` checks both.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

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
    """

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    **_EXAMPLE_APPLICATION,
                    "id": 42,
                    "default_probability": 0.137,
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
