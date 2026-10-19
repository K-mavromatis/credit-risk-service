"""Tests for the API data contract (src/api/schemas.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from frontend.app import (
    HOME_OWNERSHIP_OPTIONS,
    MAX_LOAN_AMOUNT,
    MIN_LOAN_AMOUNT,
    PURPOSE_OPTIONS,
    TERM_OPTIONS,
    VERIFICATION_STATUS_OPTIONS,
)
from src.api import schemas
from src.api.schemas import (
    ApplicationBase,
    ApplicationCreate,
    ApplicationResponse,
)
from src.db.models import MODEL_FEATURES, CreditApplication


def _payload(**overrides: Any) -> dict[str, Any]:
    """A valid request body, with optional field overrides."""
    payload: dict[str, Any] = {
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
    payload.update(overrides)
    return payload


# --- Contract stays in sync with the model, the DB and the frontend --------------
def test_feature_fields_match_model_features() -> None:
    assert set(ApplicationBase.model_fields) == set(MODEL_FEATURES)


def test_response_fields_are_orm_columns() -> None:
    columns = set(CreditApplication.__table__.columns.keys())
    assert set(ApplicationResponse.model_fields) <= columns


def test_category_levels_match_frontend() -> None:
    assert get_args(schemas.Term) == TERM_OPTIONS
    assert get_args(schemas.HomeOwnership) == HOME_OWNERSHIP_OPTIONS
    assert get_args(schemas.VerificationStatus) == VERIFICATION_STATUS_OPTIONS
    assert get_args(schemas.Purpose) == PURPOSE_OPTIONS
    assert (schemas.MIN_LOAN_AMOUNT, schemas.MAX_LOAN_AMOUNT) == (
        MIN_LOAN_AMOUNT,
        MAX_LOAN_AMOUNT,
    )


# --- ApplicationCreate: valid input ------------------------------------------------
def test_valid_payload_is_accepted() -> None:
    application = ApplicationCreate.model_validate(_payload())
    assert application.model_dump() == _payload()


def test_unknown_values_may_be_omitted_and_default_to_none() -> None:
    payload = _payload()
    del payload["emp_length"], payload["mort_acc"]
    application = ApplicationCreate.model_validate(payload)
    assert application.emp_length is None
    assert application.mort_acc is None


def test_int_is_accepted_for_float_field() -> None:
    application = ApplicationCreate.model_validate(_payload(loan_amnt=10_000))
    assert application.loan_amnt == 10_000.0
    assert isinstance(application.loan_amnt, float)


def test_json_body_is_accepted() -> None:
    body = json.dumps(
        _payload(loan_amnt=5_000, term="60 months", emp_length=None, mort_acc=None)
    )
    application = ApplicationCreate.model_validate_json(body)
    assert application.loan_amnt == 5_000.0
    assert application.term == "60 months"
    assert application.emp_length is None


def test_schema_examples_are_valid() -> None:
    example = ApplicationCreate.model_json_schema()["examples"][0]
    ApplicationCreate.model_validate(example)


# --- ApplicationCreate: rejected input ---------------------------------------------
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loan_amnt", "10000"),  # number sent as text
        ("annual_inc", "sixty thousand"),
        ("fico_range_low", "700"),
        ("fico_range_low", 700.5),  # float where an int is expected
        ("open_acc", True),  # bool is not a count
        ("term", 36),
    ],
)
def test_wrong_types_are_rejected_not_coerced(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ApplicationCreate.model_validate(_payload(**{field: value}))
    assert exc_info.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("term", "48 months"),
        ("purpose", "Debt Consolidation"),
        ("home_ownership", "NONE"),
        ("verification_status", "verified"),
    ],
)
def test_unknown_categories_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        ApplicationCreate.model_validate(_payload(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loan_amnt", 499.0),
        ("loan_amnt", 40_001.0),
        ("annual_inc", 0.0),
        ("dti", -1.0),
        ("emp_length", 11),
        ("fico_range_low", 299),
        ("fico_range_low", 851),
        ("open_acc", -1),
        ("pub_rec", -1),
        ("revol_util", -0.1),
        ("mort_acc", -1),
        ("annual_inc", float("nan")),
        ("dti", float("inf")),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ApplicationCreate.model_validate(_payload(**{field: value}))


@pytest.mark.parametrize("field", ["loan_amnt", "term", "fico_range_low"])
def test_required_fields_are_required(field: str) -> None:
    payload = _payload()
    del payload[field]
    with pytest.raises(ValidationError):
        ApplicationCreate.model_validate(payload)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ApplicationCreate.model_validate(_payload(annual_income=60_000.0))


def test_request_is_immutable() -> None:
    application = ApplicationCreate.model_validate(_payload())
    with pytest.raises(ValidationError):
        application.loan_amnt = 1.0  # type: ignore[misc]


# --- ApplicationResponse -----------------------------------------------------------
def test_response_is_built_from_orm_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISK_THRESHOLD", "0.5")  # a local .env must not change it
    created_at = datetime(2026, 10, 19, 9, 15, tzinfo=timezone.utc)
    row = CreditApplication(
        **_payload(), id=42, default_probability=0.137, created_at=created_at
    )

    response = ApplicationResponse.model_validate(row)

    assert response.id == 42
    assert response.default_probability == pytest.approx(0.137)
    assert response.created_at == created_at
    assert response.decision == "approved"  # computed: 0.137 < 0.5, not stored
    features = response.model_dump(
        exclude={"id", "default_probability", "created_at", "decision"}
    )
    assert features == row.to_features()


def test_response_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ApplicationResponse.model_validate(
            _payload(id=1, default_probability=1.5, created_at=datetime.now())
        )
