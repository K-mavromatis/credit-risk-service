"""Tests for the FastAPI app and the stub POST /predict endpoint (src/api/main.py)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.main import MOCK_DEFAULT_PROBABILITY, app
from src.api.schemas import ApplicationBase, ApplicationResponse


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


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


def test_app_title() -> None:
    assert app.title == "Credit Scoring API"


def test_predict_returns_valid_response(client: TestClient) -> None:
    response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    body = ApplicationResponse.model_validate(response.json())
    assert body.default_probability == MOCK_DEFAULT_PROBABILITY
    assert 0.0 <= body.default_probability <= 1.0


def test_predict_echoes_input_features(client: TestClient) -> None:
    payload = _payload(emp_length=None, mort_acc=None)
    body = client.post("/predict", json=payload).json()

    for name in ApplicationBase.model_fields:
        assert body[name] == payload[name]


@pytest.mark.parametrize(
    "overrides",
    [
        {"purpose": "crypto"},  # unknown category
        {"loan_amnt": 100.0},  # below MIN_LOAN_AMOUNT
        {"fico_range_low": "700"},  # strict mode: no str -> int coercion
        {"annual_inc": 0},  # must be > 0
    ],
)
def test_predict_rejects_invalid_values(
    client: TestClient, overrides: dict[str, Any]
) -> None:
    assert client.post("/predict", json=_payload(**overrides)).status_code == 422


def test_predict_rejects_unknown_field(client: TestClient) -> None:
    response = client.post("/predict", json=_payload(loan_amount=10_000.0))
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = _payload()
    del payload["term"]
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_is_in_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/predict"]
