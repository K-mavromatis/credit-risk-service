"""Tests for the scoring API client of the Streamlit UI (frontend/app.py).

No server is started: ``requests.post`` is replaced by a recorder that returns
a prepared response or raises a prepared exception, so the tests are fast and
deterministic. The contract tests at the bottom check the frontend against the
real Pydantic schemas of the API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
import requests

from frontend import app
from frontend.app import (
    DEFAULT_RISK_THRESHOLD,
    FEATURE_ORDER,
    SERVICE_UNAVAILABLE_MESSAGE,
    FeatureValue,
    ScoringAPIError,
    ScoringResponseError,
    ScoringResult,
    ScoringServiceUnavailableError,
    build_payload,
    format_validation_errors,
    is_high_risk,
    parse_risk_threshold,
    request_score,
)
from src.api.schemas import ApplicationCreate, ApplicationResponse

# Synthetic applicant; value types match what the Streamlit widgets return.
FINANCIAL_INPUTS: dict[str, float | int | None] = {
    "loan_amnt": 10_000.0,
    "annual_inc": 60_000.0,
    "dti": 18.0,
    "emp_length": None,  # "Unknown" in the form
    "fico_range_low": 700,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_util": 50.0,
    "mort_acc": 1,
}
PROFILE_INPUTS: dict[str, str] = {
    "term": "36 months",
    "home_ownership": "MORTGAGE",
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
}
VALID_PAYLOAD: dict[str, FeatureValue] = build_payload(FINANCIAL_INPUTS, PROFILE_INPUTS)
API_RESPONSE: dict[str, object] = {
    **VALID_PAYLOAD,
    "id": 42,
    "default_probability": 0.137,
    "created_at": "2026-10-19T09:15:00Z",
}


def make_response(
    status_code: int, body: object = None, raw_text: str | None = None
) -> requests.Response:
    """Build a real ``requests.Response`` with a JSON (or raw text) body."""
    response = requests.Response()
    response.status_code = status_code
    text: str = raw_text if raw_text is not None else json.dumps(body)
    response._content = text.encode()
    response.headers["Content-Type"] = (
        "text/plain" if raw_text is not None else "application/json"
    )
    return response


@dataclass
class PostRecorder:
    """Stand-in for ``requests.post``: records calls, returns or raises."""

    response: requests.Response | None = None
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None, "PostRecorder needs a response or error"
        return self.response


@pytest.fixture
def fake_post(monkeypatch: pytest.MonkeyPatch) -> PostRecorder:
    recorder = PostRecorder()
    monkeypatch.setattr(app.requests, "post", recorder)
    return recorder


# --- Successful request ---------------------------------------------------------
def test_request_score_posts_payload_as_json_with_timeout(
    fake_post: PostRecorder,
) -> None:
    fake_post.response = make_response(200, API_RESPONSE)

    request_score(VALID_PAYLOAD)

    assert len(fake_post.calls) == 1
    call = fake_post.calls[0]
    assert call["url"] == app.PREDICT_URL
    assert call["url"].endswith("/predict")
    assert call["json"] == VALID_PAYLOAD
    assert call["timeout"] == app.REQUEST_TIMEOUT  # never wait forever


def test_request_score_returns_parsed_result(fake_post: PostRecorder) -> None:
    fake_post.response = make_response(200, API_RESPONSE)

    result = request_score(VALID_PAYLOAD)

    assert result.id == 42
    assert result.default_probability == pytest.approx(0.137)
    assert result.created_at.tzinfo is not None


# --- Network failures -----------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "expected_detail"),
    [
        (requests.exceptions.ConnectionError("refused"), "Cannot connect"),
        (requests.exceptions.ConnectTimeout("slow connect"), "Cannot connect"),
        (requests.exceptions.ReadTimeout("slow model"), "did not answer"),
        (requests.exceptions.Timeout("generic timeout"), "did not answer"),
    ],
)
def test_unreachable_backend_raises_service_unavailable(
    fake_post: PostRecorder, error: Exception, expected_detail: str
) -> None:
    fake_post.error = error

    with pytest.raises(ScoringServiceUnavailableError) as exc_info:
        request_score(VALID_PAYLOAD)

    assert str(exc_info.value) == SERVICE_UNAVAILABLE_MESSAGE
    assert any(expected_detail in line for line in exc_info.value.details)
    assert exc_info.value.__cause__ is error


def test_connection_error_tells_how_to_start_the_backend(
    fake_post: PostRecorder,
) -> None:
    fake_post.error = requests.exceptions.ConnectionError("refused")

    with pytest.raises(ScoringServiceUnavailableError) as exc_info:
        request_score(VALID_PAYLOAD)

    assert any("uvicorn src.api.main:app" in line for line in exc_info.value.details)


def test_other_request_exception_is_a_generic_api_error(
    fake_post: PostRecorder,
) -> None:
    error = requests.exceptions.InvalidURL("bad url")
    fake_post.error = error

    with pytest.raises(ScoringAPIError, match="request to the scoring API") as exc_info:
        request_score(VALID_PAYLOAD)

    assert not isinstance(exc_info.value, ScoringServiceUnavailableError)
    assert not isinstance(exc_info.value, ScoringResponseError)
    assert exc_info.value.__cause__ is error


# --- Error responses ------------------------------------------------------------
def test_422_lists_field_errors(fake_post: PostRecorder) -> None:
    detail = [
        {
            "type": "greater_than_equal",
            "loc": ["body", "loan_amnt"],
            "msg": "Input should be greater than or equal to 500",
            "input": 100,
        }
    ]
    fake_post.response = make_response(422, {"detail": detail})

    with pytest.raises(ScoringResponseError, match="422") as exc_info:
        request_score(VALID_PAYLOAD)

    assert exc_info.value.status_code == 422
    assert exc_info.value.details == (
        "loan_amnt: Input should be greater than or equal to 500",
    )


@pytest.mark.parametrize("status_code", [500, 503])
def test_server_error_shows_api_detail(
    fake_post: PostRecorder, status_code: int
) -> None:
    fake_post.response = make_response(
        status_code, {"detail": "The scoring model is not loaded."}
    )

    with pytest.raises(ScoringResponseError) as exc_info:
        request_score(VALID_PAYLOAD)

    assert exc_info.value.status_code == status_code
    message = str(exc_info.value)
    assert f"HTTP {status_code}" in message
    assert "The scoring model is not loaded." in message


def test_error_without_json_body_is_still_reported(fake_post: PostRecorder) -> None:
    fake_post.response = make_response(502, raw_text="Bad Gateway")

    with pytest.raises(ScoringResponseError, match=r"Unexpected response.*HTTP 502"):
        request_score(VALID_PAYLOAD)


# --- Malformed successful responses ---------------------------------------------
def test_non_json_success_body_raises(fake_post: PostRecorder) -> None:
    fake_post.response = make_response(200, raw_text="<html>proxy page</html>")

    with pytest.raises(ScoringResponseError, match="non-JSON"):
        request_score(VALID_PAYLOAD)


@pytest.mark.parametrize(
    "body",
    [
        {k: v for k, v in API_RESPONSE.items() if k != "default_probability"},
        {**API_RESPONSE, "default_probability": 1.7},
        ["not", "an", "object"],
    ],
    ids=["missing-probability", "probability-above-1", "not-an-object"],
)
def test_invalid_success_body_raises(fake_post: PostRecorder, body: object) -> None:
    fake_post.response = make_response(200, body)

    with pytest.raises(ScoringResponseError, match="valid score"):
        request_score(VALID_PAYLOAD)


# --- Helpers --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (None, []),
        ("Something broke", ["Something broke"]),
        ([{"loc": ["body"], "msg": "Field required"}], ["Field required"]),
        (
            [{"loc": ["body", "items", 0, "dti"], "msg": "bad"}],
            ["items.0.dti: bad"],
        ),
        (["plain string item"], ["plain string item"]),
    ],
)
def test_format_validation_errors(detail: object, expected: list[str]) -> None:
    assert format_validation_errors(detail) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, DEFAULT_RISK_THRESHOLD), ("", DEFAULT_RISK_THRESHOLD), ("0.35", 0.35)],
)
def test_parse_risk_threshold_valid(raw: str | None, expected: float) -> None:
    assert parse_risk_threshold(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["abc", "0", "1", "1.5", "-0.2", "nan"])
def test_parse_risk_threshold_invalid(raw: str) -> None:
    with pytest.raises(ValueError, match="RISK_THRESHOLD"):
        parse_risk_threshold(raw)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, False), (0.499, False), (0.5, True), (0.9, True)],
)
def test_is_high_risk_threshold_is_inclusive(
    probability: float, expected: bool
) -> None:
    assert is_high_risk(probability, threshold=0.5) is expected


# --- Contract with the API schemas ----------------------------------------------
def test_form_payload_passes_strict_api_validation() -> None:
    """What the form sends must be accepted by ``ApplicationCreate`` (strict).

    Validated from JSON, exactly as FastAPI receives it over HTTP.
    """
    assert list(VALID_PAYLOAD) == list(FEATURE_ORDER)

    application = ApplicationCreate.model_validate_json(json.dumps(VALID_PAYLOAD))

    assert application.model_dump() == VALID_PAYLOAD


def test_scoring_result_reads_api_response_example() -> None:
    """``ScoringResult`` must accept the documented ``ApplicationResponse``."""
    example = ApplicationResponse.model_json_schema()["examples"][0]

    result = ScoringResult.model_validate(example)

    assert result.id == example["id"]
    assert result.default_probability == example["default_probability"]
