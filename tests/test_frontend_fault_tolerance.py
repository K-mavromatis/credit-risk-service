"""Fault tolerance of the Streamlit UI (Sprint 4, Card 5).

Three layers are tested:

1. ``validate_inputs`` -- client-side boundary checks that block the request
   (zero / negative loan amounts, negative incomes, negative counts, NaN ...).
   Each blocked payload is also checked against the real ``ApplicationCreate``
   schema, so the client rules can never be looser than the API's.
2. ``render_api_error`` -- which Streamlit element each failure maps to
   (``st.error`` for an unreachable backend, ``st.warning`` for 422 / 500).
   ``streamlit`` is replaced by a recorder, so these tests are instant.
3. End to end with ``streamlit.testing.v1.AppTest``: the real script is run,
   the form is submitted, and ``requests.post`` is faked, so no server is
   needed and the app must never show a traceback.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from frontend import app
from frontend.app import (
    FEATURE_ORDER,
    NUMERIC_BOUNDS,
    SERVICE_UNAVAILABLE_MESSAGE,
    FeatureValue,
    ScoringAPIError,
    ScoringResponseError,
    ScoringServiceUnavailableError,
    format_details_markdown,
    format_error_markdown,
    render_api_error,
    validate_inputs,
)
from src.api.schemas import ApplicationCreate

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"

# Synthetic applicant that passes every rule (values as the widgets return them).
VALID_PAYLOAD: dict[str, FeatureValue] = {
    "loan_amnt": 10_000.0,
    "annual_inc": 60_000.0,
    "dti": 18.0,
    "emp_length": 5,
    "fico_range_low": 700,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_util": 50.0,
    "mort_acc": 1,
    "term": "36 months",
    "home_ownership": "MORTGAGE",
    "verification_status": "Source Verified",
    "purpose": "debt_consolidation",
}


def with_value(name: str, value: FeatureValue) -> dict[str, FeatureValue]:
    """Return a copy of VALID_PAYLOAD with one feature replaced."""
    return {**VALID_PAYLOAD, name: value}


# --- 1. Client-side validation -----------------------------------------------------
def test_valid_payload_has_no_errors_or_warnings() -> None:
    assert list(VALID_PAYLOAD) == list(FEATURE_ORDER)

    result = validate_inputs(VALID_PAYLOAD)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


@pytest.mark.parametrize("name", ["emp_length", "mort_acc"])
def test_unknown_nullable_features_are_allowed(name: str) -> None:
    assert validate_inputs(with_value(name, None)).is_valid


@pytest.mark.parametrize(
    ("name", "value", "expected_message"),
    [
        ("loan_amnt", 0.0, "Loan amount must be greater than $0."),
        ("loan_amnt", -5_000.0, "Loan amount must be greater than $0."),
        ("loan_amnt", 100.0, "Loan amount must be between $500 and $40,000."),
        ("loan_amnt", 50_000.0, "Loan amount must be between $500 and $40,000."),
        ("annual_inc", -1.0, "Annual income cannot be negative."),
        ("annual_inc", 0.0, "Annual income must be greater than $0."),
        ("emp_length", 11, "`emp_length` must be 0–10, got 11."),
        ("fico_range_low", 900, "`fico_range_low` must be 300–850, got 900."),
        ("loan_amnt", math.nan, "`loan_amnt` must be a finite number"),
        ("annual_inc", math.inf, "`annual_inc` must be a finite number"),
        ("dti", "18", "`dti` must be a finite number"),
        ("loan_amnt", None, "`loan_amnt` is required."),
    ],
)
def test_out_of_bounds_values_block_the_request(
    name: str, value: FeatureValue, expected_message: str
) -> None:
    payload = with_value(name, value)

    result = validate_inputs(payload)

    assert not result.is_valid
    assert any(error.startswith(expected_message) for error in result.errors), (
        result.errors
    )
    # The API would reject it too: the client is never looser than the server.
    with pytest.raises(ValidationError):
        ApplicationCreate.model_validate(payload)


@pytest.mark.parametrize("name", sorted(NUMERIC_BOUNDS))
def test_negative_values_are_blocked_for_every_bounded_feature(name: str) -> None:
    payload = with_value(name, -1)

    result = validate_inputs(payload)

    assert result.errors == [f"`{name}` must be {_expected_range(name)}, got -1."]
    with pytest.raises(ValidationError):
        ApplicationCreate.model_validate(payload)


def _expected_range(name: str) -> str:
    low, high = NUMERIC_BOUNDS[name]
    return f"≥ {low:g}" if high is None else f"{low:g}–{high:g}"


def test_each_problem_is_reported_once() -> None:
    payload = {**VALID_PAYLOAD, "loan_amnt": 0.0, "annual_inc": -1.0, "pub_rec": -2}

    result = validate_inputs(payload)

    assert result.errors == [
        "Loan amount must be greater than $0.",
        "Annual income cannot be negative.",
        "`pub_rec` must be ≥ 0, got -2.",
    ]


def test_unknown_category_is_blocked() -> None:
    result = validate_inputs(with_value("purpose", "lottery"))

    assert result.errors == ["`purpose` has an unknown value: 'lottery'."]


def test_fico_outside_training_range_only_warns() -> None:
    result = validate_inputs(with_value("fico_range_low", 600))

    assert result.is_valid
    assert len(result.warnings) == 1
    assert "outside the training range" in result.warnings[0]


def test_invalid_fico_is_an_error_not_a_warning() -> None:
    result = validate_inputs(with_value("fico_range_low", 200))

    assert not result.is_valid
    assert result.warnings == []


# --- 2. Error rendering (streamlit replaced by a recorder) -------------------------
@dataclass
class StreamlitRecorder:
    """Stand-in for the ``st`` module: records which elements were drawn."""

    calls: list[tuple[str, str]] = field(default_factory=list)

    def error(self, body: str, icon: str | None = None) -> None:
        self.calls.append(("error", body))

    def warning(self, body: str, icon: str | None = None) -> None:
        self.calls.append(("warning", body))

    def markdown(self, body: str) -> None:
        self.calls.append(("markdown", body))

    @contextmanager
    def expander(self, label: str) -> Iterator[None]:
        self.calls.append(("expander", label))
        yield

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> StreamlitRecorder:
    recorder = StreamlitRecorder()
    monkeypatch.setattr(app, "st", recorder)
    return recorder


def test_unreachable_backend_shows_fixed_error_message(
    fake_st: StreamlitRecorder,
) -> None:
    error = ScoringServiceUnavailableError(details=["Cannot connect to http://x."])

    render_api_error(error)

    assert fake_st.calls[0] == ("error", SERVICE_UNAVAILABLE_MESSAGE)
    assert fake_st.kinds() == ["error", "expander", "markdown"]
    assert "warning" not in fake_st.kinds()


def test_unreachable_backend_without_details_has_no_expander(
    fake_st: StreamlitRecorder,
) -> None:
    render_api_error(ScoringServiceUnavailableError())

    assert fake_st.calls == [("error", SERVICE_UNAVAILABLE_MESSAGE)]


@pytest.mark.parametrize(
    ("status_code", "message", "details"),
    [
        (422, "The API rejected the application data (HTTP 422).", ["dti: bad"]),
        (500, "Scoring API error (HTTP 500): Model failed.", []),
        (503, "Scoring API error (HTTP 503): DB down.", []),
    ],
)
def test_error_responses_show_a_detailed_warning(
    fake_st: StreamlitRecorder, status_code: int, message: str, details: list[str]
) -> None:
    error = ScoringResponseError(message, status_code=status_code, details=details)

    render_api_error(error)

    assert fake_st.kinds() == ["warning"]
    body = fake_st.calls[0][1]
    assert f"HTTP {status_code}" in body
    for line in details:
        assert f"- {line}" in body


def test_other_failures_show_an_error_with_details(
    fake_st: StreamlitRecorder,
) -> None:
    render_api_error(ScoringAPIError("The request failed.", details=["bad url"]))

    assert fake_st.calls == [("error", "The request failed.\n- bad url")]


def test_error_markdown_escapes_dollars_and_lists_details() -> None:
    error = ScoringAPIError("Limit is $500", details=["min $500", "max $40,000"])

    assert format_error_markdown(error) == (
        r"Limit is \$500" "\n" r"- min \$500" "\n" r"- max \$40,000"
    )
    assert format_error_markdown(ScoringAPIError("plain")) == "plain"
    assert format_details_markdown([]) == ""


# --- 3. End to end: the real Streamlit script with a faked backend ----------------
def make_response(status_code: int, body: object) -> requests.Response:
    """Build a real ``requests.Response`` with a JSON body."""
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode()
    response.headers["Content-Type"] = "application/json"
    return response


@dataclass
class PostRecorder:
    """Stand-in for ``requests.post``: records calls, returns or raises."""

    response: requests.Response | None = None
    error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def __call__(self, url: str, **_: Any) -> requests.Response:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        assert self.response is not None, "PostRecorder needs a response or error"
        return self.response


@pytest.fixture
def fake_post(monkeypatch: pytest.MonkeyPatch) -> PostRecorder:
    recorder = PostRecorder()
    # The AppTest script runs in this process and does ``import requests``,
    # so patching the module attribute reaches it.
    monkeypatch.setattr(requests, "post", recorder)
    return recorder


def submit_form(loan_amount: float | None = None) -> Any:
    """Run the app, optionally change the loan amount, click Calculate Risk."""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception
    if loan_amount is not None:
        loan_input = next(w for w in at.number_input if w.label == "Loan amount ($)")
        loan_input.set_value(loan_amount)
    return at.button[0].click().run()


def test_ui_survives_backend_down(fake_post: PostRecorder) -> None:
    fake_post.error = requests.exceptions.ConnectionError("Connection refused")

    at = submit_form()

    assert not at.exception  # no traceback in the page
    assert len(fake_post.calls) == 1
    assert [e.value for e in at.error] == [SERVICE_UNAVAILABLE_MESSAGE]
    assert len(at.success) == 0


def test_ui_shows_warning_on_server_error(fake_post: PostRecorder) -> None:
    fake_post.response = make_response(500, {"detail": "Model inference failed."})

    at = submit_form()

    assert not at.exception
    assert len(at.warning) == 1
    assert "HTTP 500" in at.warning[0].value
    assert "Model inference failed." in at.warning[0].value


def test_ui_blocks_zero_loan_before_any_request(fake_post: PostRecorder) -> None:
    at = submit_form(loan_amount=0.0)

    assert not at.exception
    assert fake_post.calls == []  # the request was never sent
    assert any("Loan amount must be greater than" in e.value for e in at.error)
