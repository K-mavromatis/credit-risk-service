"""Streamlit UI of the credit scoring service.

Pages:
    New Application -- loan form. On submit the payload is validated locally,
                       sent to ``POST /predict`` and the returned score is shown.
    Scoring History -- the most recent applications (synthetic data for now).

The UI talks to the backend only over HTTP (CONTRIBUTING.md): it never imports
from ``src/``. The API address comes from ``API_BASE_URL`` (environment or
``.env``), default ``http://localhost:8000``.

Run from the project root, with the API already running:
    uvicorn src.api.main:app --reload
    streamlit run frontend/app.py
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, TypeAlias

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PAGE_TITLE: Final[str] = "Credit Risk Scoring"
FORM_KEY: Final[str] = "loan_application_form"

NumericValue: TypeAlias = float | int | None
FeatureValue: TypeAlias = float | int | str | None
HistoryCell: TypeAlias = float | int | str | None

# --- Navigation ----------------------------------------------------------------
PAGE_SCORING: Final[str] = "New Application"
PAGE_HISTORY: Final[str] = "Scoring History"
PAGES: Final[tuple[str, ...]] = (PAGE_SCORING, PAGE_HISTORY)
NAV_KEY: Final[str] = "nav_page"

# --- Feature contract: mirrors NUMERIC/CATEGORICAL_FEATURES in notebook 02 ---
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length",
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
FEATURE_ORDER: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Business rules ----------------------------------------------------------
MIN_LOAN_AMOUNT: Final[float] = 500.0
MAX_LOAN_AMOUNT: Final[float] = 40_000.0
# FICO range seen in training (notebook 02). Outside it, the model extrapolates.
FICO_TRAINING_MIN: Final[int] = 625
FICO_TRAINING_MAX: Final[int] = 845
# Missingness is informative for these, so the form may send None (see notebook 02).
NULLABLE_FEATURES: Final[frozenset[str]] = frozenset({"emp_length", "mort_acc"})

# --- Scoring API -------------------------------------------------------------------
# Read from the environment (or .env), so the same code works locally and in Docker.
load_dotenv()
DEFAULT_API_BASE_URL: Final[str] = "http://localhost:8000"
API_BASE_URL: Final[str] = (os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL).rstrip(
    "/"
)
PREDICT_URL: Final[str] = f"{API_BASE_URL}/predict"
# (connect, read) timeouts in seconds. Without a timeout, requests can wait
# forever and the "Calculate Risk" spinner would never stop.
REQUEST_TIMEOUT: Final[tuple[float, float]] = (3.05, 30.0)
# Probability of default at or above which an application is flagged as high
# risk. Placeholder until a tuned threshold is chosen in notebook 03 (changing
# it needs evaluation metrics, see CONTRIBUTING.md). Override: RISK_THRESHOLD.
DEFAULT_RISK_THRESHOLD: Final[float] = 0.5

# Category levels exactly as they appear in the training data (see notebook 02).
# Ordered by frequency, so the default (index 0) is the most common level.
TERM_OPTIONS: Final[tuple[str, ...]] = ("36 months", "60 months")
HOME_OWNERSHIP_OPTIONS: Final[tuple[str, ...]] = ("MORTGAGE", "RENT", "OWN", "OTHER")
VERIFICATION_STATUS_OPTIONS: Final[tuple[str, ...]] = (
    "Source Verified",
    "Verified",
    "Not Verified",
)
PURPOSE_OPTIONS: Final[tuple[str, ...]] = (
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
)
CATEGORICAL_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "term": TERM_OPTIONS,
    "home_ownership": HOME_OWNERSHIP_OPTIONS,
    "verification_status": VERIFICATION_STATUS_OPTIONS,
    "purpose": PURPOSE_OPTIONS,
}

# --- Scoring history -------------------------------------------------------------
# Same default as crud.get_recent_applications(), so the page shape won't change
# when real rows arrive.
HISTORY_LIMIT: Final[int] = 10
# Column -> pandas dtype. Names are a subset of CreditApplication columns
# (src/db/models.py), so a DB query result drops in without renaming.
HISTORY_SCHEMA: Final[dict[str, str]] = {
    "created_at": "datetime64[ns, UTC]",
    "id": "int64",
    "loan_amnt": "float64",
    "term": "string",
    "purpose": "string",
    "annual_inc": "float64",
    "fico_range_low": "int64",
    "default_probability": "float64",  # NaN = not scored yet (NULL in the DB)
}


# --- Pure logic (no Streamlit calls; unit-testable) ------------------------------
@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating a form payload."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when the payload has no blocking errors (warnings are allowed)."""
        return not self.errors


def _to_float(value: FeatureValue) -> float | None:
    """Return value as float if it is a real number, otherwise None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_payload(
    financial_inputs: dict[str, NumericValue],
    profile_inputs: dict[str, str],
) -> dict[str, FeatureValue]:
    """Merge widget groups into one payload ordered by FEATURE_ORDER.

    Raises:
        ValueError: if the widgets produced missing or unexpected feature keys.
    """
    merged: dict[str, FeatureValue] = {**financial_inputs, **profile_inputs}

    missing: set[str] = set(FEATURE_ORDER) - merged.keys()
    unexpected: set[str] = merged.keys() - set(FEATURE_ORDER)
    if missing or unexpected:
        raise ValueError(
            f"Form/feature contract mismatch. Missing: {sorted(missing)}, "
            f"unexpected: {sorted(unexpected)}"
        )

    return {name: merged[name] for name in FEATURE_ORDER}


def validate_inputs(payload: dict[str, FeatureValue]) -> ValidationResult:
    """Apply business rules to the payload. Messages are plain text (no Markdown)."""
    result = ValidationResult()

    # 1. Required fields must be present.
    for name in FEATURE_ORDER:
        if payload.get(name) is None and name not in NULLABLE_FEATURES:
            result.errors.append(f"`{name}` is required.")

    # 2. Numeric business rules.
    loan_amnt: float | None = _to_float(payload.get("loan_amnt"))
    if loan_amnt is not None and not MIN_LOAN_AMOUNT <= loan_amnt <= MAX_LOAN_AMOUNT:
        result.errors.append(
            f"Loan amount must be between ${MIN_LOAN_AMOUNT:,.0f} "
            f"and ${MAX_LOAN_AMOUNT:,.0f}."
        )

    annual_inc: float | None = _to_float(payload.get("annual_inc"))
    if annual_inc is not None and annual_inc <= 0:
        result.errors.append("Annual income must be greater than $0.")

    # 3. Categorical values must be levels the model was trained on.
    for name, options in CATEGORICAL_OPTIONS.items():
        value: FeatureValue = payload.get(name)
        if value not in options:
            result.errors.append(f"`{name}` has an unknown value: {value!r}.")

    # 4. Non-blocking warnings: valid input, but outside the training distribution.
    fico: float | None = _to_float(payload.get("fico_range_low"))
    if fico is not None and not FICO_TRAINING_MIN <= fico <= FICO_TRAINING_MAX:
        result.warnings.append(
            f"FICO {fico:.0f} is outside the training range "
            f"({FICO_TRAINING_MIN}–{FICO_TRAINING_MAX}); "
            "the score may be less reliable."
        )

    return result


# --- Scoring API client (no Streamlit calls; unit-testable) -----------------------
class ScoringResult(BaseModel):
    """The part of the API's ``ApplicationResponse`` that the UI displays.

    Defined here instead of imported from ``src.api.schemas``, because the
    frontend talks to the backend only over HTTP. The echoed feature fields of
    the response are ignored.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(description="Primary key of the stored application")
    default_probability: float = Field(
        ge=0, le=1, description="Predicted probability of default, in [0, 1]"
    )
    created_at: datetime = Field(description="When the application was stored")


class ScoringAPIError(Exception):
    """The scoring API could not return a usable score.

    ``str(error)`` is a short message for the user; ``details`` holds extra
    lines, such as the field errors of a ``422`` response.
    """

    def __init__(self, message: str, details: Sequence[str] = ()) -> None:
        """Create the error.

        Args:
            message: Short, user-facing description of the failure.
            details: Optional extra lines, e.g. field errors of a 422 response.
        """
        super().__init__(message)
        self.details: tuple[str, ...] = tuple(details)


def parse_risk_threshold(raw: str | None) -> float:
    """Parse the ``RISK_THRESHOLD`` setting. ``None`` or blank gives the default.

    Raises:
        ValueError: if the value is not a number strictly between 0 and 1.
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


RISK_THRESHOLD: Final[float] = parse_risk_threshold(os.getenv("RISK_THRESHOLD"))


def is_high_risk(default_probability: float, threshold: float = RISK_THRESHOLD) -> bool:
    """Return True if the application should be flagged as high default risk."""
    return default_probability >= threshold


def format_validation_errors(detail: object) -> list[str]:
    """Turn FastAPI's ``422`` ``detail`` into readable lines.

    Example: ``[{"loc": ["body", "loan_amnt"], "msg": "Input should be ..."}]``
    becomes ``["loan_amnt: Input should be ..."]``.
    """
    if not isinstance(detail, list):
        return [str(detail)] if detail else []

    lines: list[str] = []
    for item in detail:
        if not isinstance(item, dict):
            lines.append(str(item))
            continue
        loc: object = item.get("loc", [])
        parts: list[str] = (
            [str(part) for part in loc if part != "body"]
            if isinstance(loc, (list, tuple))  # JSON gives a list, Pydantic a tuple
            else []
        )
        message: str = str(item.get("msg", "invalid value"))
        lines.append(f"{'.'.join(parts)}: {message}" if parts else message)
    return lines


def _response_detail(response: requests.Response) -> object:
    """Return the ``detail`` field of an error response, or None if there is none."""
    try:
        body: object = response.json()
    except ValueError:  # requests.JSONDecodeError subclasses ValueError
        return None
    return body.get("detail") if isinstance(body, dict) else None


def error_from_response(response: requests.Response) -> ScoringAPIError:
    """Build a user-facing error from a non-2xx API response."""
    status_code: int = response.status_code
    detail: object = _response_detail(response)

    if status_code == 422:
        return ScoringAPIError(
            "The API rejected the application data (HTTP 422).",
            details=format_validation_errors(detail),
        )
    if isinstance(detail, str) and detail:
        return ScoringAPIError(f"Scoring API error (HTTP {status_code}): {detail}")
    return ScoringAPIError(
        f"Unexpected response from the scoring API (HTTP {status_code})."
    )


def request_score(
    payload: Mapping[str, FeatureValue],
    url: str = PREDICT_URL,
    timeout: tuple[float, float] = REQUEST_TIMEOUT,
) -> ScoringResult:
    """Send one application to ``POST /predict`` and return its score.

    Raises:
        ScoringAPIError: if the API is unreachable, too slow, answers with an
            error status, or returns a body that is not a valid score.
    """
    try:
        response: requests.Response = requests.post(
            url, json=dict(payload), timeout=timeout
        )
    except requests.exceptions.ConnectionError as exc:
        # Checked before Timeout: ConnectTimeout is both, and "cannot connect"
        # is the more useful message for it.
        raise ScoringAPIError(
            f"Cannot connect to the scoring API at {url}. Is the backend running?",
            details=[
                "Start it from the project root: uvicorn src.api.main:app --reload"
            ],
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ScoringAPIError(
            f"The scoring API at {url} did not answer within {timeout[1]:.0f} s. "
            "Please try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ScoringAPIError(f"The request to the scoring API failed: {exc}") from exc

    if not response.ok:
        raise error_from_response(response)

    try:
        body: object = response.json()
    except ValueError as exc:
        raise ScoringAPIError("The scoring API returned a non-JSON response.") from exc

    try:
        return ScoringResult.model_validate(body)
    except ValidationError as exc:
        raise ScoringAPIError(
            "The scoring API response does not contain a valid score.",
            details=format_validation_errors(exc.errors(include_url=False)),
        ) from exc


def format_category(value: str) -> str:
    """Turn a raw category code into a readable label.

    Example: 'debt_consolidation' -> 'Debt consolidation'.
    """
    return value.replace("_", " ").capitalize()


def empty_history_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the history schema (columns and dtypes)."""
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in HISTORY_SCHEMA.items()}
    )


def make_dummy_history() -> pd.DataFrame:
    """Return synthetic scoring records shaped like the future DB query result.

    All values are invented (no real applicants) and fixed, so the page looks
    the same on every run. Rows are ordered newest first, like the DB query.
    """
    records: dict[str, list[HistoryCell]] = {
        "created_at": [
            "2026-10-12 09:14",
            "2026-10-12 11:02",
            "2026-10-13 15:47",
            "2026-10-14 10:30",
            "2026-10-15 13:05",
            "2026-10-16 08:51",
            "2026-10-16 17:22",
            "2026-10-17 12:40",
        ],
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "loan_amnt": [12_000, 25_000, 5_000, 35_000, 8_500, 15_000, 20_000, 3_000],
        "term": [
            "36 months",
            "60 months",
            "36 months",
            "60 months",
            "36 months",
            "36 months",
            "60 months",
            "36 months",
        ],
        "purpose": [
            "debt_consolidation",
            "home_improvement",
            "credit_card",
            "small_business",
            "car",
            "debt_consolidation",
            "medical",
            "vacation",
        ],
        "annual_inc": [65_000, 92_000, 48_000, 110_000, 54_000, 71_000, 39_000, 83_000],
        "fico_range_low": [710, 685, 740, 660, 700, 725, 645, 770],
        # The newest applicant has no score yet: the DB allows NULL here.
        "default_probability": [0.08, 0.19, 0.05, 0.34, 0.12, 0.07, 0.41, None],
    }
    frame = pd.DataFrame(records)
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame = frame.astype(HISTORY_SCHEMA)[list(HISTORY_SCHEMA)]
    return frame.sort_values(["created_at", "id"], ascending=False, ignore_index=True)


def load_scoring_history(limit: int = HISTORY_LIMIT) -> pd.DataFrame:
    """Return the ``limit`` most recent scored applications, newest first.

    This is the only place the history page gets data from. Today it returns
    dummy rows; when the API exists, only this body changes (a call to the
    endpoint backed by ``crud.get_recent_applications``), not the page.

    Raises:
        ValueError: if ``limit`` is not a positive integer.
    """
    if limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    return make_dummy_history().head(limit)


def to_display_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the history with human-readable category labels."""
    return history.assign(purpose=history["purpose"].map(format_category))


# --- UI rendering ---------------------------------------------------------------
def escape_markdown_dollars(text: str) -> str:
    """Stop Streamlit from rendering text between two '$' signs as LaTeX."""
    return text.replace("$", r"\$")


def render_header() -> None:
    """Render the page title and short usage hint."""
    st.title(PAGE_TITLE)
    st.caption("Enter the applicant's details and click **Calculate Risk**.")


def render_financial_inputs() -> dict[str, NumericValue]:
    """Render widgets for the 9 numeric model features and return their raw values."""
    loan_amnt: float = st.number_input(
        "Loan amount ($)",
        min_value=0.0,
        max_value=MAX_LOAN_AMOUNT,
        value=10_000.0,
        step=500.0,
        format="%.0f",
        help=escape_markdown_dollars(
            f"Requested principal: ${MIN_LOAN_AMOUNT:,.0f} – ${MAX_LOAN_AMOUNT:,.0f}."
        ),
    )
    annual_inc: float = st.number_input(
        "Annual income ($)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=60_000.0,
        step=1_000.0,
        format="%.0f",
    )
    dti: float = st.slider(
        "Debt-to-income ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=0.1,
        help="Monthly debt payments (excluding mortgage) / monthly income.",
    )
    emp_length: int | None = st.number_input(
        "Employment length (years)",
        min_value=0,
        max_value=10,
        value=None,
        step=1,
        placeholder="Unknown",
        help="0 = less than 1 year, 10 = 10+ years. Leave empty if unknown.",
    )
    fico_range_low: int = st.slider(
        "FICO score (lower bound)",
        min_value=300,
        max_value=850,
        value=700,
        step=5,
    )
    open_acc: int = st.number_input(
        "Open credit lines", min_value=0, max_value=100, value=10, step=1
    )
    pub_rec: int = st.number_input(
        "Public derogatory records", min_value=0, max_value=100, value=0, step=1
    )
    revol_util: float = st.slider(
        "Revolving utilization (%)",
        min_value=0.0,
        max_value=150.0,
        value=50.0,
        step=0.1,
        help="Revolving balance / total revolving credit limit. Can exceed 100%.",
    )
    mort_acc: int | None = st.number_input(
        "Mortgage accounts",
        min_value=0,
        max_value=60,
        value=None,
        step=1,
        placeholder="Unknown",
    )

    return {
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "dti": dti,
        "emp_length": emp_length,
        "fico_range_low": fico_range_low,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_util": revol_util,
        "mort_acc": mort_acc,
    }


def render_profile_inputs() -> dict[str, str]:
    """Render the 4 categorical selectboxes and return raw category codes."""
    term: str = st.selectbox("Loan term", options=TERM_OPTIONS, index=0)
    home_ownership: str = st.selectbox(
        "Housing status",
        options=HOME_OWNERSHIP_OPTIONS,
        index=0,
        format_func=format_category,
    )
    verification_status: str = st.selectbox(
        "Income verification",
        options=VERIFICATION_STATUS_OPTIONS,
        index=0,
        help="Whether the lender verified the applicant's income (or its source).",
    )
    purpose: str = st.selectbox(
        "Loan purpose",
        options=PURPOSE_OPTIONS,
        index=0,
        format_func=format_category,
    )

    return {
        "term": term,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
    }


def render_validation(result: ValidationResult) -> None:
    """Show validation errors and warnings in the UI."""
    for message in result.errors:
        st.error(escape_markdown_dollars(message))
    for message in result.warnings:
        st.warning(escape_markdown_dollars(message))


def render_api_error(error: ScoringAPIError) -> None:
    """Show a scoring API failure as one red box, with details as a bullet list."""
    lines: list[str] = [escape_markdown_dollars(str(error))]
    lines.extend(f"- {escape_markdown_dollars(detail)}" for detail in error.details)
    st.error("\n".join(lines), icon="⚠️")


def render_scoring_result(result: ScoringResult, threshold: float) -> None:
    """Show the decision banner (green / red) and the key numbers of the score."""
    probability: float = result.default_probability
    st.subheader("Scoring result")

    if is_high_risk(probability, threshold):
        st.error(
            f"**High default risk.** Probability of default {probability:.1%} "
            f"is at or above the {threshold:.1%} threshold.",
            icon="🚫",
        )
    else:
        st.success(
            f"**Credit approved.** Probability of default {probability:.1%} "
            f"is below the {threshold:.1%} threshold.",
            icon="✅",
        )

    col_probability, col_threshold, col_id = st.columns(3)
    col_probability.metric(
        "Probability of default",
        f"{probability:.1%}",
        delta=f"{(probability - threshold) * 100:+.1f} pp vs threshold",
        delta_color="inverse",  # above the threshold = red, below = green
        help="Percentage points above (+) or below (−) the decision threshold.",
    )
    col_threshold.metric("Decision threshold", f"{threshold:.1%}")
    col_id.metric("Application ID", f"#{result.id}")
    st.caption(f"Scored and saved at {result.created_at:%Y-%m-%d %H:%M:%S %Z}".strip())


def render_payload(payload: Mapping[str, FeatureValue]) -> None:
    """Show the exact JSON that was sent to the API, collapsed by default."""
    with st.expander("Request payload sent to the API"):
        st.json(dict(payload), expanded=True)


def render_sidebar() -> str:
    """Render the navigation sidebar and return the name of the selected page."""
    with st.sidebar:
        st.header("Navigation")
        selected: str | None = st.radio(
            "Section", options=PAGES, key=NAV_KEY, label_visibility="collapsed"
        )
        st.divider()
        st.caption("Credit Risk ML service · Sprint 3")
        st.caption(f"API: {API_BASE_URL}")
    return selected if selected is not None else PAGE_SCORING


def render_history_page() -> None:
    """Render the scoring history table (dummy data until the API exists)."""
    st.title(PAGE_HISTORY)
    st.caption(f"The {HISTORY_LIMIT} most recent applications, newest first.")
    st.info(
        "Demo data: this table is filled with synthetic records. Real applications "
        "will appear once the API exposes a history endpoint."
    )

    history: pd.DataFrame = load_scoring_history()
    if history.empty:
        st.info("No applications yet.")
        return

    st.dataframe(
        to_display_frame(history),
        hide_index=True,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "Received (UTC)", format="YYYY-MM-DD HH:mm"
            ),
            "id": st.column_config.NumberColumn("ID", format="%d"),
            "loan_amnt": st.column_config.NumberColumn("Loan amount", format="dollar"),
            "term": st.column_config.TextColumn("Term"),
            "purpose": st.column_config.TextColumn("Purpose"),
            "annual_inc": st.column_config.NumberColumn(
                "Annual income", format="dollar"
            ),
            "fico_range_low": st.column_config.NumberColumn("FICO", format="%d"),
            "default_probability": st.column_config.ProgressColumn(
                "P(default)",
                help="Predicted probability of default. Empty = not scored yet.",
                format="percent",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )


def render_scoring_page() -> None:
    """Render the loan form; on submit validate, score via the API, show the result."""
    render_header()

    with st.form(key=FORM_KEY):
        col_financial, col_profile = st.columns(2, gap="large")

        with col_financial:
            st.subheader("Financial metrics")
            financial_inputs: dict[str, NumericValue] = render_financial_inputs()

        with col_profile:
            st.subheader("Loan & applicant profile")
            profile_inputs: dict[str, str] = render_profile_inputs()

        submitted: bool = st.form_submit_button("Calculate Risk", type="primary")

    if not submitted:
        return

    payload: dict[str, FeatureValue] = build_payload(financial_inputs, profile_inputs)
    validation: ValidationResult = validate_inputs(payload)
    render_validation(validation)
    if not validation.is_valid:
        return  # never send a request the API would reject anyway

    try:
        with st.spinner("Scoring the application..."):
            result: ScoringResult = request_score(payload)
    except ScoringAPIError as error:
        render_api_error(error)
    else:
        render_scoring_result(result, RISK_THRESHOLD)

    render_payload(payload)


def main() -> None:
    """Entry point: configure the page, draw the sidebar, render the chosen page."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")

    page_renderers: dict[str, Callable[[], None]] = {
        PAGE_SCORING: render_scoring_page,
        PAGE_HISTORY: render_history_page,
    }
    selected_page: str = render_sidebar()
    page_renderers[selected_page]()


if __name__ == "__main__":
    main()
