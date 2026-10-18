from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, TypeAlias

import streamlit as st

PAGE_TITLE: Final[str] = "Credit Risk Scoring"
FORM_KEY: Final[str] = "loan_application_form"

NumericValue: TypeAlias = float | int | None
FeatureValue: TypeAlias = float | int | str | None

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


# --- Pure logic (no Streamlit calls; unit-testable) ------------------------------
@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating a form payload."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
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


# --- UI rendering ---------------------------------------------------------------
def escape_markdown_dollars(text: str) -> str:
    """Stop Streamlit from rendering text between two '$' signs as LaTeX."""
    return text.replace("$", r"\$")


def format_category(value: str) -> str:
    """Turn a raw category code into a readable label.

    Example: 'debt_consolidation' -> 'Debt consolidation'.
    """
    return value.replace("_", " ").capitalize()


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


def render_payload(payload: dict[str, FeatureValue]) -> None:
    """Display the validated payload that will be sent to the scoring API."""
    st.success(f"Input is valid — {len(payload)} features ready for scoring.")
    st.subheader("Request payload")
    st.caption(
        "Temporary preview. The next card sends this JSON "
        "to the FastAPI scoring endpoint."
    )
    st.json(payload, expanded=True)


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
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

    if validation.is_valid:
        render_payload(payload)


if __name__ == "__main__":
    main()
