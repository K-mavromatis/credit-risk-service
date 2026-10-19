"""Tests for the scoring history page and sidebar navigation (frontend/app.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frontend.app import (
    HISTORY_LIMIT,
    HISTORY_SCHEMA,
    PAGE_HISTORY,
    PAGE_SCORING,
    empty_history_frame,
    load_scoring_history,
    make_dummy_history,
    to_display_frame,
)
from src.db.models import CreditApplication

APP_PATH: Path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def _dtypes(frame: pd.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in frame.dtypes.items()}


def test_empty_history_frame_has_schema_and_no_rows() -> None:
    frame = empty_history_frame()

    assert frame.empty
    assert list(frame.columns) == list(HISTORY_SCHEMA)
    assert _dtypes(frame) == HISTORY_SCHEMA


def test_dummy_history_matches_schema() -> None:
    frame = make_dummy_history()

    assert not frame.empty
    assert list(frame.columns) == list(HISTORY_SCHEMA)
    assert _dtypes(frame) == HISTORY_SCHEMA


def test_history_columns_exist_in_db_model() -> None:
    db_columns: set[str] = set(CreditApplication.__table__.columns.keys())

    assert set(HISTORY_SCHEMA) <= db_columns


def test_dummy_history_is_newest_first_with_unique_ids() -> None:
    frame = make_dummy_history()

    assert frame["created_at"].is_monotonic_decreasing
    assert frame["id"].is_unique


def test_dummy_probabilities_are_in_unit_interval_or_missing() -> None:
    probabilities: pd.Series = make_dummy_history()["default_probability"]

    assert probabilities.dropna().between(0.0, 1.0).all()
    assert probabilities.isna().any(), "keep one unscored row to test the NULL case"


@pytest.mark.parametrize("limit", [1, 3, HISTORY_LIMIT])
def test_load_scoring_history_respects_limit(limit: int) -> None:
    assert len(load_scoring_history(limit)) <= limit


@pytest.mark.parametrize("limit", [0, -5])
def test_load_scoring_history_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        load_scoring_history(limit)


def test_to_display_frame_formats_purpose_without_mutating_input() -> None:
    history = make_dummy_history()
    raw_purpose: list[str] = history["purpose"].tolist()

    display = to_display_frame(history)

    assert "Debt consolidation" in display["purpose"].tolist()
    assert history["purpose"].tolist() == raw_purpose


def test_sidebar_switches_between_pages() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    nav = app.sidebar.radio[0]
    assert nav.value == PAGE_SCORING
    assert len(app.dataframe) == 0

    nav.set_value(PAGE_HISTORY).run()

    assert not app.exception
    assert app.title[0].value == PAGE_HISTORY
    assert len(app.dataframe) == 1
