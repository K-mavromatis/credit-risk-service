"""Tests for the streaming pipeline (generator stages, batching, CLI)."""

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

import pytest

from src.ml_model.streaming.cli import main
from src.ml_model.streaming.models import AcceptedLoan, to_accepted_loan
from src.ml_model.streaming.pipeline import (
    PipelineStats,
    batched,
    build_pipeline,
    collect,
    deduplicate,
    parse_all,
    run_pipeline,
)
from src.ml_model.streaming.sources import read_rows_csv, read_rows_jsonl


def make_row(loan_id: str = "1", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "loan_id": loan_id,
        "loan_amnt": 10000.0,
        "term": 36,
        "int_rate": 12.5,
        "grade": "B",
        "addr_state": "CA",
        "loan_status": "Fully Paid",
        "issue_d": "Dec-2015",
        "annual_inc": 55000.0,
    }
    row.update(overrides)
    return row


def make_loan(loan_id: str = "1", **overrides: Any) -> AcceptedLoan:
    return AcceptedLoan.model_validate(make_row(loan_id, **overrides))


def write_jsonl(path: Path, lines: Iterable[object]) -> None:
    """Write dicts as JSON lines; strings are written verbatim (broken lines)."""
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            text = line if isinstance(line, str) else json.dumps(line)
            handle.write(text + "\n")


# --- model ---------------------------------------------------------------


def test_raw_csv_style_row_is_normalised() -> None:
    loan = to_accepted_loan(
        {
            "loan_id": " 123 ",
            "loan_amnt": "3600.0",
            "term": " 36 months",
            "int_rate": "13.99",
            "grade": "C",
            "addr_state": "PA",
            "loan_status": "Fully Paid",
            "issue_d": "Dec-2015",
            "annual_inc": "",
        }
    )
    assert loan is not None
    assert loan.loan_id == "123"
    assert loan.term == 36
    assert loan.loan_amnt == 3600.0
    assert loan.annual_inc is None  # blank income is "unknown", not invalid
    assert loan.key == ("123", "Dec-2015")


@pytest.mark.parametrize(
    "overrides",
    [
        {"loan_amnt": None},
        {"loan_amnt": ""},
        {"loan_amnt": -5},
        {"grade": "Z"},
        {"grade": ""},
        {"addr_state": "CAL"},
        {"term": "abc"},
        {"loan_id": ""},
    ],
)
def test_invalid_field_values_are_rejected(overrides: dict[str, Any]) -> None:
    assert to_accepted_loan(make_row(**overrides)) is None


@pytest.mark.parametrize("junk", ["not json at all", [1, 2], None, 42, {}])
def test_non_record_values_are_rejected_not_raised(junk: object) -> None:
    assert to_accepted_loan(junk) is None


# --- stages ----------------------------------------------------------------


def test_invalid_rows_are_counted() -> None:
    rows = [make_row("1"), make_row("2", loan_amnt=""), "garbage line"]
    stats = PipelineStats()

    result = list(parse_all(rows, stats))  # materialise exactly once

    assert len(result) == 1
    assert stats.read == 3
    assert stats.invalid == 2


def test_deduplicate_counts_repeats_but_keeps_other_issue_months() -> None:
    loans = [
        make_loan("1"),
        make_loan("1"),  # exact repeat
        make_loan("1", issue_d="Jan-2016"),  # same id, different month: kept
        make_loan(" 2 "),
        make_loan("2"),  # repeat after whitespace normalisation
    ]
    stats = PipelineStats()

    unique = list(deduplicate(loans, stats))

    assert [loan.key for loan in unique] == [
        ("1", "Dec-2015"),
        ("1", "Jan-2016"),
        ("2", "Dec-2015"),
    ]
    assert stats.duplicates == 2


def test_collect_aggregates_in_one_pass() -> None:
    loans = [
        make_loan("1", loan_amnt=1000, grade="A", annual_inc=50000),
        make_loan("2", loan_amnt=3000, grade="A", annual_inc=None),
        make_loan("3", loan_amnt=2000, grade="B", annual_inc=70000),
    ]
    stats = PipelineStats()

    collect(loans, stats)

    assert stats.kept == 3
    assert stats.amount_sum == 6000
    assert (stats.amount_min, stats.amount_max) == (1000, 3000)
    assert stats.amount_avg == 2000
    assert stats.income_count == 2
    assert stats.income_avg == 60000
    assert stats.by_grade == Counter({"A": 2, "B": 1})


def test_empty_stats_have_no_averages() -> None:
    stats = PipelineStats()
    assert stats.amount_avg is None
    assert stats.income_avg is None


# --- laziness --------------------------------------------------------------


def test_pipeline_is_lazy() -> None:
    pulled = 0

    def source() -> Iterator[object]:
        nonlocal pulled
        for i in range(1_000_000):
            pulled += 1
            yield make_row(str(i))

    stats = PipelineStats()
    pipeline = deduplicate(parse_all(source(), stats), stats)
    assert pulled == 0  # building the chain reads nothing

    first_five = list(islice(pipeline, 5))

    assert len(first_five) == 5
    assert pulled == stats.read == 5  # and taking 5 pulls exactly 5


def test_preview_does_not_read_the_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "loans.jsonl"
    write_jsonl(path, (make_row(str(i)) for i in range(1000)))
    stats = PipelineStats()

    shown = list(islice(build_pipeline(path, stats), 3))

    assert len(shown) == 3
    assert stats.read == 3


# --- batching --------------------------------------------------------------


def test_batched_splits_tail() -> None:
    assert [len(b) for b in batched(range(7), 3)] == [3, 3, 1]


def test_batched_edge_cases() -> None:
    assert list(batched([], 3)) == []
    assert list(batched(iter("abc"), 10)) == [("a", "b", "c")]
    assert list(batched((n for n in range(4)), 2)) == [(0, 1), (2, 3)]


def test_batched_validates_size_immediately() -> None:
    with pytest.raises(ValueError):
        batched([1, 2, 3], 0)


def test_batched_aggregation_matches_single_pass(tmp_path: Path) -> None:
    path = tmp_path / "loans.jsonl"
    lines: list[object] = [make_row(str(i), loan_amnt=100 + i) for i in range(11)]
    lines += [make_row("3"), make_row("4"), make_row("5")]  # 3 duplicates
    lines += [make_row("bad", loan_amnt=""), "{broken json"]  # 2 invalid
    write_jsonl(path, lines)

    whole = run_pipeline(path)
    in_batches = run_pipeline(path, batch_size=4)

    assert (whole.read, whole.invalid, whole.duplicates, whole.kept) == (16, 2, 3, 11)
    assert whole.batches == 0
    assert in_batches.batches == 3  # 4 + 4 + 3
    assert {**asdict(whole), "batches": 0} == {**asdict(in_batches), "batches": 0}


# --- sources ---------------------------------------------------------------


def test_read_rows_jsonl_survives_blank_and_broken_lines(tmp_path: Path) -> None:
    path = tmp_path / "loans.jsonl"
    path.write_text(
        '{"loan_id": "1"}\n\n   \n{broken\n{"loan_id": "2"}\n', encoding="utf-8"
    )

    rows = list(read_rows_jsonl(path))

    assert rows == [{"loan_id": "1"}, "{broken", {"loan_id": "2"}]


def test_read_rows_jsonl_survives_undecodable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "loans.jsonl"
    path.write_bytes(b'{"loan_id": "1"}\n\xff\xfe\n{"loan_id": "2"}\n')

    assert len(list(read_rows_jsonl(path))) == 3


def test_read_rows_csv_handles_quotes_short_rows_and_footers(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(
        'id,junk,loan_amnt\n1,"multi\nline, with comma",100\n\n2,x\n'
        "Total amount funded: 5,,\n",
        encoding="utf-8",
    )

    rows = list(read_rows_csv(path, {"id": "loan_id", "loan_amnt": "loan_amnt"}))

    assert rows == [
        {"loan_id": "1", "loan_amnt": "100"},  # quoted newline kept in one record
        {"loan_id": "Total amount funded: 5", "loan_amnt": ""},  # rejected later
    ]


def test_read_rows_csv_fails_loudly_on_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        list(read_rows_csv(path, {"missing": "x"}))


# --- CLI -------------------------------------------------------------------


def _parse_report(output: str) -> dict[str, str]:
    pairs = (line.split(":", 1) for line in output.splitlines() if ":" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def test_cli_stats_reports_counters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "loans.jsonl"
    write_jsonl(
        path,
        [
            make_row("1", loan_amnt=1000),
            make_row("2", loan_amnt=3000),
            make_row("2", loan_amnt=3000),  # duplicate
            make_row("3", loan_amnt=""),  # invalid
            "{broken",  # invalid
        ],
    )

    exit_code = main([str(path), "--stats", "--batch-size", "1"])

    report = _parse_report(capsys.readouterr().out)
    assert exit_code == 0
    assert report["Records read"] == "5"
    assert report["Rejected"] == "2"
    assert report["Duplicates"] == "1"
    assert report["Kept"] == "2"
    assert report["Batches"] == "2"
    assert report["Loan amount"].startswith("min=1,000 max=3,000 avg=2,000.00")


def test_cli_preview_prints_n_loans_and_stops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "loans.jsonl"
    write_jsonl(path, (make_row(str(i)) for i in range(50)))

    exit_code = main([str(path), "--preview", "3"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.count("AcceptedLoan(") == 3
    assert "rows read from the file: 3" in out


def test_cli_missing_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "nope.jsonl")]) == 2
    assert "file not found" in capsys.readouterr().err
