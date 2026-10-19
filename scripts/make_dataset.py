"""Build the JSON Lines benchmark file from the raw LendingClub CSV.

The raw CSV (~1.6 GB, 2.26 M lines) is *streamed*, never loaded: only the
first ``--rows`` usable loans are read. Real rows are then mixed with the same
deliberate defects as in the course assignment, so the expected counters are
known in advance:

* every 10th row (``i % 10 == 3``) repeats the previous row   -> duplicates
* every 1000th row (``i % 1000 == 0``) loses its loan amount  -> invalid rows

With the default 200 000 rows that gives 200 invalid, 20 000 duplicates and
179 800 kept loans.

    python -m scripts.make_dataset
    python -m scripts.make_dataset --rows 50000 --output some/other/file.jsonl
"""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from src.ml_model.streaming.models import to_accepted_loan
from src.ml_model.streaming.sources import read_rows_csv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "ml_model" / "data"
DEFAULT_SOURCE = DATA_DIR / "raw" / "accepted_2007_to_2018Q4.csv"
DEFAULT_OUTPUT = DATA_DIR / "interim" / "loans_stream_200k.jsonl"

# CSV header -> field name used in the JSONL file.
CSV_COLUMNS: dict[str, str] = {
    "id": "loan_id",
    "loan_amnt": "loan_amnt",
    "term": "term",
    "int_rate": "int_rate",
    "grade": "grade",
    "addr_state": "addr_state",
    "loan_status": "loan_status",
    "issue_d": "issue_d",
    "annual_inc": "annual_inc",
}


def usable_loans(source: Path) -> Iterator[dict[str, Any]]:
    """Real loans from the raw CSV, one at a time.

    Rows the model rejects (the two "Total amount funded..." footer lines at
    the end of the file, blank lines, ...) are dropped, so the only defects in
    the benchmark file are the ones injected on purpose.
    """
    for raw in read_rows_csv(source, CSV_COLUMNS):
        loan = to_accepted_loan(raw)
        if loan is not None:
            yield loan.model_dump()


def with_defects(
    loans: Iterator[dict[str, Any]], total: int
) -> Iterator[dict[str, Any]]:
    """Yield ``total`` rows: real loans plus injected duplicates and bad rows.

    Raises:
        ValueError: If ``loans`` runs out before ``total`` rows were produced.
    """
    previous: dict[str, Any] | None = None
    for i in range(total):
        if i % 10 == 3 and previous is not None:
            row = dict(previous)  # duplicate of the preceding row
        else:
            fresh = next(loans, None)
            if fresh is None:
                raise ValueError(f"source has only {i} usable loans, need {total}")
            row = fresh
        if i % 1000 == 0:
            row = dict(row, loan_amnt=None)  # corrupted: the amount is missing
        yield row
        previous = row


def write_jsonl(rows: Iterator[dict[str, Any]], output: Path) -> int:
    """Write one JSON object per line; return the number of lines written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: Sequence[str] | None = None) -> int:
    """Build the JSON Lines sample from the raw LendingClub CSV.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on bad data, 2 if the source is missing.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=200_000)
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: raw file not found: {args.source}", file=sys.stderr)
        return 2
    try:
        written = write_jsonl(
            with_defects(usable_loans(args.source), args.rows), args.output
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {written} lines ({size_mb:.1f} MB) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
