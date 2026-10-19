"""Command line entry point.

python -m src.ml_model.streaming.cli <file.jsonl> --stats
python -m src.ml_model.streaming.cli <file.jsonl> --stats --batch-size 500
python -m src.ml_model.streaming.cli <file.jsonl> --preview 5
"""

import argparse
import sys
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

from src.ml_model.streaming.metrics import Metrics, measure
from src.ml_model.streaming.pipeline import (
    PipelineStats,
    build_pipeline,
    run_pipeline,
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the streaming CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ml_model.streaming.cli",
        description="Stream a JSON Lines file of accepted loans without loading it.",
    )
    parser.add_argument("path", type=Path, help="JSON Lines file with loan records")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stats",
        action="store_true",
        help="aggregate the whole file in one pass (default mode)",
    )
    mode.add_argument(
        "--preview",
        type=_positive_int,
        metavar="N",
        help="print the first N valid loans without reading the rest of the file",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        metavar="N",
        help="with --stats: aggregate in batches of N records",
    )
    return parser


def _print_stats(stats: PipelineStats, metrics: Metrics) -> None:
    print(f"Records read:  {stats.read}")
    print(f"Rejected:      {stats.invalid}")
    print(f"Duplicates:    {stats.duplicates}")
    print(f"Kept:          {stats.kept}")
    if stats.batches:
        print(f"Batches:       {stats.batches}")
    if (
        stats.amount_min is not None
        and stats.amount_max is not None
        and stats.amount_avg is not None
    ):
        print(
            f"Loan amount:   min={stats.amount_min:,.0f} "
            f"max={stats.amount_max:,.0f} avg={stats.amount_avg:,.2f}"
        )
    if stats.income_avg is not None:
        print(
            f"Annual income: avg={stats.income_avg:,.2f} "
            f"(reported for {stats.income_count} loans)"
        )
    for grade, count in sorted(stats.by_grade.items()):
        print(f"  grade {grade}: {count}")
    print(f"Time: {metrics.elapsed_s:.2f} s, peak memory: {metrics.peak_mb:.1f} MB")


def _stats(path: Path, batch_size: int | None) -> int:
    with measure() as metrics:
        stats = run_pipeline(path, batch_size=batch_size)
    _print_stats(stats, metrics)
    return 0


def _preview(path: Path, count: int) -> int:
    stats = PipelineStats()
    pipeline = build_pipeline(path, stats)
    for loan in islice(pipeline, count):
        print(repr(loan))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(
        f"... rows read from the file: {stats.read} "
        f"(file is {size_mb:.1f} MB, the rest was never touched)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the streaming CLI in ``--stats`` or ``--preview`` mode.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 2 if the input file is missing.
    """
    args = build_parser().parse_args(argv)
    path: Path = args.path
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2
    if args.preview is not None:
        return _preview(path, args.preview)
    return _stats(path, args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
