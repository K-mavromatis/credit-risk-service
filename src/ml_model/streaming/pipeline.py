"""Streaming pipeline: every stage is a generator, only ``collect`` consumes.

    read_rows_jsonl -> parse_all -> deduplicate -> collect
        (generator)     (generator)   (generator)    (consumer)

A record travels through all stages one by one, so nothing is read from disk
until the consumer asks for the first element, and no stage ever holds the
whole data set. The *only* state that grows is the ``seen`` set inside
``deduplicate`` (a compact key per unique loan, not the loan itself).
"""

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TypeVar

from src.ml_model.streaming.models import AcceptedLoan, to_accepted_loan
from src.ml_model.streaming.sources import read_rows_jsonl

T = TypeVar("T")


@dataclass(slots=True)
class PipelineStats:
    """Counters filled in by the stages; this is also the pipeline's result.

    Everything here is accumulated incrementally (a few numbers and a counter),
    which is why the mean can be computed in one pass without keeping the
    values. A median or percentiles could not: they need all values.
    """

    read: int = 0
    invalid: int = 0
    duplicates: int = 0
    kept: int = 0
    batches: int = 0
    by_grade: Counter[str] = field(default_factory=Counter)
    amount_sum: float = 0.0
    amount_min: float | None = None
    amount_max: float | None = None
    income_count: int = 0
    income_sum: float = 0.0

    @property
    def amount_avg(self) -> float | None:
        """Mean loan amount over kept loans (every kept loan has an amount)."""
        if not self.kept:
            return None
        return self.amount_sum / self.kept

    @property
    def income_avg(self) -> float | None:
        """Mean annual income over the kept loans that report one."""
        if not self.income_count:
            return None
        return self.income_sum / self.income_count


def parse_all(rows: Iterable[object], stats: PipelineStats) -> Iterator[AcceptedLoan]:
    """Validate raw rows; count every row read and every row rejected."""
    for row in rows:
        stats.read += 1
        loan = to_accepted_loan(row)
        if loan is None:
            stats.invalid += 1
            continue
        yield loan


def deduplicate(
    loans: Iterable[AcceptedLoan], stats: PipelineStats
) -> Iterator[AcceptedLoan]:
    """Drop loans whose key was already seen.

    The set stores the compact key (two short strings), not the model object:
    this stage is the one place where memory grows with the number of unique
    records.
    """
    seen: set[tuple[str, str]] = set()
    for loan in loans:
        if loan.key in seen:
            stats.duplicates += 1
            continue
        seen.add(loan.key)
        yield loan


def collect(loans: Iterable[AcceptedLoan], stats: PipelineStats) -> None:
    """Consume the stream and accumulate the aggregates; keep no loans.

    Safe to call once per batch: the counters simply keep accumulating.
    """
    for loan in loans:
        stats.kept += 1
        stats.by_grade[loan.grade] += 1

        amount = loan.loan_amnt
        stats.amount_sum += amount
        if stats.amount_min is None or amount < stats.amount_min:
            stats.amount_min = amount
        if stats.amount_max is None or amount > stats.amount_max:
            stats.amount_max = amount

        if loan.annual_inc is not None:
            stats.income_count += 1
            stats.income_sum += loan.annual_inc


def batched(items: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """Split a stream into tuples of ``size`` items; the last may be shorter.

    ``itertools.batched`` only exists from Python 3.12 and the project targets
    3.10+, so this is the ``islice`` equivalent. At most one batch is alive at
    a time, and ``size`` is validated immediately, not on the first ``next()``.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    return _batches(iter(items), size)


def _batches(iterator: Iterator[T], size: int) -> Iterator[tuple[T, ...]]:
    while batch := tuple(islice(iterator, size)):
        yield batch


def build_pipeline(path: Path, stats: PipelineStats) -> Iterator[AcceptedLoan]:
    """Chain the lazy stages. Nothing is read until the result is consumed."""
    return deduplicate(parse_all(read_rows_jsonl(path), stats), stats)


def run_pipeline(path: Path, *, batch_size: int | None = None) -> PipelineStats:
    """Process a whole file in a single pass and return the statistics.

    With ``batch_size`` the stream is cut into batches and each batch is
    aggregated separately. That is the shape needed later for bulk database
    inserts (``save_many(batch)``); the resulting statistics are identical.
    """
    stats = PipelineStats()
    pipeline = build_pipeline(path, stats)
    if batch_size is None:
        collect(pipeline, stats)
    else:
        for batch in batched(pipeline, batch_size):
            stats.batches += 1
            collect(batch, stats)
    return stats
