"""Time and peak-memory measurement shared by the CLI and ``scripts/compare.py``."""

import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

_BYTES_IN_MB = 1024 * 1024


@dataclass(slots=True)
class Metrics:
    """Filled in when the ``measure()`` block exits."""

    elapsed_s: float = 0.0
    peak_mb: float = 0.0


@contextmanager
def measure() -> Iterator[Metrics]:
    """Measure wall-clock time and peak Python memory of the enclosed block.

    ``tracemalloc`` counts only memory allocated by Python itself (exactly what
    we want to compare) and behaves the same on Windows, Linux and macOS,
    unlike a system task manager. It also slows the code down, so absolute
    times are inflated; only compare approaches measured the same way.
    ``peak_mb`` uses 1 MB = 1024 * 1024 bytes.
    """
    metrics = Metrics()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.elapsed_s = time.perf_counter() - started
        metrics.peak_mb = tracemalloc.get_traced_memory()[1] / _BYTES_IN_MB
        tracemalloc.stop()
