"""Compare the lazy generator pipeline with an eager, list-based version.

Runs the same job three ways on the same file and prints a Markdown table
(plus the machine description) ready to paste into ``PIPELINE.md``:

* generator pipeline      - the real pipeline, one record at a time
* generators, no dedup    - same, minus the ``seen`` set (shows where the
                            remaining memory of the real pipeline comes from)
* intermediate lists      - every stage materialised into a list

    python -m scripts.compare
    python -m scripts.compare --path some/other/file.jsonl
"""

import argparse
import ctypes
import os
import platform
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.make_dataset import DEFAULT_OUTPUT
from src.ml_model.streaming.metrics import measure
from src.ml_model.streaming.models import AcceptedLoan, to_accepted_loan
from src.ml_model.streaming.pipeline import (
    PipelineStats,
    build_pipeline,
    collect,
    parse_all,
)
from src.ml_model.streaming.sources import read_rows_jsonl


def lazy(path: Path) -> int:
    """The real pipeline: nothing is ever materialised."""
    stats = PipelineStats()
    collect(build_pipeline(path, stats), stats)
    return stats.kept


def lazy_without_dedup(path: Path) -> int:
    """Generators only; no ``seen`` set, so no state that grows with the data."""
    stats = PipelineStats()
    collect(parse_all(read_rows_jsonl(path), stats), stats)
    return stats.kept


def greedy(path: Path) -> int:
    """Same result, but every stage builds a full list first."""
    rows = list(read_rows_jsonl(path))  # everything in memory
    parsed = [to_accepted_loan(row) for row in rows]
    valid = [loan for loan in parsed if loan is not None]
    seen: set[tuple[str, str]] = set()
    unique: list[AcceptedLoan] = []
    for loan in valid:
        if loan.key not in seen:
            seen.add(loan.key)
            unique.append(loan)
    return len(unique)


def total_ram_gib() -> float | None:
    """Installed RAM in GiB, or ``None`` if it cannot be determined."""
    try:
        if sys.platform == "win32":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.ullTotalPhys) / 1024**3
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / 1024**3
    except (OSError, ValueError, AttributeError):
        return None


def describe_machine() -> str:
    """Return a one-line hardware/OS summary for the benchmark report."""
    ram = total_ram_gib()
    ram_text = f"{ram:.1f} GiB RAM" if ram is not None else "RAM: fill in manually"
    return (
        f"{platform.platform()}; {platform.processor() or 'CPU: fill in manually'}; "
        f"{os.cpu_count()} logical cores; {ram_text}; "
        f"Python {platform.python_version()}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Benchmark the streaming approaches and print a Markdown table.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 2 if the input file is missing.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    path: Path = args.path
    if not path.is_file():
        print(
            f"error: {path} not found - run `python -m scripts.make_dataset` first",
            file=sys.stderr,
        )
        return 2

    approaches: list[tuple[str, Callable[[Path], int]]] = [
        ("generator pipeline", lazy),
        ("generators, no dedup", lazy_without_dedup),
        ("intermediate lists", greedy),
    ]
    print(f"Machine: {describe_machine()}")
    print(f"File: {path.name}, {path.stat().st_size / (1024 * 1024):.1f} MB\n")
    print("| approach | records kept | time, s | peak memory, MB |")
    print("|---|---:|---:|---:|")
    kept_by_name: dict[str, int] = {}
    for name, function in approaches:
        with measure() as metrics:
            kept = function(path)
        kept_by_name[name] = kept
        print(f"| {name} | {kept} | {metrics.elapsed_s:.2f} | {metrics.peak_mb:.1f} |")

    if kept_by_name["generator pipeline"] != kept_by_name["intermediate lists"]:
        print("error: lazy and list-based results differ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
