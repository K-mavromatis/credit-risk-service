"""Row sources: lazy readers that hold at most one record in memory.

Both readers are generators. They open the file on the first ``next()`` and
keep it open until they are exhausted or closed; the ``with`` block closes it
either way.
"""

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path


def read_rows_jsonl(path: Path) -> Iterator[object]:
    """Yield one decoded JSON value per non-blank line of a JSON Lines file.

    A line that is not valid JSON does not stop the stream. It is yielded as
    the raw string instead, so the parsing stage rejects it *and counts it*
    rather than losing it silently. Undecodable bytes are replaced rather than
    raising ``UnicodeDecodeError`` half-way through a multi-hundred-MB file.
    """
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value: object = json.loads(text)
            except json.JSONDecodeError:
                value = text
            yield value


def read_rows_csv(path: Path, columns: Mapping[str, str]) -> Iterator[dict[str, str]]:
    """Yield one dict per CSV record, keeping only the requested columns.

    Args:
        path: CSV file with a header row.
        columns: Maps a CSV header name to the key used in the yielded dict.

    Raises:
        ValueError: If a requested column is missing from the header. This is a
            configuration error, so it fails loudly instead of skipping rows.

    Records too short to contain every requested column (blank lines,
    truncated lines) are skipped. The ``csv`` module copes with quoted fields
    that contain commas or line breaks, which ``line.split(",")`` would not.
    Footer or garbage lines that *are* long enough come through as ordinary
    rows and are rejected later by validation.
    """
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return
        positions = {name: header.index(name) for name in columns}
        needed = max(positions.values(), default=-1) + 1
        for record in reader:
            if len(record) < needed:
                continue
            yield {key: record[positions[name]] for name, key in columns.items()}
