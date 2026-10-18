# Streaming pipeline for large loan files

Practical assignment No. 4: *data stream processing, a generator pipeline and `itertools`*.

## Purpose

The raw LendingClub file (`accepted_2007_to_2018Q4.csv`) is about 1.6 GB and 2.26 million
lines; the scoring service must be able to ingest files like that without holding them in
memory. The pipeline reads a file **one record at a time**, throws away broken records
(and counts them), removes duplicates and computes statistics in **a single pass**. The same
stream can later be cut into batches for bulk inserts into PostgreSQL (`src/db/`).

## Where things live

| Course name                       | In this repository                                  |
|-----------------------------------|-----------------------------------------------------|
| `domain/` (`Vacancy`, `to_vacancy`) | `src/ml_model/streaming/models.py` (`AcceptedLoan`, `to_accepted_loan`) |
| `sources/json_file.py`            | `src/ml_model/streaming/sources.py` (`read_rows_jsonl`, `read_rows_csv`) |
| `services/pipeline.py`            | `src/ml_model/streaming/pipeline.py` (`PipelineStats`, `parse_all`, `deduplicate`, `collect`, `batched`) |
| `cli.py`                          | `src/ml_model/streaming/cli.py` (`--stats`, `--preview N`, `--batch-size N`) |
| time / memory measurement         | `src/ml_model/streaming/metrics.py` (`time.perf_counter` + `tracemalloc`) |
| `scripts/make_dataset.py`         | `scripts/make_dataset.py`                           |
| `scripts/compare.py`              | `scripts/compare.py`                                |
| `tests/test_pipeline.py`          | `tests/test_pipeline.py`                            |

The scripts import from `src`, so run them from the repository root with `python -m`.

## How to run (Anaconda Prompt or PowerShell, repository root, one command per line)

```powershell
python -m scripts.make_dataset                 # builds the 200 000-line benchmark file
python -m src.ml_model.streaming.cli src\ml_model\data\interim\loans_stream_200k.jsonl --stats
python -m src.ml_model.streaming.cli src\ml_model\data\interim\loans_stream_200k.jsonl --stats --batch-size 500
python -m src.ml_model.streaming.cli src\ml_model\data\interim\loans_stream_200k.jsonl --preview 5
python -m scripts.compare                      # lazy pipeline vs intermediate lists
python -m pytest -q
```

## Data

`scripts/make_dataset.py` streams the raw CSV (it never loads it) and keeps the first 180 000
usable loans. Rows the model rejects, such as the two `Total amount funded...` footer lines
at the end of the file, are dropped. The remaining 20 000 lines and 200 defects are injected
on purpose so the expected counters are known in advance:

* every 10th line repeats the previous one: **20 000 duplicates**;
* every 1000th line loses its loan amount: **200 invalid rows**;
* therefore 200 000 lines read, **179 800 kept**, 360 batches of 500.

The output goes to `src/ml_model/data/interim/`, which is git-ignored (derived data, the
raw file stays untouched).

## How the pipeline works

```text
read_rows_jsonl -> parse_all -> deduplicate -> collect
   generator       generator    generator      consumer
```

Each stage is a generator that takes an iterable and returns an iterable, so a record passes
through all stages before the next one is read. The file is not opened until `collect`
(or a `for` loop, `islice`, `batched`) asks for the first element.

* **Materialisation point:** there is exactly one, and it is `collect` (inside
  `run_pipeline`). There is no `list()` between the stages.
* **Aggregation in one pass:** `PipelineStats` keeps a few numbers and a `Counter`
  (count, sum, min, max, per-grade counts). The mean is `sum / count`, so no values are
  stored. A median would need all values, which is where laziness stops being free.
* **Batches:** `batched()` is the `islice` equivalent of `itertools.batched`, which only
  exists in Python 3.12+ (the project targets 3.10+). `run_pipeline(path, batch_size=500)`
  aggregates each batch separately; the statistics are identical to a run without batches
  (covered by a test).
* **Preview:** `--preview N` is `islice(pipeline, N)`. It stops after N records, so
  `stats.read` ends up at 7 for N = 5 (one invalid and one duplicate line come first), not
  200 000.
* **Bad data:** a line that is not JSON, a JSON value that is not an object, or an object
  that fails `AcceptedLoan` validation is counted as *rejected* and skipped; blank lines are
  ignored; undecodable bytes are replaced instead of raising. The pipeline never crashes on
  a bad record.

## Measurements

Machine (printed by `scripts/compare.py`): Windows 11 (build 10.0.26300; Python's
`platform` module labels it "Windows-10"), AMD64 Family 25 Model 80 Stepping 0
(AuthenticAMD), 12 logical cores, 15.9 GiB RAM, Python 3.11.15.

File: `loans_stream_200k.jsonl`, 200 000 lines, 35.4 MB. Time includes `tracemalloc`
overhead.

| approach | records kept | time, s | peak memory, MB |
|---|---:|---:|---:|
| generator pipeline | 179800 | 5.09 | 29.0 |
| generators, no dedup | 199800 | 4.81 | 0.9 |
| intermediate lists | 179800 | 6.37 | 447.8 |

Output of `python -m scripts.compare`, as printed:

```text
| approach | records kept | time, s | peak memory, MB |
|---|---:|---:|---:|
| generator pipeline | 179800 | 5.09 | 29.0 |
| generators, no dedup | 199800 | 4.81 | 0.9 |
| intermediate lists | 179800 | 6.37 | 447.8 |
```

Reading the table:

* Both approaches give the same answer (179 800 loans), but the list-based version peaks
  at 447.8 MB against 29.0 MB for the generators, about **15 times more**, and it is
  about 25 % slower (6.37 s against 5.09 s).
* "No dedup" keeps 199 800 records (the 20 000 duplicates stay in, the 200 invalid rows
  are still dropped) and peaks at only 0.9 MB: the generators alone hold one record at a
  time.
* The CLI gives the same figures for the real run: `--stats` took 5.16 s with a peak of
  29.0 MB, and `--preview 5` read only 7 rows (one invalid and one duplicate line come
  first) out of 200 000.

## Why peak memory is not zero

The generators themselves hold one record at a time (the "no dedup" row: 0.9 MB). The rest
of the peak is the `seen` set in `deduplicate`: it has to remember every unique key, so it
grows with the number of **unique** records. For 179 800 unique loans that is 179 800
tuples of two short strings. The measurement agrees: 29.0 MB - 0.9 MB = 28.1 MB, about
164 bytes per key.

The set stores the key, not the `AcceptedLoan` object. Keeping whole records around costs
far more, as the "intermediate lists" row shows (447.8 MB). Without deduplication the peak
falls to 0.9 MB.

If the number of unique records reaches the millions, deduplication should move to the
database (a unique constraint with `INSERT ... ON CONFLICT DO NOTHING`).

## Checks

```powershell
python -m pytest -q
python -m mypy src\ml_model\streaming scripts
python -m ruff check .
```

Screenshots of `--stats` and `--preview 5` are attached to the pull request.
