"""Evaluate the saved scoring pipeline on the held-out test set (Sprint 5, Card 5).

Loads ``src/ml_model/scoring_pipeline.pkl`` exactly the way the API does (via
``CreditRiskPredictor``), scores ``loans_test.parquet`` and prints a Markdown
metrics table: ROC-AUC, PR-AUC, and Precision / Recall / F1 at the decision
threshold the API uses (``RISK_THRESHOLD``, default 0.5). Precision and Recall
are therefore the ones of the ``approved`` / ``rejected`` decision returned by
``POST /predict``, not of some other cut-off.

With ``--update-readme`` the table replaces the block between the
``<!-- METRICS:START -->`` and ``<!-- METRICS:END -->`` markers in README.md,
so the published numbers always come from the committed model.

    python -m scripts.evaluate_model
    python -m scripts.evaluate_model --update-readme
    python -m scripts.evaluate_model --threshold 0.35

Needs the local test split produced by notebook 01
(``src/ml_model/data/processed/loans_test.parquet``, git-ignored).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.api.schemas import parse_risk_threshold
from src.ml_model.service import DEFAULT_PIPELINE_PATH, CreditRiskPredictor

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH: Final[Path] = (
    PROJECT_ROOT / "src" / "ml_model" / "data" / "processed" / "loans_test.parquet"
)
README_PATH: Final[Path] = PROJECT_ROOT / "README.md"
TARGET: Final[str] = "is_default"
POSITIVE_CLASS: Final[int] = 1
START_MARKER: Final[str] = "<!-- METRICS:START -->"
END_MARKER: Final[str] = "<!-- METRICS:END -->"
DESCRIPTION: Final[str] = "Evaluate scoring_pipeline.pkl on the held-out test set."


class ModelMetrics(BaseModel):
    """Test-set metrics of one scoring pipeline."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    n_rows: int = Field(gt=0)
    default_rate: float = Field(ge=0, le=1)
    threshold: float = Field(gt=0, lt=1)
    roc_auc: float = Field(ge=0, le=1)
    pr_auc: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


def evaluate(pipeline_path: Path, test_path: Path, threshold: float) -> ModelMetrics:
    """Score the test split with the saved pipeline and compute the metrics.

    Args:
        pipeline_path: Path to the serialized scikit-learn Pipeline.
        test_path: Parquet file with the 13 raw features and ``is_default``.
        threshold: Probability of default at or above which a loan is rejected.

    Returns:
        The computed metrics.

    Raises:
        FileNotFoundError: If the test split does not exist.
        KeyError: If the test split has no ``is_default`` column.
    """
    if not test_path.is_file():
        raise FileNotFoundError(
            f"Test split not found at '{test_path}'. Run "
            "notebooks/01_data_preparation.ipynb first."
        )
    test_df: pd.DataFrame = pd.read_parquet(test_path)
    if TARGET not in test_df.columns:
        raise KeyError(f"Column '{TARGET}' is missing from '{test_path}'.")

    y_true: np.ndarray = test_df[TARGET].to_numpy(dtype=int)
    features: pd.DataFrame = test_df.drop(columns=[TARGET])

    pipeline = CreditRiskPredictor(pipeline_path).pipeline
    class_labels: list[int] = [int(label) for label in pipeline.classes_]
    probabilities: np.ndarray = pipeline.predict_proba(features)[
        :, class_labels.index(POSITIVE_CLASS)
    ]
    # Same rule as src.api.schemas.decide: rejected if p >= threshold.
    y_pred: np.ndarray = (probabilities >= threshold).astype(int)

    return ModelMetrics(
        model_name=type(pipeline.steps[-1][1]).__name__,
        n_rows=len(test_df),
        default_rate=float(y_true.mean()),
        threshold=threshold,
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def to_markdown(metrics: ModelMetrics) -> str:
    """Render the metrics as the README table (with its caption).

    Args:
        metrics: Metrics returned by ``evaluate``.

    Returns:
        A Markdown snippet without the surrounding markers.
    """
    return (
        f"| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |\n"
        f"|---|---:|---:|---:|---:|---:|\n"
        f"| **{metrics.model_name}** (`scoring_pipeline.pkl`) "
        f"| {metrics.roc_auc:.3f} | {metrics.pr_auc:.3f} "
        f"| {metrics.precision:.3f} | {metrics.recall:.3f} | {metrics.f1:.3f} |\n"
        f"\n"
        f"Test set: {metrics.n_rows:,} loans, default rate "
        f"{metrics.default_rate:.1%}. Precision / Recall / F1 are for the "
        f"*default* class at the API decision threshold "
        f"`RISK_THRESHOLD = {metrics.threshold:g}`. "
        f"Generated by `python -m scripts.evaluate_model --update-readme`."
    )


def update_readme(readme_path: Path, table: str) -> None:
    """Replace the text between the metrics markers in the README.

    Args:
        readme_path: README file to rewrite in place.
        table: New Markdown content for the block.

    Raises:
        ValueError: If either marker is missing or they are out of order.
    """
    text: str = readme_path.read_text(encoding="utf-8")
    start: int = text.find(START_MARKER)
    end: int = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"'{readme_path}' must contain {START_MARKER} ... {END_MARKER}."
        )
    head: str = text[: start + len(START_MARKER)]
    tail: str = text[end:]
    readme_path.write_text(f"{head}\n{table}\n{tail}", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument(
        "--threshold",
        type=str,
        default=os.getenv("RISK_THRESHOLD"),
        help="Decision threshold in (0, 1); defaults to RISK_THRESHOLD or 0.5.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Write the table into README.md between the METRICS markers.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and print (and optionally publish) the table.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    threshold: float = parse_risk_threshold(args.threshold)
    metrics = evaluate(args.pipeline, args.test_data, threshold)
    table: str = to_markdown(metrics)
    print(table)
    if args.update_readme:
        update_readme(README_PATH, table)
        print(f"\nREADME updated: {README_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
