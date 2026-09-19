"""Throwaway pipeline generator — unblocks Card 4 while the real
scikit-learn preprocessing pipeline is being fixed.

Trains a trivial DummyClassifier (ignores feature values entirely, predicts
from the class prior) wrapped in a Pipeline, and serializes it to
`scoring_pipeline.pkl` so CreditScoringService has something to load.

DO NOT use the resulting artifact for anything but plumbing/smoke tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

# Mirrors the raw feature contract from notebooks/01_data_preparation.ipynb.
NUMERICAL_FEATURES: Final[list[str]] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "emp_length",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "pub_rec",
    "revol_util",
    "mort_acc",
]
CATEGORICAL_FEATURES: Final[list[str]] = [
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
]
OUTPUT_PATH: Final[Path] = Path(__file__).resolve().parent / "scoring_pipeline.pkl"
RANDOM_SEED: Final[int] = 42


def _build_dummy_training_frame(
    n_samples: int = 200,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a tiny synthetic dataset shaped like the real raw feature set."""
    rng = np.random.default_rng(RANDOM_SEED)

    features = pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1_000, 40_000, n_samples),
            "int_rate": rng.uniform(5, 30, n_samples),
            "installment": rng.uniform(50, 1_500, n_samples),
            "emp_length": rng.integers(0, 11, n_samples),
            "annual_inc": rng.uniform(20_000, 200_000, n_samples),
            "dti": rng.uniform(0, 40, n_samples),
            "fico_range_low": rng.uniform(620, 800, n_samples),
            "open_acc": rng.integers(1, 20, n_samples),
            "pub_rec": rng.integers(0, 3, n_samples),
            "revol_util": rng.uniform(0, 100, n_samples),
            "mort_acc": rng.integers(0, 5, n_samples),
            "term": rng.choice(["36 months", "60 months"], n_samples),
            "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN"], n_samples),
            "verification_status": rng.choice(
                ["Verified", "Not Verified", "Source Verified"], n_samples
            ),
            "purpose": rng.choice(
                ["debt_consolidation", "credit_card", "car"], n_samples
            ),
        }
    )
    target = rng.integers(0, 2, n_samples)  # 0 = repaid, 1 = default
    return features[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], target


def main() -> None:
    """Train the dummy pipeline and serialize it to OUTPUT_PATH."""
    features, target = _build_dummy_training_frame()

    # A trivial classifier: predicts from the class prior, ignores X entirely.
    # No preprocessing step is needed since DummyClassifier never inspects
    # feature values, so raw mixed-dtype input (numeric + categorical strings)
    # passes straight through without error.
    dummy_pipeline = Pipeline(
        steps=[
            ("classifier", DummyClassifier(strategy="prior", random_state=RANDOM_SEED))
        ]
    )
    dummy_pipeline.fit(features, target)

    try:
        joblib.dump(dummy_pipeline, OUTPUT_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write pipeline to '{OUTPUT_PATH}': {exc}"
        ) from exc

    print(f"Dummy pipeline saved to: {OUTPUT_PATH}")
    print(f"Classes: {dummy_pipeline.classes_}")


if __name__ == "__main__":
    main()
