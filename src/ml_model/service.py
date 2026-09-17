"""Inference service wrapper for the credit scoring pipeline.

Loads a serialized scikit-learn Pipeline (preprocessing + estimator) from
disk and exposes a single method to score one client's raw feature dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

DEFAULT_PIPELINE_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "scoring_pipeline.pkl"
)


class CreditRiskPredictor:
    """Loads a trained credit-scoring pipeline and serves
    default-probability predictions.
    Attributes:
        pipeline: The deserialized scikit-learn Pipeline used for inference.
    """

    def __init__(self, pipeline_path: str | Path = DEFAULT_PIPELINE_PATH) -> None:
        """Load the serialized scoring pipeline from disk.

        Args:
            pipeline_path: Path to the joblib-serialized scikit-learn Pipeline.
                Defaults to `scoring_pipeline.pkl` next to this module.

        Raises:
            FileNotFoundError: If no file exists at `pipeline_path`.
            RuntimeError: If the file exists but fails to deserialize, or the
                deserialized object does not expose `predict_proba`.
        """
        resolved_path = Path(pipeline_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"Scoring pipeline not found at '{resolved_path}'. Train/export "
                "the real pipeline, or run generate_dummy_pipeline.py for a "
                "throwaway stand-in, before instantiating CreditScoringService."
            )

        try:
            loaded_object: Any = joblib.load(resolved_path)
        except Exception as exc:  # noqa: BLE001 - surface any deserialization failure
            raise RuntimeError(
                f"Failed to deserialize pipeline from '{resolved_path}': {exc}"
            ) from exc

        if not hasattr(loaded_object, "predict_proba"):
            raise RuntimeError(
                f"Object loaded from '{resolved_path}' has no 'predict_proba' "
                f"method (got {type(loaded_object).__name__!r})."
            )

        self.pipeline: Pipeline = loaded_object

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Score one client's raw features and return the probability of default.

        Args:
            features: Mapping of raw feature names to raw values for a single
                client, in the same shape the pipeline's preprocessing step
                expects (e.g. `{"annual_inc": 65000.0, "term": "36 months", ...}`).

        Returns:
            The predicted probability of default (class 1) as a float in [0.0, 1.0].

        Raises:
            ValueError: If `features` is empty.
            RuntimeError: If the underlying pipeline fails to produce a prediction.
        """
        if not features:
            raise ValueError("`features` must be a non-empty dict of raw client data.")

        record_frame = pd.DataFrame(features, index=[0])

        try:
            probabilities = self.pipeline.predict_proba(record_frame)
        except Exception as exc:  # noqa: BLE001 - wrap any pipeline-internal failure
            raise RuntimeError(f"Pipeline inference failed: {exc}") from exc

        # Locate the "default" (class 1) column defensively instead of
        # assuming column index 1, in case class ordering ever changes.
        class_labels = list(getattr(self.pipeline, "classes_", [0, 1]))
        positive_class_index = class_labels.index(1) if 1 in class_labels else 1

        probability_of_default = float(probabilities[0, positive_class_index])
        return min(max(probability_of_default, 0.0), 1.0)
