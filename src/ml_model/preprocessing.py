"""Custom preprocessing components for the credit-risk scoring pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PercentileClipper(TransformerMixin, BaseEstimator):
    """Winsorize numeric features at percentile bounds learned on training data.

    Bounds are computed per column with missing values ignored, and missing
    values stay missing in the output, so the clipper can sit in front of an
    imputer. Extreme but genuine values (for example ``revol_util`` above 100%)
    are capped rather than removed.

    Attributes:
        lower_bounds_: Per-column lower clipping bounds learned in ``fit``.
        upper_bounds_: Per-column upper clipping bounds learned in ``fit``.
        n_features_in_: Number of columns seen in ``fit``.
        feature_names_in_: Column names seen in ``fit`` (only if ``X`` had them).
    """

    def __init__(self, lower_q: float = 0.001, upper_q: float = 0.999) -> None:
        """Store the quantiles used as clipping bounds.

        Args:
            lower_q: Lower quantile in [0, 1); smaller values are raised to it.
            upper_q: Upper quantile in (0, 1]; larger values are lowered to it.
        """
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: ArrayLike, y: object = None) -> PercentileClipper:
        """Learn the clipping bounds from the training data.

        Args:
            X: 2D numeric feature matrix (DataFrame or array); NaN is allowed.
            y: Ignored; present for scikit-learn API compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If the quantiles are invalid or ``X`` is not 2D.
        """
        if not 0.0 <= self.lower_q < self.upper_q <= 1.0:
            raise ValueError(
                "Quantiles must satisfy 0 <= lower_q < upper_q <= 1, got "
                f"lower_q={self.lower_q}, upper_q={self.upper_q}."
            )
        values: NDArray[np.float64] = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {values.shape}.")

        self.lower_bounds_: NDArray[np.float64] = np.nanquantile(
            values, self.lower_q, axis=0
        )
        self.upper_bounds_: NDArray[np.float64] = np.nanquantile(
            values, self.upper_q, axis=0
        )
        self.n_features_in_: int = values.shape[1]
        column_names = getattr(X, "columns", None)
        if column_names is not None:
            self.feature_names_in_: NDArray[np.object_] = np.asarray(
                column_names, dtype=object
            )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Clip each column to its learned bounds; NaN stays NaN.

        Args:
            X: 2D numeric feature matrix with the same columns as in ``fit``.

        Returns:
            The clipped matrix as a float array.

        Raises:
            ValueError: If ``X`` is not 2D or has a different number of columns.
        """
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        values: NDArray[np.float64] = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected a 2D input with {self.n_features_in_} columns, "
                f"got shape {values.shape}."
            )
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return the output feature names (identical to the input names)."""
        check_is_fitted(self, ["lower_bounds_"])
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
