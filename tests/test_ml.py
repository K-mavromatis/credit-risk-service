"""Unit tests for the ML domain: ``CreditRiskPredictor`` and its scoring pipeline.

Sprint 4, Card 2. Exercises the inference service against the *real* trained
artifact (``src/ml_model/scoring_pipeline.pkl``) on valid, extreme, unseen and
missing inputs. The structural tests in ``TestPipelineContract`` also act as a
guard: if the throwaway ``generate_dummy_pipeline.py`` artifact (no
preprocessing) is ever shipped by mistake, they fail loudly instead of letting
the behavioural tests below pass vacuously.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from src.ml_model.service import DEFAULT_PIPELINE_PATH, CreditRiskPredictor

# ---------------------------------------------------------------------------
# Feature contract (mirrors src/api/schemas.py::ApplicationBase)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "loan_amnt",
    "emp_length",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "pub_rec",
    "revol_util",
    "mort_acc",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
)

# Benchmark ("golden") applicant: every value is in-distribution and every
# category is one the encoder saw during training.
BENCHMARK_FEATURES: Final[dict[str, Any]] = {
    "loan_amnt": 15_000.0,
    "term": "36 months",
    "emp_length": 5,
    "home_ownership": "MORTGAGE",
    "annual_inc": 72_000.0,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 18.4,
    "fico_range_low": 690,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_util": 42.3,
    "mort_acc": 1,
}


def _with(**overrides: Any) -> dict[str, Any]:
    """Return a copy of the benchmark applicant with some fields overridden."""
    return {**BENCHMARK_FEATURES, **overrides}


def _assert_valid_probability(value: object) -> None:
    """Assert ``value`` is a finite built-in float inside [0.0, 1.0]."""
    assert isinstance(value, float), f"expected float, got {type(value).__name__}"
    assert math.isfinite(value), f"probability is not finite: {value!r}"
    assert 0.0 <= value <= 1.0, f"probability out of [0, 1]: {value!r}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def predictor() -> CreditRiskPredictor:
    """Load the real trained pipeline once per module (deserialization is I/O)."""
    return CreditRiskPredictor()


@pytest.fixture(scope="module")
def preprocessor(predictor: CreditRiskPredictor) -> ColumnTransformer:
    """The fitted ColumnTransformer step of the loaded pipeline."""
    step = predictor.pipeline.named_steps["preprocessor"]
    assert isinstance(step, ColumnTransformer)
    return step


@pytest.fixture(scope="module")
def numeric_imputer(preprocessor: ColumnTransformer) -> SimpleImputer:
    """The fitted median imputer inside the numeric branch."""
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer)
    return imputer


@pytest.fixture(scope="module")
def one_hot_encoder(preprocessor: ColumnTransformer) -> OneHotEncoder:
    """The fitted OneHotEncoder inside the categorical branch."""
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    assert isinstance(encoder, OneHotEncoder)
    return encoder


# ---------------------------------------------------------------------------
# 1. Smoke test: initialization and pipeline loading
# ---------------------------------------------------------------------------
class TestInitialization:
    def test_default_artifact_exists_on_disk(self) -> None:
        assert DEFAULT_PIPELINE_PATH.is_file(), (
            f"Trained pipeline missing at {DEFAULT_PIPELINE_PATH}; "
            "export it from notebooks/03_model_benchmarking.ipynb."
        )

    def test_predictor_loads_fitted_pipeline(
        self, predictor: CreditRiskPredictor
    ) -> None:
        assert isinstance(predictor, CreditRiskPredictor)
        assert isinstance(predictor.pipeline, Pipeline)
        assert callable(getattr(predictor.pipeline, "predict_proba", None))
        # Raises NotFittedError if the final estimator was never trained.
        check_is_fitted(predictor.pipeline)

    def test_pipeline_is_binary_default_classifier(
        self, predictor: CreditRiskPredictor
    ) -> None:
        assert list(predictor.pipeline.classes_) == [0, 1]

    def test_accepts_explicit_path(self) -> None:
        service = CreditRiskPredictor(pipeline_path=str(DEFAULT_PIPELINE_PATH))
        assert isinstance(service.pipeline, Pipeline)


class TestPipelineContract:
    """Guards that make the behavioural tests below meaningful."""

    def test_has_preprocessor_then_classifier(
        self, predictor: CreditRiskPredictor
    ) -> None:
        assert list(predictor.pipeline.named_steps) == ["preprocessor", "classifier"]

    def test_preprocessor_expects_the_api_feature_contract(
        self, preprocessor: ColumnTransformer
    ) -> None:
        expected = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
        assert set(preprocessor.feature_names_in_) == expected

    def test_numeric_imputer_uses_median(self, numeric_imputer: SimpleImputer) -> None:
        assert numeric_imputer.strategy == "median"
        assert numeric_imputer.statistics_.shape == (len(NUMERIC_FEATURES),)
        assert np.isfinite(numeric_imputer.statistics_).all()

    def test_encoder_ignores_unknown_categories(
        self, one_hot_encoder: OneHotEncoder
    ) -> None:
        assert one_hot_encoder.handle_unknown == "ignore"


# ---------------------------------------------------------------------------
# 2. predict_proba(): valid and extreme inputs
# ---------------------------------------------------------------------------
class TestPredictProbaValidInput:
    @pytest.mark.parametrize(
        "features",
        [
            pytest.param(BENCHMARK_FEATURES, id="benchmark"),
            pytest.param(
                _with(
                    fico_range_low=810,
                    annual_inc=250_000.0,
                    dti=3.0,
                    revol_util=5.0,
                    term="36 months",
                    verification_status="Source Verified",
                ),
                id="prime-borrower",
            ),
            pytest.param(
                _with(
                    fico_range_low=620,
                    annual_inc=22_000.0,
                    dti=38.0,
                    revol_util=97.0,
                    pub_rec=2,
                    term="60 months",
                    home_ownership="RENT",
                    purpose="small_business",
                ),
                id="subprime-borrower",
            ),
        ],
    )
    def test_returns_float_in_unit_interval(
        self, predictor: CreditRiskPredictor, features: dict[str, Any]
    ) -> None:
        _assert_valid_probability(predictor.predict_proba(features))

    def test_is_deterministic(self, predictor: CreditRiskPredictor) -> None:
        first = predictor.predict_proba(BENCHMARK_FEATURES)
        second = predictor.predict_proba(BENCHMARK_FEATURES)
        assert first == second

    def test_does_not_mutate_input(self, predictor: CreditRiskPredictor) -> None:
        features = dict(BENCHMARK_FEATURES)
        predictor.predict_proba(features)
        assert features == BENCHMARK_FEATURES

    def test_ignores_extra_columns(self, predictor: CreditRiskPredictor) -> None:
        # Older payloads carried int_rate/installment; they must be dropped.
        enriched = _with(int_rate=13.56, installment=456.78)
        assert predictor.predict_proba(enriched) == pytest.approx(
            predictor.predict_proba(BENCHMARK_FEATURES)
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"annual_inc": 1e9}, id="billionaire-income"),
            pytest.param({"annual_inc": 1.0}, id="near-zero-income"),
            pytest.param({"loan_amnt": 0.0}, id="zero-loan"),
            pytest.param({"loan_amnt": 1e7}, id="huge-loan"),
            pytest.param({"dti": 999.0}, id="extreme-dti"),
            pytest.param({"revol_util": 500.0}, id="revol-util-over-100pct"),
            pytest.param({"fico_range_low": 300}, id="fico-floor"),
            pytest.param({"fico_range_low": 850}, id="fico-ceiling"),
            pytest.param({"pub_rec": 50, "open_acc": 0}, id="many-public-records"),
            pytest.param({"emp_length": 0, "mort_acc": 0}, id="all-zero-counts"),
        ],
    )
    def test_extreme_values_stay_in_unit_interval(
        self, predictor: CreditRiskPredictor, overrides: dict[str, Any]
    ) -> None:
        _assert_valid_probability(predictor.predict_proba(_with(**overrides)))


# ---------------------------------------------------------------------------
# 3. predict_proba(): unknown categories -> OneHotEncoder(handle_unknown="ignore")
# ---------------------------------------------------------------------------
UNKNOWN_CATEGORY_CASES: Final[list[Any]] = [
    pytest.param("purpose", "space_travel", id="purpose=space_travel"),
    pytest.param("home_ownership", "CASTLE", id="home_ownership=CASTLE"),
    pytest.param("term", "120 months", id="term=120-months"),
    pytest.param("verification_status", "Blockchain", id="verification=Blockchain"),
]


class TestPredictProbaUnknownCategories:
    @pytest.mark.parametrize(("feature", "unseen_value"), UNKNOWN_CATEGORY_CASES)
    def test_unknown_category_does_not_raise(
        self, predictor: CreditRiskPredictor, feature: str, unseen_value: str
    ) -> None:
        # Would raise KeyError/ValueError -> RuntimeError if handle_unknown="error".
        probability = predictor.predict_proba(_with(**{feature: unseen_value}))
        _assert_valid_probability(probability)

    @pytest.mark.parametrize(("feature", "unseen_value"), UNKNOWN_CATEGORY_CASES)
    def test_unknown_category_encodes_to_all_zeros(
        self,
        preprocessor: ColumnTransformer,
        one_hot_encoder: OneHotEncoder,
        feature: str,
        unseen_value: str,
    ) -> None:
        feature_index = CATEGORICAL_FEATURES.index(feature)
        assert unseen_value not in set(one_hot_encoder.categories_[feature_index])

        frame = pd.DataFrame(_with(**{feature: unseen_value}), index=[0])
        encoded = np.asarray(preprocessor.transform(frame))
        output_names = list(preprocessor.get_feature_names_out())
        block = [
            i for i, n in enumerate(output_names) if n.startswith(f"cat__{feature}_")
        ]

        assert block, f"no one-hot columns found for {feature!r}"
        assert encoded[0, block].sum() == 0.0

    def test_known_category_sets_exactly_one_flag(
        self, preprocessor: ColumnTransformer
    ) -> None:
        frame = pd.DataFrame(BENCHMARK_FEATURES, index=[0])
        encoded = np.asarray(preprocessor.transform(frame))
        output_names = list(preprocessor.get_feature_names_out())
        for feature in CATEGORICAL_FEATURES:
            block = [
                i
                for i, n in enumerate(output_names)
                if n.startswith(f"cat__{feature}_")
            ]
            assert encoded[0, block].sum() == 1.0, feature


# ---------------------------------------------------------------------------
# 4. predict_proba(): missing numeric values -> median imputation
# ---------------------------------------------------------------------------
class TestPredictProbaMissingValues:
    @pytest.mark.parametrize("feature", NUMERIC_FEATURES)
    @pytest.mark.parametrize("missing", [None, np.nan], ids=["None", "NaN"])
    def test_single_missing_numeric_field_is_scored(
        self, predictor: CreditRiskPredictor, feature: str, missing: float | None
    ) -> None:
        _assert_valid_probability(predictor.predict_proba(_with(**{feature: missing})))

    def test_all_numeric_fields_missing_is_scored(
        self, predictor: CreditRiskPredictor
    ) -> None:
        features = _with(**dict.fromkeys(NUMERIC_FEATURES))
        _assert_valid_probability(predictor.predict_proba(features))

    @pytest.mark.parametrize("feature", NUMERIC_FEATURES)
    def test_missing_value_is_replaced_by_training_median(
        self, numeric_imputer: SimpleImputer, feature: str
    ) -> None:
        frame = pd.DataFrame(_with(**{feature: None}), index=[0])
        imputed = numeric_imputer.transform(frame[list(NUMERIC_FEATURES)])
        column = NUMERIC_FEATURES.index(feature)

        assert not np.isnan(imputed).any()
        assert imputed[0, column] == pytest.approx(numeric_imputer.statistics_[column])

    def test_imputing_median_equals_passing_median(
        self, predictor: CreditRiskPredictor, numeric_imputer: SimpleImputer
    ) -> None:
        # End-to-end: a missing field must score exactly like its median value.
        column = NUMERIC_FEATURES.index("annual_inc")
        median_income = float(numeric_imputer.statistics_[column])
        assert predictor.predict_proba(_with(annual_inc=None)) == pytest.approx(
            predictor.predict_proba(_with(annual_inc=median_income))
        )

    def test_preprocessed_matrix_has_no_nans(
        self, preprocessor: ColumnTransformer
    ) -> None:
        frame = pd.DataFrame(_with(**dict.fromkeys(NUMERIC_FEATURES)), index=[0])
        assert not np.isnan(np.asarray(preprocessor.transform(frame))).any()

    @pytest.mark.parametrize("feature", CATEGORICAL_FEATURES)
    def test_missing_categorical_field_is_scored(
        self, predictor: CreditRiskPredictor, feature: str
    ) -> None:
        # Categorical branch imputes with the most frequent training level.
        _assert_valid_probability(predictor.predict_proba(_with(**{feature: None})))


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_empty_features_raise_value_error(
        self, predictor: CreditRiskPredictor
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            predictor.predict_proba({})

    def test_missing_artifact_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            CreditRiskPredictor(tmp_path / "does_not_exist.pkl")

    def test_corrupt_artifact_raises_runtime_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_bytes(b"definitely not a pickle")
        with pytest.raises(RuntimeError, match="Failed to deserialize"):
            CreditRiskPredictor(corrupt)

    def test_artifact_without_predict_proba_raises_runtime_error(
        self, tmp_path: Path
    ) -> None:
        not_a_model = tmp_path / "not_a_model.pkl"
        joblib.dump({"weights": [1, 2, 3]}, not_a_model)
        with pytest.raises(RuntimeError, match="predict_proba"):
            CreditRiskPredictor(not_a_model)

    def test_pipeline_failure_is_wrapped_in_runtime_error(
        self, predictor: CreditRiskPredictor
    ) -> None:
        # Dropping a required column makes ColumnTransformer raise internally.
        incomplete = {k: v for k, v in BENCHMARK_FEATURES.items() if k != "purpose"}
        with pytest.raises(RuntimeError, match="Pipeline inference failed"):
            predictor.predict_proba(incomplete)
