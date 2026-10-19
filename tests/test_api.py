"""HTTP integration tests through TestClient and mocks (Sprint 4, Card 3).

The requests go through the real FastAPI app: routing, Pydantic validation, the
lifespan handler, the real ``get_predictor`` dependency and the real CRUD layer.
Only the two expensive or external parts are replaced:

* **The model.** ``unittest.mock.patch`` replaces ``CreditRiskPredictor.__init__``
  (so ``scoring_pipeline.pkl`` is never read) and ``predict_proba`` (so it
  returns ``MOCK_PROBABILITY`` at once, with no sklearn inference).
* **The database.** ``get_db`` is overridden with a session on in-memory
  SQLite, so no PostgreSQL server is needed and nothing is written to disk.

``tests/test_api_main.py`` covers the same endpoint with hand-written fakes and
``dependency_overrides``; this module checks the API contract from the client's
side with ``unittest.mock`` and the app's real startup path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final
from unittest.mock import ANY, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, Table, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import API_VERSION, app
from src.api.schemas import ApplicationBase
from src.db.database import get_db
from src.db.models import CreditApplication
from src.ml_model.service import CreditRiskPredictor

TABLE: Final[Table] = CreditApplication.__table__  # type: ignore[assignment]

MOCK_PROBABILITY: Final[float] = 0.25
TEST_RISK_THRESHOLD: Final[str] = "0.5"

VALID_PAYLOAD: Final[dict[str, Any]] = {
    "loan_amnt": 10_000.0,
    "term": "36 months",
    "purpose": "debt_consolidation",
    "annual_inc": 60_000.0,
    "dti": 18.0,
    "emp_length": 5,
    "home_ownership": "MORTGAGE",
    "verification_status": "Source Verified",
    "fico_range_low": 700,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_util": 50.0,
    "mort_acc": 1,
}

# The 13 echoed features + what the service adds.
EXPECTED_RESPONSE_KEYS: Final[frozenset[str]] = frozenset(
    {
        *ApplicationBase.model_fields,
        "id",
        "default_probability",
        "decision",
        "created_at",
    }
)


def _payload(**overrides: Any) -> dict[str, Any]:
    """Return a copy of ``VALID_PAYLOAD`` with some fields replaced."""
    return {**VALID_PAYLOAD, **overrides}


def _count_rows(engine: Engine) -> int:
    """Count the applications stored in the test database."""
    with Session(engine) as db:
        return db.scalar(select(func.count()).select_from(CreditApplication)) or 0


# --- Fixtures -------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fixed_risk_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin RISK_THRESHOLD so a developer's ``.env`` cannot change the results."""
    monkeypatch.setenv("RISK_THRESHOLD", TEST_RISK_THRESHOLD)


@pytest.fixture
def db_engine() -> Iterator[Engine]:
    """In-memory SQLite shared by all threads (TestClient runs a thread pool)."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TABLE.create(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def mock_predict_proba() -> Iterator[MagicMock]:
    """Patch model loading and inference on the real ``CreditRiskPredictor``.

    ``__init__`` is patched so the lifespan can build a predictor without the
    ``.pkl`` file. ``predict_proba`` is patched with ``autospec=True`` so the
    mock keeps the real signature (a call with wrong arguments fails) and
    records ``self`` as its first argument.

    Yields:
        The ``predict_proba`` mock; tests may change its ``return_value``.
    """
    with (
        patch.object(CreditRiskPredictor, "__init__", autospec=True, return_value=None),
        patch.object(
            CreditRiskPredictor,
            "predict_proba",
            autospec=True,
            return_value=MOCK_PROBABILITY,
        ) as mocked,
    ):
        yield mocked


@pytest.fixture
def client(db_engine: Engine, mock_predict_proba: MagicMock) -> Iterator[TestClient]:
    """TestClient with the lifespan running, the model mocked and SQLite as the DB.

    ``with TestClient(app)`` runs the startup code, so the predictor is created
    by the app itself and reached through the real ``get_predictor`` dependency.
    """
    testing_session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


# --- GET /health ----------------------------------------------------------------------
def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": API_VERSION,
        "model_loaded": True,
    }


def test_health_needs_neither_model_nor_database() -> None:
    """Without the lifespan and without any override, /health still answers 200."""
    app.dependency_overrides.clear()

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# --- POST /predict: valid request -----------------------------------------------------
def test_predict_valid_payload_returns_200_with_expected_structure(
    client: TestClient,
) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert set(body) == EXPECTED_RESPONSE_KEYS
    assert body["default_probability"] == pytest.approx(MOCK_PROBABILITY)
    assert body["decision"] == "approved"
    assert isinstance(body["id"], int)
    assert body["created_at"]


def test_predict_calls_mocked_model_once_with_validated_features(
    client: TestClient, mock_predict_proba: MagicMock
) -> None:
    client.post("/predict", json=VALID_PAYLOAD)

    mock_predict_proba.assert_called_once_with(ANY, VALID_PAYLOAD)
    predictor_used = mock_predict_proba.call_args.args[0]
    assert isinstance(predictor_used, CreditRiskPredictor)
    assert predictor_used is app.state.predictor  # the instance loaded at startup


def test_real_pipeline_is_never_loaded(client: TestClient) -> None:
    """The patched __init__ never sets ``pipeline``: no .pkl was deserialized."""
    assert isinstance(app.state.predictor, CreditRiskPredictor)
    assert not hasattr(app.state.predictor, "pipeline")


def test_predict_stores_the_mocked_score(client: TestClient, db_engine: Engine) -> None:
    body: dict[str, Any] = client.post("/predict", json=VALID_PAYLOAD).json()

    with Session(db_engine) as db:
        stored = db.get(CreditApplication, body["id"])
        assert stored is not None
        assert stored.default_probability == pytest.approx(MOCK_PROBABILITY)


@pytest.mark.parametrize(
    ("probability", "expected_decision"),
    [
        (0.0, "approved"),
        (0.25, "approved"),
        (0.4999, "approved"),
        (0.5, "rejected"),  # at the threshold counts as high risk, like the UI
        (0.91, "rejected"),
        (1.0, "rejected"),
    ],
)
def test_decision_follows_the_risk_threshold(
    client: TestClient,
    mock_predict_proba: MagicMock,
    probability: float,
    expected_decision: str,
) -> None:
    mock_predict_proba.return_value = probability

    body: dict[str, Any] = client.post("/predict", json=VALID_PAYLOAD).json()

    assert body["default_probability"] == pytest.approx(probability)
    assert body["decision"] == expected_decision


def test_decision_uses_risk_threshold_setting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RISK_THRESHOLD", "0.2")

    body: dict[str, Any] = client.post("/predict", json=VALID_PAYLOAD).json()

    assert body["default_probability"] == pytest.approx(MOCK_PROBABILITY)
    assert body["decision"] == "rejected"  # 0.25 >= 0.2


def test_invalid_risk_threshold_stops_startup(
    mock_predict_proba: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RISK_THRESHOLD", "1.5")

    with pytest.raises(ValueError, match="RISK_THRESHOLD"), TestClient(app):
        pass  # pragma: no cover - startup fails before the body runs


# --- POST /predict: Pydantic validation -----------------------------------------------
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("annual_inc", "sixty thousand"),  # the case from the card
        ("annual_inc", "60000"),  # numeric string: strict mode does not coerce it
        ("loan_amnt", "10000"),
        ("dti", None),  # required float sent as null
        ("fico_range_low", "700"),
        ("open_acc", 2.5),  # float for an int field
    ],
)
def test_invalid_types_return_422(
    client: TestClient,
    mock_predict_proba: MagicMock,
    db_engine: Engine,
    field: str,
    bad_value: object,
) -> None:
    response = client.post("/predict", json=_payload(**{field: bad_value}))

    assert response.status_code == 422
    error_locations = [error["loc"] for error in response.json()["detail"]]
    assert ["body", field] in error_locations
    # Validation fails before the endpoint body runs: no inference, no DB write.
    mock_predict_proba.assert_not_called()
    assert _count_rows(db_engine) == 0


def test_non_json_body_returns_422(
    client: TestClient, mock_predict_proba: MagicMock
) -> None:
    response = client.post(
        "/predict",
        content=b"annual_inc=60000",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    mock_predict_proba.assert_not_called()


# --- OpenAPI contract -----------------------------------------------------------------
def test_openapi_documents_health_and_decision(client: TestClient) -> None:
    schema: dict[str, Any] = client.get("/openapi.json").json()

    assert "get" in schema["paths"]["/health"]
    response_schema = schema["components"]["schemas"]["ApplicationResponse"]
    assert "decision" in response_schema["properties"]
    assert response_schema["properties"]["decision"]["enum"] == [
        "approved",
        "rejected",
    ]
