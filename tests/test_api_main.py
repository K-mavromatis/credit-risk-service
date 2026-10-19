"""Tests for the FastAPI app and POST /predict (src/api/main.py).

The real model and PostgreSQL are never used: ``get_predictor`` is replaced by
a fake predictor and ``get_db`` by a session on in-memory SQLite, through
``app.dependency_overrides``.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, Table, create_engine, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import main
from src.api.main import app, get_predictor
from src.api.schemas import ApplicationBase, ApplicationResponse
from src.db.database import get_db
from src.db.models import MODEL_FEATURES, CreditApplication

TABLE: Table = CreditApplication.__table__  # type: ignore[assignment]
FAKE_PROBABILITY: float = 0.137


class FakePredictor:
    """Stands in for CreditRiskPredictor: fixed score, records its inputs."""

    instances: int = 0

    def __init__(self, probability: float = FAKE_PROBABILITY) -> None:
        FakePredictor.instances += 1
        self.probability = probability
        self.calls: list[dict[str, Any]] = []

    def predict_proba(self, features: dict[str, Any]) -> float:
        self.calls.append(features)
        return self.probability


class FailingPredictor(FakePredictor):
    def predict_proba(self, features: dict[str, Any]) -> float:
        raise RuntimeError("Pipeline inference failed: boom")


def _payload(**overrides: Any) -> dict[str, Any]:
    """A valid request body, with optional field overrides."""
    payload: dict[str, Any] = {
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
    payload.update(overrides)
    return payload


# --- Fixtures --------------------------------------------------------------------
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
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(db_engine: Engine, predictor: FakePredictor) -> Iterator[TestClient]:
    """Client with the model and the database replaced by test doubles.

    Used without ``with``, so the lifespan (real .pkl load) does not run.
    """
    testing_session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def override_get_db() -> Generator[Session, None, None]:
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_predictor] = lambda: predictor
    yield TestClient(app)
    app.dependency_overrides.clear()


def _count_rows(engine: Engine) -> int:
    with Session(engine) as db:
        return db.scalar(select(func.count()).select_from(CreditApplication)) or 0


# --- Happy path ------------------------------------------------------------------
def test_app_title() -> None:
    assert app.title == "Credit Scoring API"


def test_predict_returns_model_score(client: TestClient) -> None:
    response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    body = ApplicationResponse.model_validate(response.json())
    assert body.default_probability == pytest.approx(FAKE_PROBABILITY)
    assert body.id >= 1
    assert body.created_at is not None


def test_predict_passes_validated_features_to_model(
    client: TestClient, predictor: FakePredictor
) -> None:
    payload = _payload()
    client.post("/predict", json=payload)

    assert predictor.calls == [payload]
    assert set(predictor.calls[0]) == set(MODEL_FEATURES)


def test_predict_saves_application_to_db(client: TestClient, db_engine: Engine) -> None:
    payload = _payload(emp_length=None, mort_acc=None)
    body = client.post("/predict", json=payload).json()

    with Session(db_engine) as db:
        stored = db.get(CreditApplication, body["id"])
        assert stored is not None
        assert stored.to_features() == payload
        assert stored.default_probability == pytest.approx(FAKE_PROBABILITY)


def test_predict_response_echoes_input_features(client: TestClient) -> None:
    payload = _payload(emp_length=None, mort_acc=None)
    body = client.post("/predict", json=payload).json()

    for name in ApplicationBase.model_fields:
        assert body[name] == payload[name]


def test_each_request_creates_a_new_row(client: TestClient, db_engine: Engine) -> None:
    ids = [client.post("/predict", json=_payload()).json()["id"] for _ in range(3)]

    assert len(set(ids)) == 3
    assert _count_rows(db_engine) == 3


# --- Validation: nothing reaches the model or the DB -------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        {"purpose": "crypto"},  # unknown category
        {"loan_amnt": 100.0},  # below MIN_LOAN_AMOUNT
        {"fico_range_low": "700"},  # strict mode: no str -> int coercion
        {"annual_inc": 0},  # must be > 0
        {"loan_amount": 10_000.0},  # unknown field
    ],
)
def test_invalid_request_is_rejected_before_model_and_db(
    client: TestClient,
    predictor: FakePredictor,
    db_engine: Engine,
    overrides: dict[str, Any],
) -> None:
    assert client.post("/predict", json=_payload(**overrides)).status_code == 422
    assert predictor.calls == []
    assert _count_rows(db_engine) == 0


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = _payload()
    del payload["term"]
    assert client.post("/predict", json=payload).status_code == 422


# --- Failure handling -------------------------------------------------------------
def test_inference_failure_returns_500_and_saves_nothing(
    client: TestClient, db_engine: Engine
) -> None:
    app.dependency_overrides[get_predictor] = lambda: FailingPredictor()

    response = client.post("/predict", json=_payload())

    assert response.status_code == 500
    assert _count_rows(db_engine) == 0


def test_db_failure_returns_503(
    client: TestClient, db_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_create(db: Session, data: Any) -> CreditApplication:
        raise OperationalError("INSERT ...", {}, Exception("connection refused"))

    monkeypatch.setattr(main.crud, "create_application", broken_create)

    response = client.post("/predict", json=_payload())

    assert response.status_code == 503


def test_model_not_loaded_returns_503() -> None:
    """Without the lifespan and without an override there is no model."""
    app.dependency_overrides.clear()
    response = TestClient(app).post("/predict", json=_payload())
    assert response.status_code == 503


# --- Lifespan: the model is loaded once per process -------------------------------
def test_model_is_loaded_once_at_startup(
    db_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakePredictor.instances = 0
    monkeypatch.setattr(main, "CreditRiskPredictor", FakePredictor)
    testing_session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def override_get_db() -> Generator[Session, None, None]:
        with testing_session() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:  # `with` runs the lifespan
            assert isinstance(app.state.predictor, FakePredictor)
            for _ in range(3):
                assert client.post("/predict", json=_payload()).status_code == 200
            assert FakePredictor.instances == 1
            assert len(app.state.predictor.calls) == 3
        assert app.state.predictor is None  # released at shutdown
    finally:
        app.dependency_overrides.clear()


def test_predict_is_in_openapi_schema(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/predict"]
