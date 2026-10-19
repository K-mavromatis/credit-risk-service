"""FastAPI application entry point (Sprint 3, Card 3: model + database integration).

``POST /predict`` scores one loan application end to end:

1. FastAPI validates the JSON body against ``ApplicationCreate`` (Card 1).
2. The ``CreditRiskPredictor`` loaded once at startup returns the probability
   of default for the 13 raw features.
3. ``crud.create_application`` (Sprint 2) stores the features and the score in
   PostgreSQL; the database assigns ``id`` and ``created_at``.
4. The stored row is returned as ``ApplicationResponse``.

The model is loaded in the ``lifespan`` handler, so ``scoring_pipeline.pkl`` is
read from disk exactly once per process, not once per request. The endpoint
gets it, and the database session, through dependencies (``get_predictor`` and
``get_db``), which tests replace with ``app.dependency_overrides``.

Run locally (from the project root, with ``DATABASE_URL`` set in ``.env``):
    uvicorn src.api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Final

from fastapi import Depends, FastAPI, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.schemas import ApplicationCreate, ApplicationResponse
from src.db import crud
from src.db.database import engine, get_db
from src.ml_model.service import CreditRiskPredictor

API_VERSION: Final[str] = "0.2.0"

logger: logging.Logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the scoring model at startup and release resources at shutdown.

    Everything before ``yield`` runs once, before the first request is served;
    everything after it runs once, when the server stops.

    If the pipeline file is missing or corrupt, ``CreditRiskPredictor`` raises
    and the application refuses to start. That is deliberate: a scoring
    service without a model should fail loudly at deploy time, not answer
    every request with an error.
    """
    logger.info("Loading the credit scoring pipeline...")
    app.state.predictor = CreditRiskPredictor()
    logger.info("Credit scoring pipeline loaded.")
    try:
        yield
    finally:
        app.state.predictor = None
        engine.dispose()  # close the pooled PostgreSQL connections cleanly


app = FastAPI(
    title="Credit Scoring API",
    version=API_VERSION,
    description=(
        "Scores a loan application with the Random Forest scoring pipeline, "
        "stores the application and its probability of default in PostgreSQL, "
        "and returns the stored record."
    ),
    lifespan=lifespan,
)


def get_predictor(request: Request) -> CreditRiskPredictor:
    """FastAPI dependency: the model instance loaded once at startup.

    Raises ``503`` if the model is not loaded, which only happens when the app
    runs without its lifespan (for example ``TestClient(app)`` used outside a
    ``with`` block and without a dependency override).
    """
    predictor: CreditRiskPredictor | None = getattr(
        request.app.state, "predictor", None
    )
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The scoring model is not loaded.",
        )
    return predictor


DbSession = Annotated[Session, Depends(get_db)]
Predictor = Annotated[CreditRiskPredictor, Depends(get_predictor)]


@app.post(
    "/predict",
    response_model=ApplicationResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
    summary="Score one loan application and store the result",
    response_description="The stored application with its default probability",
    responses={
        500: {"description": "The model failed to score the application"},
        503: {"description": "The model is not loaded or the database is down"},
    },
)
def predict(
    application: ApplicationCreate,
    db: DbSession,
    predictor: Predictor,
) -> ApplicationResponse:
    """Score one application, save it to the database and return the stored row.

    FastAPI validates the JSON body against ``ApplicationCreate`` before this
    function runs, so any invalid input is answered with ``422`` and never
    reaches the model or the database.

    Declared as a plain ``def`` (not ``async def``) on purpose: sklearn
    inference and the psycopg2 driver are both blocking, and FastAPI runs sync
    endpoints in a thread pool, so they do not block the event loop.
    """
    features = application.model_dump()

    # 1. Model inference -----------------------------------------------------------
    try:
        default_probability: float = predictor.predict_proba(features)
    except (RuntimeError, ValueError) as exc:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The model failed to score this application.",
        ) from exc

    # 2. Persistence ---------------------------------------------------------------
    # crud.ApplicationCreate is the DB-layer input schema (features + score);
    # it is a different class from the API request schema of the same name.
    record = crud.ApplicationCreate(**features, default_probability=default_probability)
    try:
        stored = crud.create_application(db, record)
    except SQLAlchemyError as exc:
        # create_application has already rolled the session back.
        logger.exception("Could not save the scored application")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The application was scored but could not be saved.",
        ) from exc

    # 3. Response ------------------------------------------------------------------
    # from_attributes=True lets Pydantic read the ORM row directly, including
    # the id and created_at values generated by PostgreSQL.
    return ApplicationResponse.model_validate(stored)
