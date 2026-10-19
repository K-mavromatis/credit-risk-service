"""FastAPI application entry point (Sprint 3, Cards 3-4: model, database, CORS).

``POST /predict`` scores one loan application end to end:

1. FastAPI validates the JSON body against ``ApplicationCreate`` (Card 1).
2. The ``CreditRiskPredictor`` loaded once at startup returns the probability
   of default for the 13 raw features.
3. ``crud.create_application`` (Sprint 2) stores the features and the score in
   PostgreSQL; the database assigns ``id`` and ``created_at``.
4. The stored row is returned as ``ApplicationResponse``, together with the
   computed ``decision`` (``approved`` / ``rejected``, Sprint 4 Card 3).

``GET /health`` is a liveness check for monitoring and container probes: it
answers ``200`` without touching the model or the database.

The model is loaded in the ``lifespan`` handler, so ``scoring_pipeline.pkl`` is
read from disk exactly once per process, not once per request. The endpoint
gets it, and the database session, through dependencies (``get_predictor`` and
``get_db``), which tests replace with ``app.dependency_overrides``.

CORS (Card 4): ``CORSMiddleware`` lets browser code served from another origin
(the Streamlit UI on ``http://localhost:8501``) call this API. Allowed origins
are an explicit list (``CORS_ORIGINS``) plus a regex that accepts any port on
``localhost`` / ``127.0.0.1`` for local development (``CORS_ORIGIN_REGEX``).
A bare ``"*"`` is deliberately not used: browsers reject ``*`` together with
credentials, and it would let any website call the API from a user's browser.

Run locally (from the project root, with ``DATABASE_URL`` set in ``.env``):
    uvicorn src.api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Final

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.schemas import (
    ApplicationCreate,
    ApplicationResponse,
    HealthResponse,
    get_risk_threshold,
)
from src.db import crud
from src.db.database import engine, get_db
from src.ml_model.service import CreditRiskPredictor

API_VERSION: Final[str] = "0.4.0"

logger: logging.Logger = logging.getLogger(__name__)

# --- CORS configuration ------------------------------------------------------------
# Streamlit's default port is 8501; both host spellings are listed because the
# browser treats ``localhost`` and ``127.0.0.1`` as different origins.
DEFAULT_CORS_ORIGINS: Final[tuple[str, ...]] = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
)
# Any port on localhost / 127.0.0.1 (http or https): the safe "wildcard" for
# local development. Set CORS_ORIGIN_REGEX to an empty value to disable it.
DEFAULT_CORS_ORIGIN_REGEX: Final[str] = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
CORS_ALLOW_METHODS: Final[list[str]] = ["GET", "POST", "OPTIONS"]
CORS_ALLOW_HEADERS: Final[list[str]] = ["Accept", "Authorization", "Content-Type"]
CORS_MAX_AGE_SECONDS: Final[int] = 600  # browsers may cache a preflight for 10 min


def parse_cors_origins(raw: str | None) -> list[str]:
    """Turn a comma-separated ``CORS_ORIGINS`` value into a clean origin list.

    ``None`` or a blank string falls back to ``DEFAULT_CORS_ORIGINS``. Trailing
    slashes are stripped, because an origin never has one and
    ``"http://localhost:8501/"`` would silently never match.

    Args:
        raw: Raw ``CORS_ORIGINS`` environment value, or None if unset.

    Returns:
        The list of allowed origins without trailing slashes.
    """
    if raw is None or not raw.strip():
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


# ``src.db.database`` has already run ``load_dotenv()``, so values from ``.env``
# are visible here.
CORS_ORIGINS: Final[list[str]] = parse_cors_origins(os.getenv("CORS_ORIGINS"))
CORS_ORIGIN_REGEX: Final[str | None] = (
    os.getenv("CORS_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX) or None
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the scoring model at startup and release resources at shutdown.

    Everything before ``yield`` runs once, before the first request is served;
    everything after it runs once, when the server stops.

    If the pipeline file is missing or corrupt, ``CreditRiskPredictor`` raises
    and the application refuses to start. That is deliberate: a scoring
    service without a model should fail loudly at deploy time, not answer
    every request with an error.

    An invalid ``RISK_THRESHOLD`` also stops startup, for the same reason.

    Args:
        app: The FastAPI application whose ``state`` holds the predictor.

    Yields:
        Control to FastAPI while the application is serving requests.

    Raises:
        ValueError: If ``RISK_THRESHOLD`` is set but not a number in (0, 1).
    """
    logger.info("Decision threshold: %.3f", get_risk_threshold())
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
        "Scores a loan application with the LightGBM scoring pipeline, "
        "stores the application and its probability of default in PostgreSQL, "
        "and returns the stored record."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "scoring",
            "description": "Score a loan application and store the result.",
        },
        {
            "name": "monitoring",
            "description": "Service health checks.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,  # cookies / Authorization header may be sent
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
    max_age=CORS_MAX_AGE_SECONDS,
)


def get_predictor(request: Request) -> CreditRiskPredictor:
    """FastAPI dependency: the model instance loaded once at startup.

    Raises ``503`` if the model is not loaded, which only happens when the app
    runs without its lifespan (for example ``TestClient(app)`` used outside a
    ``with`` block and without a dependency override).

    Args:
        request: The incoming request, used to reach ``app.state``.

    Returns:
        The ``CreditRiskPredictor`` loaded at startup.

    Raises:
        HTTPException: 503 if the model is not loaded.
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


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["monitoring"],
    summary="Liveness check",
)
def health(request: Request) -> HealthResponse:
    """Report that the service is up, and whether the model is loaded.

    Liveness only: no database query and no inference, so the check is cheap
    enough for a load balancer or Kubernetes probe to call every few seconds.

    Args:
        request: The incoming request, used to reach ``app.state``.

    Returns:
        ``status="ok"``, the API version and the model-loaded flag.
    """
    model_loaded: bool = getattr(request.app.state, "predictor", None) is not None
    return HealthResponse(version=API_VERSION, model_loaded=model_loaded)


@app.post(
    "/predict",
    response_model=ApplicationResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
    summary="Score one loan application and store the result",
    response_description="The stored application, its default probability and decision",
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
    \f
    Args:
        application: Validated request body with the 13 model features.
        db: Database session for this request.
        predictor: The scoring model loaded at startup.

    Returns:
        The stored application with its ``id``, score, ``decision`` and ``created_at``.

    Raises:
        HTTPException: 500 if the model fails, 503 if the row cannot be saved.
    """  # noqa: D301 - "\f" hides the sections below it from Swagger UI
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
