"""FastAPI application entry point (Sprint 3, Card 2).

Creates the ``app`` instance and the ``POST /predict`` endpoint.

Current state: **stub scoring**. The endpoint validates the request with the
Card 1 schemas and returns a fixed placeholder probability. It does not call
``CreditRiskPredictor`` and does not write to the database yet. Both are wired
in by later cards; only the body of ``predict`` has to change, because the
request and response contracts are already final.

Run locally (from the project root):
    uvicorn src.api.main:app --reload
    # or, as written on the card, after `pip install -e .`:
    uvicorn api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

from fastapi import FastAPI, status

# Relative import: works whether this module is loaded as ``src.api.main``
# (tests, `uvicorn src.api.main:app`) or ``api.main`` (editable install).
from .schemas import ApplicationCreate, ApplicationResponse

API_VERSION: Final[str] = "0.1.0"

# --- Stub values, removed once the model and the DB are wired in ------------------
# 0.5 = "no information": clearly a placeholder, never mistaken for a real score.
MOCK_DEFAULT_PROBABILITY: Final[float] = 0.5
# 0 is never issued by the PostgreSQL sequence (it starts at 1), so it marks
# "not persisted" unambiguously.
MOCK_APPLICATION_ID: Final[int] = 0

app = FastAPI(
    title="Credit Scoring API",
    version=API_VERSION,
    description=(
        "Scores a loan application and returns its probability of default. "
        "**Stub mode:** `/predict` currently returns a fixed placeholder "
        "probability; model inference is not wired in yet."
    ),
)


@app.post(
    "/predict",
    response_model=ApplicationResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
    summary="Score one loan application",
    response_description="The application echoed back with its default probability",
)
def predict(application: ApplicationCreate) -> ApplicationResponse:
    """Validate one application and return its (stub) default probability.

    FastAPI validates the JSON body against ``ApplicationCreate`` before this
    function runs. Any invalid input (wrong type, unknown category, value out
    of range, unexpected field) is answered with ``422`` automatically.

    Declared as a plain ``def`` (not ``async def``) on purpose: real inference
    is CPU-bound sklearn code, and FastAPI runs sync endpoints in a thread
    pool, so a slow prediction will not block the event loop.
    """
    # TODO(Sprint 3): replace the stub with
    #   probability = predictor.predict_proba(application.model_dump())
    #   row = crud.create_application(db, ..., default_probability=probability)
    #   return ApplicationResponse.model_validate(row)
    return ApplicationResponse(
        **application.model_dump(),
        id=MOCK_APPLICATION_ID,
        default_probability=MOCK_DEFAULT_PROBABILITY,
        created_at=datetime.now(timezone.utc),
    )
