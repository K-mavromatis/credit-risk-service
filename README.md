# Credit Risk Service

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-LightGBM%20%2F%20Scikit--Learn-F7931E)

## Project Overview

This project is an end-to-end Machine Learning microservice designed to assess borrower credit default risk. Built around the [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club), the system evaluates client financial profiles (e.g., income, debt-to-income ratio, employment length) to predict the probability of loan default.

The target architecture is a RESTful API backed by PostgreSQL, fronted by a Streamlit dashboard, with an isolated ML domain for feature preparation and inference. **The API layer does not exist yet** — see [Current status](#current-status) below for what is actually implemented versus planned.

## System Architecture

| Layer | Technology | Responsibility | Status |
| :--- | :--- | :--- | :--- |
| **Interface** | Streamlit | Loan application form with client-side validation. | Form implemented; not wired to a backend yet. |
| **Routing / API** | FastAPI, Pydantic | REST endpoints, request/response validation. | Pydantic schemas and `POST /predict` implemented; `/predict` returns a stub probability (model not wired in yet). |
| **Domain / ML** | Scikit-Learn, LightGBM, Pandas | Feature engineering and the default-probability model. | Notebook-trained pipeline; inference wrapper implemented. |
| **Storage** | PostgreSQL, SQLAlchemy, Alembic | ORM model and schema migration for scored applications. | Model and migration implemented; no code writes to it yet. |
| **Batch ingestion** | Generators, Pydantic | Streams large LendingClub-style files without loading them fully into memory. | Implemented (`src/ml_model/streaming/`), see [PIPELINE.md](PIPELINE.md). |
| **Infrastructure** | Docker, Pytest, Git, pre-commit | Containerization, tests, linting. | Pytest + ruff + pre-commit in place; no Docker/CI yet. |

## Tech Stack & Dependencies

* **Data Processing & ML:** `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`
* **Backend & API (planned, not yet wired up):** `fastapi`, `uvicorn`, `pydantic`
* **Database & ORM:** `sqlalchemy`, `alembic`, `psycopg2-binary`
* **Testing & Quality:** `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`

> ⚠️ `lightgbm` is the library the saved scoring pipeline actually needs in order to unpickle (see [Current status](#current-status)), but it is **not yet listed** in `requirements.txt` / `pyproject.toml`. Install it manually (`pip install lightgbm`) until that gap is closed.

## Project Structure

```text
credit-risk-service/
├── frontend/
│   └── app.py                        # Streamlit loan application form (standalone, no API call yet)
├── notebooks/                        # Research notebooks — the only place the model is currently trained
│   ├── 01_data_preparation.ipynb     # Feature whitelist, target definition, leakage-safe train/test split
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_model_benchmarking.ipynb   # Trains LogisticRegression / RandomForest / LightGBM, saves the best one
│   └── figures/roc_curves_benchmark.png
├── alembic/                          # Schema migrations
│   └── versions/1b4951074b27_init_tables.py
├── scripts/
│   ├── make_dataset.py               # Builds the streaming-pipeline benchmark fixture
│   └── compare.py                    # Lazy generator pipeline vs. intermediate-lists benchmark
├── src/
│   ├── api/
│   │   ├── schemas.py                # Pydantic v2 request/response contract
│   │   └── main.py                   # FastAPI app + POST /predict (stub score for now)
│   ├── db/
│   │   ├── database.py               # Engine, SessionLocal, declarative Base, get_db() dependency
│   │   └── models.py                 # CreditApplication ORM model (mirrors the model's 13 features)
│   │                                  # No crud.py yet — nothing in the codebase writes to the table
│   └── ml_model/
│       ├── data/                      # Local datasets (git-ignored): raw/, interim/, processed/
│       ├── scoring_pipeline.pkl       # Serialized sklearn Pipeline (ColumnTransformer + LightGBM); git-ignored
│       ├── service.py                 # CreditRiskPredictor: loads the pipeline, exposes predict_proba()
│       ├── preprocessing.py           # PercentileClipper transformer (defined, not currently used anywhere)
│       ├── generate_dummy_pipeline.py # Throwaway DummyClassifier stand-in; feature list is stale (15 cols)
│       └── streaming/                 # Generator-based ingestion pipeline for large raw files, see PIPELINE.md
├── tests/
│   ├── conftest.py
│   ├── test_db_models.py             # ORM schema + constraint tests (SQLite, no real DB needed)
│   ├── test_ml_service.py            # Smoke script only — no test_* functions, pytest collects 0 from it
│   └── test_pipeline.py              # Streaming pipeline tests
├── .env.example
├── pyproject.toml
├── requirements.txt
├── PIPELINE.md                       # Deep dive on the streaming/batch ingestion pipeline
├── CONTRIBUTING.md
└── REFACTORING.md                    # Commit-by-commit change log and known discrepancies
```

## Current status

What is real right now, so this README does not promise more than the code does:

- **Model:** three candidates were benchmarked on a stratified, leakage-safe 80/20 split (1,345,350 resolved loans, `notebooks/03_model_benchmarking.ipynb`). **LightGBM won** and is the one serialized to `scoring_pipeline.pkl` — not Random Forest.

  | Model | ROC-AUC | PR-AUC | F1 |
  |---|---:|---:|---:|
  | LightGBM (saved) | 0.699 | 0.367 | 0.417 |
  | Random Forest | 0.694 | 0.360 | 0.412 |
  | Logistic Regression | 0.687 | 0.352 | 0.408 |

  No out-of-time validation and no probability calibration have been done yet — treat the model as a baseline, not a production decision engine.
- **`scoring_pipeline.pkl` is git-ignored and not reproducible from a clean clone**: there is no `train.py`, training only happens by running `03_model_benchmarking.ipynb`, and `lightgbm` is missing from the dependency files. See [REFACTORING.md](REFACTORING.md) for the full list of known gaps.
- **The database and the model are not connected to anything.** `CreditApplication` (the ORM model) and `CreditRiskPredictor` (the inference wrapper) both exist and are tested in isolation, but there is no API endpoint and no `crud.py` that calls one from the other.
- **The Streamlit form stops at validation.** `frontend/app.py` builds and validates a request payload and shows it as JSON; it does not call any backend (there is not one yet).

## Quick Start (Local Development)

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/K-mavromatis/credit-risk-service.git
cd credit-risk-service
pip install -r requirements.txt
pip install lightgbm   # not yet in requirements.txt — needed to load scoring_pipeline.pkl
```

**2. Configure environment variables:**

Copy `.env.example` to `.env` and fill in your local PostgreSQL credentials:
```bash
cp .env.example .env
# DATABASE_URL=postgresql+psycopg2://<user>:<password>@localhost:5432/credit_scoring_db
```

**3. Apply database migrations:**
```bash
alembic upgrade head
```
This creates the `credit_applications` table (see `src/db/models.py`). Nothing in the codebase inserts into it yet — the table exists ahead of the API that will use it.

**4. Train / obtain the scoring pipeline:**

There is no standalone training script. Run `notebooks/03_model_benchmarking.ipynb` top to bottom (it expects `notebooks/01_data_preparation.ipynb` and `02_exploratory_data_analysis.ipynb` to have produced `src/ml_model/data/processed/loans_train.parquet`, `loans_test.parquet` and `preprocessor.joblib` first). The notebook writes `src/ml_model/scoring_pipeline.pkl`.

Alternatively, for plumbing/smoke-testing only (never for real predictions):
```bash
python -m src.ml_model.generate_dummy_pipeline
```

**5. Run the Streamlit form:**
```bash
streamlit run frontend/app.py
```

**6. Run the FastAPI server:**
```bash
uvicorn src.api.main:app --reload
```
Interactive docs at http://127.0.0.1:8000/docs. `POST /predict` validates the request but currently returns a fixed placeholder probability (`0.5`) — model inference is not wired in yet.

**7. Run the tests:**
```bash
python -m pytest -q
python -m pytest -q --cov=src --cov=frontend --cov-report=term-missing
```

For the batch/streaming ingestion pipeline (large raw LendingClub files), see [PIPELINE.md](PIPELINE.md).

## Development Roadmap (Sprints)

- **Sprint 1 — done:** architecture initialization, EDA, baseline model export.
- **Sprint 2 — done:** PostgreSQL integration, SQLAlchemy models (`CreditApplication`), Alembic migrations.
- **Sprint 3 — in progress:** Streamlit loan application form done (validation only, no API call). FastAPI routing, Pydantic schemas, and the DB write path (`crud.py`) not started.
- **Sprint 4 — partial:** ruff + pre-commit hooks and a pytest suite are in place; test coverage is below the 70% target (see [CONTRIBUTING.md](CONTRIBUTING.md#testing)), and `src/api/` and `frontend/app.py` have little to no test coverage yet.
- **Sprint 5 — not started:** Docker containerization and CI/CD.

## Core Team

**Konstantinos Mavromatis** — *Machine Learning & Backend API*
* **Focus:** Data Analytics, ML Engineering, PostgreSQL
* **GitHub:** [@K-mavromatis](https://github.com/K-mavromatis)
* **LinkedIn:** [Kostas Mavromatis](https://www.linkedin.com/in/kostas-mavromatis-193171310/)

**[Liza Zinkina]** — *Frontend & UI Integration*
* **Focus:** Interface Design, Streamlit, API Integration
* **GitHub:** [@Liza0316](https://github.com/Liza0316)
* **LinkedIn:** [Liza Zinkina](https://www.linkedin.com/in/%D0%BB%D0%B8%D0%B7%D0%B0-%D0%B7%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD%D0%B0-30796a336/)
