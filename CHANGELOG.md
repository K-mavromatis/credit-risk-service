# Changelog

All notable changes to the Credit Risk ML microservice are documented in this file. Entries are grouped by area, newest first within each area, with one line per change. Pull request numbers refer to [K-mavromatis/credit-risk-service](https://github.com/K-mavromatis/credit-risk-service/pulls).

## [Unreleased]

### Database
- Added Alembic schema migrations (Sprint 2, Card 3): `alembic.ini` with no stored credentials, an `alembic/env.py` that reads `DATABASE_URL` through `DatabaseSettings` and targets `Base.metadata` from `src.db.models`, and the initial revision `1b4951074b27` ("Init tables"), which creates `credit_applications` with its primary key, three named CHECK constraints, and an index on `created_at`.
- Added the `CreditApplication` ORM model (`src/db/models.py`) holding the 13 raw scoring features, the predicted default probability, and a server-side `created_at` timestamp, with schema tests in `tests/test_db_models.py` (Sprint 2, Card 2, #13).
- Added the PostgreSQL connection layer (`src/db/database.py`): Pydantic v2 `DatabaseSettings` loaded from `.env`, a pre-pinging pooled engine, `SessionLocal`, a declarative `Base` with a constraint naming convention, and the `get_db` FastAPI dependency; added the `.env.example` template (Sprint 2, Card 1, #12).

### Frontend
- Added the Streamlit loan application form with input validation (`frontend/app.py`) and pinned `streamlit==1.63.0` (#11).

### Data / ML Pipeline
- Added a streaming pipeline with running statistics and batching (`src/ml_model/streaming/`), the helper scripts `scripts/make_dataset.py` and `scripts/compare.py`, and `PIPELINE.md` (#10).
- Benchmarked candidate models and saved the LightGBM pipeline and ROC curves (`notebooks/03_model_benchmarking.ipynb`) (#9).
- Added the exploratory data analysis notebook (`notebooks/02_exploratory_data_analysis.ipynb`) and reusable preprocessing in `src/ml_model/preprocessing.py` (#8).
- Fixed data leakage in the data preparation notebook (#6, #8).
- Added the `CreditRiskPredictor` inference wrapper (`src/ml_model/service.py`), a dummy pipeline generator, and smoke tests (#4).
- Completed the initial data preparation and preprocessing pipeline (`notebooks/01_data_preparation.ipynb`) (#1).

### CI / Tooling
- Added `mypy` to the dev dependencies with a `[tool.mypy]` configuration, and put the repository root on the pytest path for `src`-rooted imports (#10).
- Added a `pre-commit` configuration that runs `ruff --fix` and `ruff-format` on every commit (#3).

### Documentation
- Documented the Alembic workflow: database setup and `alembic upgrade head` in the README quick start, and migration review rules in `CONTRIBUTING.md`; added this changelog.
- Added `REFACTORING.md` (repository audit) and removed committed IDE files and `egg-info` build artifacts (#7).
- Added `CONTRIBUTING.md` with setup, code standards, PR checklist, and ownership (#5).

### Project Structure & Configuration
- Reorganized the codebase into a `src`-based layout and introduced `pyproject.toml` with `ruff` configured as the linter and formatter (#2).
- Initialized the project skeleton, base dependencies (`requirements.txt`), `README.md`, and `.gitignore`.

---

When versioned releases begin, move the `[Unreleased]` entries under a dated heading such as `## [0.1.0] - YYYY-MM-DD`.
