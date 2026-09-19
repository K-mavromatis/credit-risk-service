# Contributing to Credit Risk Service

Thanks for helping improve Credit Risk Service, an ML microservice that scores loan applications for default risk. The stack is Python 3.10+, FastAPI, Pydantic v2, PostgreSQL with SQLAlchemy 2.0, scikit-learn, and a Streamlit front end.

This guide explains how to set up your environment, the standards we hold code to, and how changes are reviewed and merged. For a project overview and instructions for running the service, see the [README](README.md).

## Ways to contribute

- Report a bug or propose a feature by opening an issue (see [Reporting issues](#reporting-issues)).
- Improve the data pipeline, model, API, database layer, or UI.
- Add or improve tests and documentation.

Before starting anything sizeable, such as a new module, database table, or dependency, open an issue and agree on the approach with a maintainer. It keeps the architecture consistent and saves everyone wasted effort.

## Project layout and layer boundaries

```text
credit-risk-service/
├── frontend/          # Streamlit UI
├── notebooks/         # Research, EDA, and prototyping
├── src/
│   ├── api/           # FastAPI routes and Pydantic schemas
│   ├── db/            # SQLAlchemy models, sessions, and CRUD
│   └── ml_model/      # ML pipelines, training, and inference
│       └── data/      # Local datasets (git-ignored): raw/, interim/, processed/
├── tests/             # Automated test suite
├── pyproject.toml     # Project metadata and tool configuration
└── .pre-commit-config.yaml
```

A scoring request flows from the Streamlit UI to the FastAPI service, which calls the ML service for a prediction, saves the application and result through the DB layer, and returns a structured response. These rules keep that flow maintainable:

- **Respect the layers.** The API calls into the DB and ML services. ORM and SQL logic lives in `src/db/`; endpoints never build queries themselves.
- **The front end talks to the API only over HTTP.** It never imports from `src/db/` or `src/ml_model/`.
- **Validate every request and response with Pydantic v2 schemas** defined in `src/api/`. Changing a schema the front end consumes requires review from the front-end owner (see [Review process and ownership](#review-process-and-ownership)).
- **No new tables, modules, or dependencies without prior discussion** in an issue.

## Getting started

You will need Python 3.10+, Git, and a local PostgreSQL instance (for API and database work).

```bash
# 1. Clone (external contributors: fork first, then clone your fork)
git clone https://github.com/K-mavromatis/credit-risk-service.git
cd credit-risk-service

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install the project with the dev tools (pytest, pytest-cov, ruff, pre-commit)
pip install -e ".[dev]"

# 4. Enable the pre-commit hooks
pre-commit install
```

Create a `.env` file in the repository root. It is git-ignored, so never commit it:

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/credit_risk_db
```

Use local placeholder credentials only, and create the `credit_risk_db` database (or your own name) in your local PostgreSQL first.

**Data.** The raw dataset is not stored in git. Ask a maintainer for it and place it in `src/ml_model/data/raw/`; the filename the preparation notebook expects is set at the top of `notebooks/01_data_preparation.ipynb`. Everything under `src/ml_model/data/` is git-ignored, as are `*.csv`, `*.parquet`, `*.pkl`, and `*.joblib` files.

For running the API and the UI, follow the README.

## Development workflow

1. Update your local `main`: `git switch main && git pull`.
2. Create a branch named `<type>/<short-description>`, where the type is one of `feature`, `fix`, `docs`, `refactor`, `test`, or `chore`. For example: `git switch -c feature/scoring-endpoint`.
3. Make small, focused commits. Write the subject line in the imperative mood and keep it under 72 characters (for example, `Add loan request schema`). Use the body to explain why when it isn't obvious.
4. Push your branch and open a pull request against `main`. Keep each PR to one logical change.
5. Respond to review comments, then merge once the PR is approved and the checklist below is complete.

Please don't push directly to `main`.

## Code standards

**Python and typing**

- Target Python 3.10+ and use modern syntax (`list[str]`, `X | None`).
- Use explicit type annotations on all function parameters and return values, and on class attributes and module-level constants. Reviewers check this.
- Use absolute imports only, rooted at `src`, for example `from src.db.models import LoanRecord`. Keep imports sorted (ruff enforces this).
- Read configuration from environment variables (via `python-dotenv`). Never hard-code credentials, hosts, or absolute file paths; use `pathlib.Path` relative to the project root.

**Pydantic v2**

- Configure models with `model_config = ConfigDict(...)` and validate with `field_validator` / `model_validator`.
- Use `model_validate()` and `model_dump()`. Do not use Pydantic v1 APIs (`class Config`, `@validator`, `.dict()`, `.parse_obj()`).

**SQLAlchemy 2.0 and Alembic**

- Use 2.0-style `select()` statements and typed models (`Mapped[...]`, `mapped_column()`). Do not use the legacy `Query` API.
- Every change to a database model ships with an Alembic migration. Generate it, then review the result before committing:

  ```bash
  alembic revision --autogenerate -m "describe the change"
  alembic upgrade head
  ```

**Linting and formatting**

Ruff is configured in `pyproject.toml` (line length 88, target `py310`, rules `E`, `F`, and `I`; notebooks are exempt from `E501`). The pre-commit hooks run `ruff --fix` and `ruff-format` on every commit. You can also run them manually:

```bash
ruff check .                  # lint (add --fix to apply safe fixes)
ruff format .                 # format
pre-commit run --all-files    # everything the hooks run
```

Fix warnings rather than silencing them. If a `# noqa` is unavoidable, use a specific rule code and add a short reason.

## Machine learning and data guidelines

- **Notebooks are for exploration.** Once logic is stable, move it into importable, tested code under `src/ml_model/`. The service must never depend on a notebook.
- **Clear notebook outputs before committing** (they can leak local paths and data samples), and make sure the notebook runs top to bottom after "Restart & Run All".
- **Treat raw data as immutable.** Files in `src/ml_model/data/raw/` are never edited; derived data goes to `interim/` or `processed/` and is produced by code.
- **Keep results reproducible.** Set `random_state` wherever randomness is involved, and build preprocessing and model together as a scikit-learn `Pipeline` / `ColumnTransformer` so transformations are fit on training data only (no leakage). Record the scikit-learn version when you export a model, since pickles are not guaranteed to load across versions.
- **Model artifacts are not committed.** `*.pkl` and `*.joblib` files are git-ignored; regenerate them with the training script in `src/ml_model/` (`train.py`). Only load artifacts you produced yourself or trust, because unpickling can execute arbitrary code.
- **Model changes need evidence.** A PR that changes features, training, the decision threshold, or scoring logic must explain what changed and why, and report evaluation metrics (for example ROC AUC, and precision and recall at the decision threshold) for the old and new model on the same held-out set. These changes affect credit decisions, so they get closer scrutiny.

## Testing

- Tests live in `tests/` and are named `test_*.py`, for example `test_api.py` for endpoints and `test_ml.py` for model sanity checks such as probabilities staying within [0, 1].
- Every bug fix gets a regression test, and every new endpoint, schema, or pipeline step gets tests.
- Tests must be deterministic and self-contained: no production database, and no dependence on the full raw dataset. Use small synthetic fixtures, FastAPI dependency overrides, or a dedicated test database.
- The project targets at least 70% test coverage. Don't let it drop.

```bash
python -m pytest                                        # run the suite
python -m pytest --cov=src --cov-report=term-missing    # with coverage
```

## Pull request checklist

Before requesting a review, confirm that:

- [ ] `pre-commit run --all-files` passes.
- [ ] `python -m pytest` passes, and new or changed behavior is covered by tests.
- [ ] Type annotations, Pydantic v2, and SQLAlchemy 2.0 conventions are followed.
- [ ] An Alembic migration is included for any database model change.
- [ ] `CHANGELOG.md` has an entry under `## [Unreleased]`, grouped by area (for example CI / Tooling, Data / ML Pipeline, API, Database), with one line per change.
- [ ] The README and other docs are updated if setup, behavior, or the API contract changed.
- [ ] No secrets, `.env` files, datasets, model artifacts, or real customer data are included.
- [ ] The PR description explains what changed and why, links the related issue, and (for ML changes) includes the evaluation metrics.

## Review process and ownership

- Every PR needs at least one approving review from a maintainer other than the author before it is merged into `main`.
- Ownership:
  - **Konstantinos Mavromatis** ([@K-mavromatis](https://github.com/K-mavromatis)): ML, backend API, and PostgreSQL (`src/api/`, `src/db/`, `src/ml_model/`, `notebooks/`).
  - **Liza Zinkina** ([@Liza0316](https://github.com/Liza0316)): front end and UI integration (`frontend/`).
- Changes that cross layers, or that alter the API contract the front end relies on, need review from both owners.
- Keep review feedback specific and constructive, and resolve or reply to every comment.

## Reporting issues

When you open a bug report, please include:

- What you expected to happen and what actually happened.
- Steps to reproduce, with a minimal request payload built from **synthetic data only**.
- Your environment: OS, Python version, and key package versions.
- Relevant logs or tracebacks, with secrets and personal data removed.

For feature requests, describe the problem you want to solve, your proposed approach, and which layers (API, DB, ML, UI) it affects.

## Security and sensitive data

This service works with financial risk data. Never commit or post secrets, credentials, `.env` contents, or real applicant or customer information anywhere: not in code, tests, notebook outputs, issues, PRs, logs, or screenshots. Use synthetic or anonymized data instead.

If you find a security vulnerability, don't open a public issue. Contact a maintainer directly through their GitHub profile.
