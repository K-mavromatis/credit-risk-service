# Credit Risk Service

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E)

## Project Overview

This project is an end-to-end Machine Learning microservice designed to assess borrower credit default risk. Built around the [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club), the system evaluates client financial profiles (e.g., income, debt-to-income ratio, employment length) to predict the probability of loan default.

Unlike a standard analytical script, this repository implements a production-ready pipeline: an isolated ML domain, a RESTful API, relational database storage, and an interactive inference dashboard.
## System Architecture

| Layer | Technology | Responsibility |
| :--- | :--- | :--- |
| **Interface** | Streamlit | Interactive web UI for submitting loan applications. |
| **Routing / API** | FastAPI, Pydantic | REST API endpoints, strict data validation, and JSON responses. |
| **Domain / ML** | Scikit-Learn, Pandas | Feature engineering pipelines and the Random Forest predictive model. |
| **Storage** | PostgreSQL, SQLAlchemy, Alembic | Persistent storage of application history, client features, and scoring results; schema changes are versioned as Alembic migrations. |
| **Infrastructure** | Docker, Pytest, Git | Containerization, unit testing (>=70% coverage), and CI/CD pipelines. |

## Tech Stack & Dependencies

* **Data Processing & ML:** `pandas`, `numpy`, `scikit-learn`, `joblib`
* **Backend & API:** `fastapi`, `uvicorn`, `pydantic`
* **Database & ORM:** `sqlalchemy`, `alembic`, `psycopg2-binary`
* **Testing & Quality:** `pytest`, `ruff`

## Project Structure

```text
credit-risk-service/
├── alembic/                      # Database migrations (Alembic)
│   ├── versions/                 # One migration script per schema change
│   └── env.py                    # Reads DATABASE_URL from .env, targets Base.metadata
├── frontend/                     # User interface layer
│   └── app.py                    # Interactive Streamlit dashboard
├── notebooks/                    # Research, EDA, and prototyping
│   └── 01_data_preparation.ipynb # Data cleaning, filtering, and initial pipeline
├── src/                          # Application source code (Python package)
│   ├── api/                      # FastAPI web service layer
│   │   ├── main.py               # Application entrypoint & route handlers
│   │   └── schemas.py            # Pydantic data contracts (request/response validation)
│   ├── db/                       # Relational database layer
│   │   ├── crud.py               # Persistence queries (create, read application records)
│   │   ├── database.py           # SQLAlchemy engine setup & session management
│   │   └── models.py             # ORM schemas (PostgreSQL tables)
│   └── ml_model/                 # Machine Learning pipeline & artifacts
│       ├── data/                 # Local data storage (excluded from git)
│       │   ├── interim/          # Intermediate transformed subsets
│       │   ├── processed/        # Final training datasets (parquet/csv)
│       │   └── raw/              # Immutable raw source data
│       ├── rf_model.pkl          # Serialized Random Forest model artifact
│       ├── service.py            # Inference class for loading model & scoring
│       └── train.py              # Model training & feature engineering script
├── tests/                        # Automated test suite
│   ├── test_api.py               # REST API endpoint tests
│   └── test_ml.py                # Predictive sanity & probability bounds tests
├── .env                          # Local environment variables (git-ignored, never committed)
├── .env.example                  # Committed template for .env
├── .gitignore                    # Git ignore rules (caches, data, venv, secrets)
├── alembic.ini                   # Alembic configuration (no credentials; URL comes from .env)
├── CHANGELOG.md                  # Notable changes, grouped by area
├── CONTRIBUTING.md               # Setup, code standards, and PR workflow
├── pyproject.toml                # Project metadata, dependencies, Ruff & Pytest configs
├── README.md                     # Project overview and run documentation
└── requirements.txt              # Pinned production & development dependencies
```

## Quick Start (Local Development)

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/K-mavromatis/credit-risk-service.git
cd credit-risk-service
pip install -r requirements.txt
```

**2. Configure Environment Variables:**
Copy the template and fill in your local PostgreSQL credentials. `.env` is git-ignored, so never commit it:
```bash
cp .env.example .env              # Windows: copy .env.example .env
```
```bash
DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/credit_scoring_db
```

**3. Create the Database Schema:**
Create an empty `credit_scoring_db` database in PostgreSQL (for example in pgAdmin), check the connection, then apply all migrations:
```bash
python src/db/database.py         # expect: Connection OK
alembic upgrade head              # creates/updates every table
alembic current                   # shows the applied revision, e.g. "1b4951074b27 (head)"
```
Run `alembic upgrade head` again whenever you pull new migrations. The database URL is read from `.env` by `alembic/env.py`; it is never stored in `alembic.ini`.

**4. Run the API Server:**

```bash
uvicorn api.main:app --reload
```

*The interactive API documentation (Swagger) will be available at* `http://localhost:8000/docs.`


**Development Roadmap (Sprints):**
- Sprint 1: Architecture initialization, basic EDA, and baseline ML model export.
- Sprint 2: PostgreSQL integration, SQLAlchemy models, and Alembic migrations.
- Sprint 3: FastAPI routing, Pydantic schemas, and Streamlit UI integration.
- Sprint 4: Code quality, linting, and Pytest implementation.
- Sprint 5: Docker containerization and final deployment.


## Core Team

**Konstantinos Mavromatis** — *Machine Learning & Backend API*
* **Focus:** Data Analytics, ML Engineering, PostgreSQL
* **GitHub:** [@K-mavromatis](https://github.com/K-mavromatis)
* **LinkedIn:** [Kostas Mavromatis](https://www.linkedin.com/in/kostas-mavromatis-193171310/)

**[Liza Zinkina]** — *Frontend & UI Integration*
* **Focus:** Interface Design, Streamlit, API Integration
* **GitHub:** [@Liza0316](https://github.com/Liza0316)
* **LinkedIn:** [Liza Zinkina](https://www.linkedin.com/in/%D0%BB%D0%B8%D0%B7%D0%B0-%D0%B7%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD%D0%B0-30796a336/)