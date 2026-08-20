# ClaimSight — Intelligent Insurance Claims Triage & Routing System

<div align="center">

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

> **Production-grade multi-task claims triage pipeline delivering automated severity reserve estimation, explainable fraud anomaly detection, and capacity-aware adjuster routing.**

---

## 📖 Executive Summary & Value Proposition

**`claimsight`** is a production-grade, end-to-end machine learning system built with strict engineering discipline, reproducible pipelines, and enterprise MLOps best practices. It bridges the gap between theoretical statistical rigor and high-availability operational microservices.

## 🏥 Core Methodologies & System Modules

### 1. Loss Severity & Reserve Estimation
- Tweedie compound Poisson-gamma generalized linear models and LightGBM quantile regressors designed specifically for heavy-tailed loss distributions.
- Produces expected loss and P90 reserve recommendations to prevent reserve deficit drift.

### 2. Multi-Signal Fraud Detection & Audit Trail
- Rule-based triggers (rapid filing after policy inception, prior claim clusters, unverified provider IDs).
- Unsupervised anomaly scoring and gradient-boosted fraud classification generating transparent evidence tags for Special Investigation Unit (SIU) review.

### 3. Complexity Scoring & Workload-Balanced Routing
- Computes claim complexity index based on multi-party injury risk, litigation propensity, and coverage ambiguity.
- Solves an integer linear programming assignment matching claims to adjuster skill certifications while balancing open caseload constraints.

## 📊 Architecture & Pipeline

```mermaid
flowchart LR
    Claim[Incoming FNOL Claim] --> Prep[Pandera Feature Pipeline]
    Prep --> Sev[Severity & Reserve Quantiles]
    Prep --> Frd[Fraud Anomaly & Evidence Flags]
    Prep --> Cmp[Complexity Index Scorer]
    Sev & Frd & Cmp --> Route[Capacity-Aware Adjuster Routing]
    Route --> API[FastAPI :8250] --> UI[Streamlit Claims Desk :8751]
```

## 🛠️ Tech Stack & Engineering Standards
- **Core Engine:** Python 3.12, NumPy, SciPy, Pandas, LightGBM, Scikit-Learn
- **Serving & UI:** FastAPI, Streamlit, MLflow
- **Testing & Quality:** Pytest test suite, Ruff linting, Docker Compose


---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites & Environment Setup
Using **[uv](https://docs.astral.sh/uv/)** for lightning-fast, reproducible dependency resolution:

```bash
# Clone the repository
git clone https://github.com/jackson-marcus/claimsight.git
cd claimsight

# Install dependencies and pre-commit hooks
uv sync --group dev
```

### 2. Run Test Suite & Code Quality Checks
```bash
# Run unit & integration tests with coverage
uv run pytest --cov

# Run ruff linter and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### 3. Launch Services Locally
```bash
# Start FastAPI REST API (listening on port :8250)
make api
# Or: uv run uvicorn claimsight.api.main:app --reload --port 8250

# Start interactive Streamlit dashboard (listening on port :8751)
make ui

# Launch local MLflow Experiment Tracking UI (listening on port :5026)
make mlflow
```

### 4. Run with Docker Compose
```bash
# Spin up the complete microservice stack
docker compose up --build
```

---

## 📂 Repository Layout

```
claimsight/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline (lint, test, build)
├── configs/                      # Configuration files and hyperparameters
├── data/                         # Data directory (raw, interim, processed)
├── scripts/                      # Data generators and operational scripts
├── src/claimsight/               # Core Python package
│   ├── api/                      # FastAPI routes, schemas, and endpoints
│   ├── models/                   # Statistical models, ML algorithms, and estimators
│   ├── ui/                       # Streamlit interactive application
│   └── settings.py               # Centralized configuration & environment loader
├── tests/                        # Comprehensive Pytest suite
├── docker-compose.yml            # Multi-service container orchestration
├── Dockerfile                    # Container definition for API service
├── Makefile                      # Standardized project tasks
└── pyproject.toml                # Pinned dependencies and tool configs
```

---

## 👤 Author & Contact

**Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus)

*Available for machine learning engineering, MLOps, data science, and AI system architecture consulting and contract engagements.*

