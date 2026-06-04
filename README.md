<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-Phase%201%20In%20Progress-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/models-CPU%20Only-purple?style=for-the-badge" />
</p>

# XAURA — eXtendable Automated Unified Research & Analytics

> An intelligent, dataset-aware Python ML library with built-in profiling, experiment tracking, model-specific visualisations, and a local web UI — all in one `pip install`.

XAURA is designed to make machine learning workflows **faster, smarter, and fully reproducible**. Instead of writing boilerplate for every project, you call `profile()` to understand your data and `run_model()` to train with dataset-aware defaults. Every run is automatically logged, visualised, and exportable.

---

## Table of Contents

- [Why XAURA?](#why-xaura)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [Phase 1 — MVP Scope (CPU-Only)](#phase-1--mvp-scope-cpu-only)
  - [1. Dataset Profiling](#1-dataset-profiling)
  - [2. Supported Models](#2-supported-models)
  - [3. Dataset-Aware Defaults](#3-dataset-aware-defaults)
  - [4. Model-Aware Visualisations](#4-model-aware-visualisations)
  - [5. Experiment Tracking](#5-experiment-tracking)
  - [6. Export](#6-export)
  - [7. Local Web UI](#7-local-web-ui)
  - [8. CLI Interface](#8-cli-interface)
- [Phase 2 — Agentic Layer (Future)](#phase-2--agentic-layer-future)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Development Setup](#development-setup)
- [Development Roadmap & Work Division](#development-roadmap--work-division)
- [Contributing](#contributing)
- [License](#license)

---

## Why XAURA?

| Problem | XAURA's Solution |
|---|---|
| Writing the same boilerplate for every ML project | One function call: `run_model(data, profile)` |
| Forgetting what hyperparameters you used last week | Automatic SQLite experiment logging — every run is tracked |
| Generic plots that don't match your model type | Model-aware visualisations — only relevant plots are shown |
| Hardcoded defaults that ignore your data | Dataset-aware defaults computed from your actual data |
| Scattered results across notebooks | Unified web UI with sortable experiment log, plot viewer, and export |

---

## Key Features

- 🔍 **Automatic Dataset Profiling** — shape, types, class balance, correlations, missing values, and warnings
- 🎯 **Dataset-Aware Defaults** — hyperparameters adapt based on your data's characteristics
- 📊 **Model-Specific Visualisations** — confusion matrices for classifiers, residual plots for regressors, silhouette plots for clusterers
- 💾 **Silent Experiment Tracking** — every run auto-logged to SQLite with full reproducibility
- 🌐 **Local Web UI** — FastAPI-powered dashboard served at `localhost:8000`
- 📦 **One-Click Export** — PNG plots, JSON configs, ZIP run bundles, CSV experiment logs
- 🖥️ **CLI Interface** — profile, run models, and export from the terminal
- ⚡ **CPU-Optimised** — all Phase 1 models run efficiently on CPU (no GPU required)

---

## Tech Stack

### Core Library

| Component | Technology | Purpose |
|---|---|---|
| Language | **Python 3.10+** | Core runtime |
| ML Models | **scikit-learn** | Logistic Regression, Random Forest, Ridge/Lasso, K-Means, DBSCAN, Hierarchical |
| Gradient Boosting | **XGBoost**, **LightGBM** | High-performance classifiers & regressors |
| Data Handling | **pandas**, **numpy** | DataFrames, arrays, preprocessing |
| Statistical Analysis | **scipy.stats** | Profiling statistics, normality tests |

### Visualisation

| Component | Technology | Purpose |
|---|---|---|
| Interactive Plots (UI) | **Plotly.js** (via CDN) | Zoomable, hoverable browser charts |
| Static Plots (Export) | **Matplotlib**, **seaborn** | Publication-quality PNG/PDF |

### Server & UI

| Component | Technology | Purpose |
|---|---|---|
| API Server | **FastAPI** + **Uvicorn** | REST API + static file serving |
| Templating | **Jinja2** | Server-rendered HTML pages |
| Frontend | **Vanilla JS** + **CSS** | No build step, no Node.js dependency |
| Interactive Charts | **Plotly.js** (CDN) | Client-side plot rendering |

### Storage & Serialisation

| Component | Technology | Purpose |
|---|---|---|
| Experiment Store | **SQLite** (stdlib `sqlite3`) | Zero-config, file-portable logging |
| Model Serialisation | **joblib** | scikit-learn model persistence |
| Config/Metrics | **JSON** | Human-readable, portable |

### Development & CI

| Component | Technology | Purpose |
|---|---|---|
| Testing | **pytest** + **pytest-cov** | Unit & integration tests |
| CLI | **click** or **typer** | Terminal commands |
| Packaging | **pyproject.toml** + **setuptools** | Modern Python packaging |
| CI/CD | **GitHub Actions** | Automated test/lint on PR |
| Linting | **ruff** | Fast Python linter |
| Formatting | **black** | Consistent code style |

---

## Architecture

XAURA is built with strict layer separation. Each component has a clear responsibility and there are no circular dependencies.

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │   Web UI      │  │   CLI        │  │   Python API       │ │
│  │  (Browser)    │  │  (Terminal)  │  │   (import xaura)   │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘ │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI SERVER                            │
│         Routes: /profile  /run  /experiments  /export        │
│         Serves: REST API + Static UI assets                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     CORE LIBRARY (xaura/)                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Profiler  │  │  Models   │  │   Viz     │  │  Export    │  │
│  │          │  │          │  │           │  │           │  │
│  │profile() │  │run_model()│ │plotly_json│  │zip_bundle │  │
│  │DataProfile│ │Result obj │  │matplotlib │  │csv_log    │  │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     STORE (SQLite)                            │
│              Experiment log: runs, metrics, configs           │
│              File: xaura_experiments.db                       │
└─────────────────────────────────────────────────────────────┘
```

### Layer Rules

| Layer | Can Import | Cannot Import |
|---|---|---|
| `xaura/` (core) | `xaura/store/` | `xaura/server/`, `xaura/agent/` |
| `xaura/server/` | `xaura/`, `xaura/store/` | `xaura/agent/` |
| `xaura/agent/` (Phase 2) | `xaura/` | `xaura/server/`, `xaura/store/` |
| `xaura/store/` | stdlib only | anything else |

---

## Project Structure

```
xaura/
├── pyproject.toml                    # Package config, dependencies, entry points
├── README.md
├── LICENSE
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions: test + lint on PR
│
├── src/
│   └── xaura/
│       ├── __init__.py               # Public API: profile(), run_model()
│       ├── cli.py                    # CLI entry points (xaura profile, run, serve, export)
│       │
│       ├── profiler/
│       │   ├── __init__.py
│       │   ├── profiler.py           # profile() function implementation
│       │   └── dataprofile.py        # DataProfile dataclass
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py               # BaseModel ABC, Result dataclass
│       │   ├── registry.py           # Model name → class mapping
│       │   ├── defaults.py           # Dataset-aware default engine
│       │   ├── classifiers/
│       │   │   ├── __init__.py
│       │   │   ├── logistic.py       # Logistic Regression
│       │   │   ├── random_forest.py  # Random Forest Classifier
│       │   │   ├── xgboost_cls.py    # XGBoost Classifier
│       │   │   └── lightgbm_cls.py   # LightGBM Classifier
│       │   ├── regressors/
│       │   │   ├── __init__.py
│       │   │   ├── linear.py         # Linear Regression
│       │   │   ├── ridge_lasso.py    # Ridge & Lasso
│       │   │   ├── random_forest_reg.py
│       │   │   └── xgboost_reg.py    # XGBoost Regressor
│       │   └── clusterers/
│       │       ├── __init__.py
│       │       ├── kmeans.py
│       │       ├── dbscan.py
│       │       └── hierarchical.py   # Agglomerative Clustering
│       │
│       ├── visualisation/
│       │   ├── __init__.py
│       │   ├── plotly_charts.py      # Plotly JSON generators (for UI)
│       │   └── matplotlib_charts.py  # Static PNG/PDF generators (for export)
│       │
│       ├── store/
│       │   ├── __init__.py
│       │   └── sqlite_store.py       # SQLite read/write operations
│       │
│       ├── export/
│       │   ├── __init__.py
│       │   └── exporter.py           # ZIP bundles, CSV logs
│       │
│       └── server/
│           ├── __init__.py
│           ├── app.py                # FastAPI application
│           ├── routes/
│           │   ├── __init__.py
│           │   ├── profile_routes.py
│           │   ├── model_routes.py
│           │   ├── experiment_routes.py
│           │   └── export_routes.py
│           ├── static/
│           │   ├── css/
│           │   │   └── style.css
│           │   └── js/
│           │       ├── app.js        # File upload, navigation
│           │       ├── plots.js      # Plotly rendering logic
│           │       └── experiments.js # Experiment table logic
│           └── templates/
│               ├── base.html
│               ├── index.html        # Landing / upload page
│               ├── profile.html      # Dataset profile view
│               ├── run.html          # Model runner + results
│               └── experiments.html  # Experiment log table
│
└── tests/
    ├── conftest.py                   # Shared fixtures (sample datasets)
    ├── test_profiler.py
    ├── test_classifiers.py
    ├── test_regressors.py
    ├── test_clusterers.py
    ├── test_store.py
    ├── test_export.py
    └── test_api.py
```

---

## Data Flow

```
                    ┌─────────────┐
                    │  CSV / Data  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  profile()   │
                    │              │
                    │ • Shape      │
                    │ • Types      │
                    │ • Balance    │
                    │ • Missing    │
                    │ • Corr       │
                    │ • Stats      │
                    │ • Warnings   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ DataProfile  │──────────────────────┐
                    └──────┬──────┘                       │
                           │                              │
                    ┌──────▼──────┐                ┌──────▼──────┐
                    │  defaults()  │                │  Show in UI  │
                    │              │                │  (summary    │
                    │ Data-aware   │                │   panel)     │
                    │ config       │                └─────────────┘
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  run_model(data, profile) │
              │                          │
              │  • Train/test split      │
              │  • Fit model             │
              │  • Compute metrics       │
              │  • Generate plots        │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │ Result Object│
                    │              │
                    │ • metrics    │──→ SQLite Store (auto)
                    │ • plots      │──→ UI Rendering (Plotly)
                    │ • weights    │──→ Export (joblib)
                    │ • run_id     │──→ Experiment Log
                    │ • config_used│──→ JSON Export
                    └─────────────┘
```

---

## Phase 1 — MVP Scope (CPU-Only)

### 1. Dataset Profiling

```python
from xaura import profile

data_profile = profile(df)    # or profile("path/to/data.csv")

# Returns a DataProfile dataclass:
# - shape: (rows, cols)
# - feature_types: {'numeric': [...], 'categorical': [...], 'binary': [...]}
# - class_balance: {'class_0': 3200, 'class_1': 1000, 'ratio': 3.2}
# - missing_values: {'col_a': 15, 'col_b': 0, ...}
# - correlations: pd.DataFrame correlation matrix
# - basic_stats: pd.DataFrame (mean, std, min, max, skew)
# - warnings: ["High imbalance: 3.2:1", "12% missing in 'age'"]
```

### 2. Supported Models

#### Classification (CPU)
| Model | Wrapper Function | Backend |
|---|---|---|
| Logistic Regression | `run_logistic_classifier` | scikit-learn |
| Random Forest | `run_rf_classifier` | scikit-learn |
| XGBoost | `run_xgb_classifier` | XGBoost |
| LightGBM | `run_lgbm_classifier` | LightGBM |

#### Regression (CPU)
| Model | Wrapper Function | Backend |
|---|---|---|
| Linear Regression | `run_linear_regressor` | scikit-learn |
| Ridge / Lasso | `run_ridge_regressor`, `run_lasso_regressor` | scikit-learn |
| Random Forest | `run_rf_regressor` | scikit-learn |
| XGBoost | `run_xgb_regressor` | XGBoost |

#### Clustering (CPU)
| Model | Wrapper Function | Backend |
|---|---|---|
| K-Means | `run_kmeans` | scikit-learn |
| DBSCAN | `run_dbscan` | scikit-learn |
| Agglomerative | `run_hierarchical` | scikit-learn |

### 3. Dataset-Aware Defaults

The library inspects your `DataProfile` and computes intelligent defaults:

| Data Condition | Automatic Adjustment |
|---|---|
| Small dataset (< 1k rows) | Stronger regularisation, cross-validation enabled |
| Large dataset (> 100k rows) | Mini-batch processing, early stopping |
| Imbalanced classes (> 5:1) | Auto class weights, F1 as default metric |
| High-cardinality categoricals | Target-encoding recommended over one-hot |
| High missing values (> 20%) | Tree-based models preferred, imputation flagged |
| Many correlated features | L1 regularisation, dimensionality warning |

### 4. Model-Aware Visualisations

Each model type renders **only the plots relevant to it**:

| Model Type | Visualisations |
|---|---|
| Classification | Confusion matrix, ROC curve (per class), PR curve, feature importance |
| Regression | Residuals vs fitted, Q-Q plot, predicted vs actual, residual distribution |
| Clustering | Cluster scatter (PCA 2D), silhouette plot, elbow curve, dendrogram |
| All models | Dataset profile summary, metrics card, config panel |

### 5. Experiment Tracking

Every `run_model()` call auto-logs to SQLite:

| Field | Description |
|---|---|
| `run_id` | UUID — unique identifier |
| `timestamp` | ISO 8601 datetime |
| `model_type` | e.g., `random_forest_classifier` |
| `dataset_hash` | SHA-256 fingerprint for reproducibility |
| `config_used` | Full parameter dict (defaults + overrides) |
| `metrics` | All evaluation metrics |
| `tags` | User-defined labels for filtering |
| `notes` | Optional text annotation |

### 6. Export

- **Plots** → PNG or PDF (one plot or full set)
- **Run bundle** → ZIP containing: model weights (joblib), config (JSON), metrics (JSON), dataset profile
- **Experiment log** → Full SQLite log as CSV

### 7. Local Web UI

A clean, functional dashboard served by FastAPI at `localhost:8000`:

- **Upload page** — drag-and-drop CSV upload
- **Profile view** — dataset summary with interactive charts
- **Model runner** — select model, configure params, run, view results
- **Experiment log** — sortable/filterable table of all past runs
- **Run comparison** — side-by-side diff of two runs
- **Export buttons** — one-click download of plots, bundles, logs

### 8. CLI Interface

```bash
xaura profile data.csv              # Profile a dataset, print summary
xaura run rf_classifier data.csv    # Run a model from terminal
xaura serve                         # Start the web UI at localhost:8000
xaura export <run_id>               # Export a run bundle as ZIP
```

---

## Phase 2 — Agentic Layer (Future)

> Phase 2 is **optional** and sits on top of Phase 1. Phase 1 is fully functional without it.

- 🤖 **Conversational interface** — describe what you want in plain language
- 📥 **Multi-source data ingestion** — file path, URL, database connection string
- 💡 **Model recommendation** — suggests 2-3 models based on DataProfile
- 📝 **Plain-language explanations** — what the metrics mean + concrete next steps
- 🔧 **Hyperparameter suggestions** — data-driven, explained, not random
- 🧠 **LLM-backed** — Claude API or local model

---

## Installation

```bash
# Install from PyPI (once published)
pip install xaura

# Or install from source
git clone https://github.com/Vinamra3215/Xaura.git
cd Xaura
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- No GPU required (Phase 1 is CPU-only)
- ~200 MB disk space for dependencies

---

## Quick Start

### Python API

```python
import pandas as pd
from xaura import profile, run_model

# Load data
df = pd.read_csv("data.csv")

# Step 1: Profile
data_profile = profile(df)
print(data_profile.warnings)  # ["High imbalance: 3.2:1"]

# Step 2: Run a model (dataset-aware defaults applied automatically)
result = run_model("rf_classifier", df, data_profile)

# Step 3: Inspect results
print(result.metrics)       # {'accuracy': 0.91, 'f1': 0.85, 'recall': 0.78, ...}
print(result.config_used)   # Full config with all defaults resolved
print(result.run_id)        # 'a3f8c21d-...' — logged to SQLite automatically

# Step 4: Override defaults if needed
result2 = run_model("xgb_classifier", df, data_profile, config={
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.01
})
```

### Web UI

```bash
xaura serve
# Open http://localhost:8000 in your browser
```

### CLI

```bash
xaura profile data.csv
xaura run rf_classifier data.csv --config '{"n_estimators": 200}'
xaura export a3f8c21d
```

---

## API Reference

### `profile(data) → DataProfile`

| Parameter | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame`, `str`, `np.ndarray` | Dataset or path to CSV |

### `run_model(model_name, data, profile, config=None) → Result`

| Parameter | Type | Description |
|---|---|---|
| `model_name` | `str` | Model identifier (e.g., `"rf_classifier"`) |
| `data` | `pd.DataFrame` | Dataset |
| `profile` | `DataProfile` | From `profile()` call |
| `config` | `dict`, optional | Hyperparameter overrides |

### `Result` Object

| Attribute | Type | Description |
|---|---|---|
| `metrics` | `dict` | Evaluation metrics |
| `plots` | `list` | Plotly JSON chart objects |
| `weights` | `object` | Trained model (serialisable) |
| `run_id` | `str` | UUID in experiment log |
| `config_used` | `dict` | Full resolved config |

---

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Vinamra3215/Xaura.git
cd Xaura

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src/xaura

# Run linter
ruff check src/

# Start dev server
xaura serve --reload
```

---

## Development Roadmap & Work Division

The project is developed by two contributors working in parallel. The division ensures **both developers touch every layer** (core library, store, visualisation, server/UI, tests).

### Sprint Overview

| Sprint | Duration | Focus |
|---|---|---|
| Sprint 1 | Week 1-2 | Project setup, profiling, store |
| Sprint 2 | Week 3-4 | Model wrappers (classifiers + regressors) |
| Sprint 3 | Week 5-6 | Clustering, visualisation, export |
| Sprint 4 | Week 7-8 | FastAPI server, web UI, CLI |
| Sprint 5 | Week 9-10 | Integration testing, docs, polish |

### Work Assignment

<details>
<summary><strong>Sprint 1 — Foundation & Profiling (Week 1-2)</strong></summary>

| Task | Person A | Person B |
|---|---|---|
| Project setup | `pyproject.toml`, structure, CI skeleton | README, LICENSE, dev environment, pre-commit |
| DataProfile | Core `profile()` logic (shape, types, stats) | Profile extensions (balance, correlations, missing, warnings) |
| Store | SQLite schema, `create_run()`, `get_run()` | `list_runs()`, `delete_run()`, `get_metrics_comparison()` |
| Tests | Profiler core tests | Store operation tests |

</details>

<details>
<summary><strong>Sprint 2 — Models (Week 3-4)</strong></summary>

| Task | Person A | Person B |
|---|---|---|
| Infrastructure | `BaseModel` ABC, `Result` dataclass, registry | `defaults.py` — dataset-aware default engine |
| Classifiers | Logistic Regression + Random Forest | XGBoost + LightGBM |
| Regressors | Linear Regression + Ridge/Lasso | Random Forest Regressor + XGBoost Regressor |
| Tests | Tests for A's models + integration test | Tests for B's models + integration test |

</details>

<details>
<summary><strong>Sprint 3 — Visualisation & Export (Week 5-6)</strong></summary>

| Task | Person A | Person B |
|---|---|---|
| Clusterers | K-Means + DBSCAN | Hierarchical Clustering |
| Plotly charts | Confusion matrix, ROC, PR, feature importance | Residuals, Q-Q, predicted-vs-actual, cluster plots |
| Matplotlib | Classification static plots | Regression + clustering static plots |
| Export | ZIP bundle exporter | CSV log exporter |
| Tests | Classification vis + clustering tests | Regression vis + export tests |

</details>

<details>
<summary><strong>Sprint 4 — Server & UI (Week 7-8)</strong></summary>

| Task | Person A | Person B |
|---|---|---|
| FastAPI | `app.py`, profile routes, model routes | Experiment routes, export routes |
| Templates | `base.html`, `index.html`, `profile.html` | `run.html`, `experiments.html` |
| JavaScript | `app.js`, `plots.js` | `experiments.js` |
| CSS | Pair program on `style.css` | Pair program on `style.css` |
| CLI | `xaura profile`, `xaura run` | `xaura serve`, `xaura export` |
| Tests | Profile + model API tests | Experiment + export API tests |

</details>

<details>
<summary><strong>Sprint 5 — Polish & Release (Week 9-10)</strong></summary>

| Task | Person A | Person B |
|---|---|---|
| Testing | End-to-end flow tests | Edge case tests |
| Docs | API docs + docstrings | User guide / tutorial |
| README | Final README polish | Contributing guide + changelog |
| CI/CD | Test + lint workflow | Build + publish workflow |

</details>

### Cross-Learning Rule

After every sprint, both contributors:
1. **Code review** each other's PRs
2. **Write one test** for each other's code
3. **Demo** their work to each other with a walkthrough

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/model-name`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a PR with a clear description

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>XAURA</strong> — Because ML should be intelligent about your data, not just your model.
</p>