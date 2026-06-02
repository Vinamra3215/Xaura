# XAURA — 6-Week Implementation Plan (Phase 1, CPU-Only)

> **eXtendable Automated Unified Research & Analytics**
> A Python-based intelligent ML library with dataset-aware defaults, experiment tracking, and a local web UI.

---

## Overview

This document outlines the complete 6-week build plan for XAURA Phase 1. All models are **CPU-only** (scikit-learn, XGBoost, LightGBM). Deep learning models (PyTorch/TensorFlow) are deferred to Phase 2.

Two contributors (**Person A** and **Person B**) work in parallel, with tasks divided so that **both touch every layer** — core library, store, visualisation, server/UI, and tests. Neither becomes a single-area specialist.

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Data | pandas, numpy, scipy |
| Visualisation (UI) | Plotly.js (CDN) |
| Visualisation (Export) | Matplotlib, seaborn |
| API Server | FastAPI + Uvicorn |
| UI | Jinja2 + Vanilla JS + CSS |
| Experiment Store | SQLite (stdlib) |
| Serialisation | joblib (models), JSON (config/metrics) |
| CLI | click |
| Testing | pytest + pytest-cov |
| CI | GitHub Actions |

---

## Week 1 — Project Foundation & Data Profiling

### Goals
- Project scaffolding is complete and installable via `pip install -e .`
- `profile()` function works end-to-end on any CSV/DataFrame
- SQLite store schema is defined and basic CRUD works
- CI pipeline runs on every PR

### Person A

| Task | Details |
|---|---|
| **Project setup** | Create `pyproject.toml` with all dependencies, entry points, dev extras. Set up `src/xaura/` layout. |
| **`.gitignore` + CI** | Standard Python `.gitignore`. GitHub Actions workflow: test + lint on push/PR for Python 3.10-3.12. |
| **DataProfile dataclass** | Implement `src/xaura/profiler/dataprofile.py` — all fields (shape, feature_types, class_balance, missing_values, correlations, basic_stats, warnings), properties (`is_imbalanced`, `is_small`, `has_missing`), and `summary()` method. |
| **Profiler core** | Implement `profile()` in `src/xaura/profiler/profiler.py` — shape detection, feature type inference (numeric/categorical/binary/datetime), basic statistics (mean, std, min, max, skew). |
| **Tests** | `tests/conftest.py` (shared sample datasets), `tests/test_profiler.py` (core profiling logic). |

### Person B

| Task | Details |
|---|---|
| **LICENSE + pre-commit** | MIT License file. Set up `pre-commit-config.yaml` with ruff + black hooks. |
| **Profiler extensions** | Extend `profile()` with: class balance detection, correlation matrix + high-correlation pair flagging (|r| > 0.85), missing value analysis (counts + percentages), and warning generation. |
| **Target column detection** | Heuristic to identify target column and infer task type (classification vs regression). |
| **SQLite store** | `src/xaura/store/sqlite_store.py` — schema design, `init_db()`, `create_run()`, `get_run()`, `list_runs()`, `delete_run()`. |
| **Tests** | `tests/test_store.py` (all CRUD operations, edge cases). |

### Deliverables
```
src/xaura/
├── __init__.py
├── profiler/
│   ├── __init__.py
│   ├── dataprofile.py      ✅
│   └── profiler.py          ✅
└── store/
    ├── __init__.py
    └── sqlite_store.py      ✅

tests/
├── conftest.py              ✅
├── test_profiler.py         ✅
└── test_store.py            ✅
```

---

## Week 2 — Model Infrastructure & Classifiers

### Goals
- `BaseModel` and `Result` abstractions are solid
- All 4 classifiers work with dataset-aware defaults
- Every model run is auto-logged to SQLite

### Person A

| Task | Details |
|---|---|
| **BaseModel ABC** | `src/xaura/models/base.py` — abstract base with `fit()`, `predict()`, `evaluate()`. `Result` dataclass (metrics, plots, weights, run_id, config_used). |
| **Model registry** | `src/xaura/models/registry.py` — `run_model(name, data, profile, config)` dispatcher. `list_models()` returns available models. |
| **Logistic Regression** | `src/xaura/models/classifiers/logistic.py` — wraps scikit-learn, applies dataset-aware defaults. |
| **Random Forest Classifier** | `src/xaura/models/classifiers/random_forest.py` — same pattern. |
| **Tests** | Tests for both classifiers + integration test (profile → run → check result → verify SQLite entry). |

### Person B

| Task | Details |
|---|---|
| **Dataset-aware defaults engine** | `src/xaura/models/defaults.py` — reads DataProfile, computes config: regularisation strength, class weights, CV folds, metric selection, etc. |
| **XGBoost Classifier** | `src/xaura/models/classifiers/xgboost_cls.py` — wraps XGBoost with auto class weights, scale_pos_weight from DataProfile. |
| **LightGBM Classifier** | `src/xaura/models/classifiers/lightgbm_cls.py` — same pattern. |
| **Auto-logging integration** | Wire up model runs to automatically call `store.create_run()` after every `run_model()`. |
| **Tests** | Tests for both classifiers + defaults engine + auto-logging integration test. |

### Deliverables
```
src/xaura/models/
├── __init__.py
├── base.py                  ✅
├── registry.py              ✅
├── defaults.py              ✅
└── classifiers/
    ├── __init__.py
    ├── logistic.py           ✅
    ├── random_forest.py      ✅
    ├── xgboost_cls.py        ✅
    └── lightgbm_cls.py       ✅

tests/
├── test_classifiers.py      ✅
└── test_defaults.py         ✅
```

---

## Week 3 — Regressors, Clusterers & Visualisation

### Goals
- All regressors and clusterers work
- Plotly JSON chart generators produce interactive visualisations
- Matplotlib generators produce export-quality static plots

### Person A

| Task | Details |
|---|---|
| **Linear Regression** | `src/xaura/models/regressors/linear.py` |
| **Ridge / Lasso** | `src/xaura/models/regressors/ridge_lasso.py` — both models in one file, selected via config. |
| **K-Means** | `src/xaura/models/clusterers/kmeans.py` — includes elbow method for auto-k selection. |
| **DBSCAN** | `src/xaura/models/clusterers/dbscan.py` — eps estimation from DataProfile. |
| **Plotly: Classification charts** | `src/xaura/visualisation/plotly_charts.py` — confusion matrix, ROC curve (per-class), Precision-Recall curve, feature importance bar chart. |
| **Matplotlib: Classification charts** | `src/xaura/visualisation/matplotlib_charts.py` — static PNG/PDF versions of the same. |
| **Tests** | Tests for A's models + classification visualisation output validation. |

### Person B

| Task | Details |
|---|---|
| **Random Forest Regressor** | `src/xaura/models/regressors/random_forest_reg.py` |
| **XGBoost Regressor** | `src/xaura/models/regressors/xgboost_reg.py` |
| **Hierarchical Clustering** | `src/xaura/models/clusterers/hierarchical.py` — Agglomerative with dendrogram support. |
| **Plotly: Regression charts** | Residuals vs fitted, Q-Q plot, predicted vs actual scatter, residual distribution histogram. |
| **Plotly: Clustering charts** | Cluster scatter (PCA 2D projection), silhouette score plot, elbow curve, dendrogram. |
| **Matplotlib: Regression + Clustering** | Static versions for export. |
| **Tests** | Tests for B's models + regression/clustering visualisation output validation. |

### Deliverables
```
src/xaura/models/
├── regressors/
│   ├── __init__.py
│   ├── linear.py             ✅
│   ├── ridge_lasso.py        ✅
│   ├── random_forest_reg.py  ✅
│   └── xgboost_reg.py        ✅
└── clusterers/
    ├── __init__.py
    ├── kmeans.py              ✅
    ├── dbscan.py              ✅
    └── hierarchical.py        ✅

src/xaura/visualisation/
├── __init__.py
├── plotly_charts.py           ✅
└── matplotlib_charts.py       ✅
```

---

## Week 4 — Export, CLI & FastAPI Server

### Goals
- Export system works (ZIP bundles, CSV logs, PNG/PDF plots)
- CLI commands are functional (`xaura profile`, `xaura run`, `xaura serve`, `xaura export`)
- FastAPI server is up with all REST routes

### Person A

| Task | Details |
|---|---|
| **Export: ZIP bundle** | `src/xaura/export/exporter.py` — packages model weights (joblib), config (JSON), metrics (JSON), dataset profile into a ZIP. |
| **Export: CSV log** | Export full SQLite experiment log as CSV. |
| **CLI: `xaura profile`** | Profile a dataset from terminal, print summary. |
| **CLI: `xaura run`** | Run a model from terminal with optional `--config` JSON override. |
| **FastAPI: app.py** | Create the FastAPI application, mount static files, configure Jinja2. |
| **FastAPI: profile routes** | `POST /api/profile` (upload CSV → return DataProfile JSON), `GET /api/profile/{id}`. |
| **FastAPI: model routes** | `POST /api/run` (run a model → return Result JSON), `GET /api/models` (list available models). |

### Person B

| Task | Details |
|---|---|
| **Export: PNG/PDF plots** | Export individual or all plots as PNG/PDF from a Result object. |
| **Experiment comparison** | `store.get_metrics_comparison(run_ids)` — side-by-side metrics comparison for multiple runs. |
| **CLI: `xaura serve`** | Start the FastAPI dev server. |
| **CLI: `xaura export`** | Export a run bundle by run_id. |
| **FastAPI: experiment routes** | `GET /api/experiments` (list runs, filterable), `GET /api/experiments/{run_id}`, `DELETE /api/experiments/{run_id}`, `GET /api/experiments/compare?ids=...`. |
| **FastAPI: export routes** | `GET /api/export/{run_id}/zip`, `GET /api/export/{run_id}/plots`, `GET /api/export/log/csv`. |
| **Tests** | API endpoint tests using `httpx` + FastAPI TestClient. |

### Deliverables
```
src/xaura/
├── cli.py                     ✅
├── export/
│   ├── __init__.py
│   └── exporter.py            ✅
└── server/
    ├── __init__.py
    ├── app.py                 ✅
    └── routes/
        ├── __init__.py
        ├── profile_routes.py  ✅
        ├── model_routes.py    ✅
        ├── experiment_routes.py ✅
        └── export_routes.py   ✅

tests/
├── test_export.py             ✅
└── test_api.py                ✅
```

---

## Week 5 — Web UI (Frontend)

### Goals
- Fully functional browser UI at `localhost:8000`
- Users can upload data, view profiles, run models, see results, browse experiment history
- All plots render interactively via Plotly.js

### Person A

| Task | Details |
|---|---|
| **`base.html`** | Jinja2 base template: navigation bar, footer, Plotly.js CDN, CSS/JS includes. |
| **`index.html`** | Landing page with drag-and-drop CSV upload, project description. |
| **`profile.html`** | Dataset profile view: stats table, feature type breakdown, missing values heatmap, correlation matrix, warnings panel. |
| **`app.js`** | File upload handler (FormData → fetch to `/api/profile`), navigation logic, loading states. |
| **`plots.js`** | Plotly rendering: receives plot JSON from API, renders into DOM containers. |
| **`style.css` (shared)** | Pair-program with Person B on the full stylesheet. Clean, dark-mode, functional design. |

### Person B

| Task | Details |
|---|---|
| **`run.html`** | Model runner page: model selector dropdown, config editor (JSON), run button, results panel (metrics card + plots). |
| **`experiments.html`** | Experiment log table: sortable columns, search/filter, click-to-expand run details, side-by-side comparison view, delete button. |
| **`experiments.js`** | Table rendering, sorting, filtering, run comparison logic, export buttons. |
| **`style.css` (shared)** | Pair-program with Person A. |
| **Responsive design** | Ensure all pages work on tablet/desktop widths. |

### Deliverables
```
src/xaura/server/
├── templates/
│   ├── base.html              ✅
│   ├── index.html             ✅
│   ├── profile.html           ✅
│   ├── run.html               ✅
│   └── experiments.html       ✅
└── static/
    ├── css/
    │   └── style.css          ✅
    └── js/
        ├── app.js             ✅
        ├── plots.js           ✅
        └── experiments.js     ✅
```

---

## Week 6 — Integration Testing, Docs & Release Prep

### Goals
- Full end-to-end flow tested
- Edge cases handled gracefully
- Documentation complete
- Package installable via `pip install`

### Person A

| Task | Details |
|---|---|
| **End-to-end tests** | Test complete flow: upload CSV → profile → select model → run → view results → export. Use multiple sample datasets (Iris, Boston, synthetic). |
| **Error handling** | Graceful handling of: malformed CSV, empty datasets, single-column data, all-NaN columns, unsupported file types. |
| **API documentation** | Comprehensive docstrings on all public functions. Usage examples in docstrings. |
| **README final polish** | Badges, GIFs/screenshots of UI, installation verification. |

### Person B

| Task | Details |
|---|---|
| **Edge case tests** | Large files (100k+ rows), datasets with only categoricals, datasets with 50%+ missing, single-class target, highly imbalanced (100:1). |
| **Input validation** | Add validation to all API endpoints and model functions. Clear error messages. |
| **User guide** | `docs/user_guide.md` — tutorial walkthrough with a sample dataset, step-by-step. |
| **Contributing guide** | `CONTRIBUTING.md` — how to add a new model, coding standards, PR process. |
| **CI/CD finalization** | Ensure all tests pass in CI. Add badge to README. |

### Deliverables
```
docs/
├── user_guide.md              ✅
└── CONTRIBUTING.md            ✅

tests/
├── test_e2e.py                ✅
├── test_edge_cases.py         ✅
└── test_validation.py         ✅
```

---

## Weekly Rituals

| Activity | When | Details |
|---|---|---|
| **Code Review** | End of each week | Both review each other's PRs — this is how you learn code you didn't write |
| **Cross-Testing** | End of each week | Each person writes 1-2 tests for the other's code |
| **Demo** | Friday | Both demo their week's work, explain design decisions |
| **Retro** | Friday | Quick check: what went well, what's blocking, what to adjust |

---

## Definition of Done (per week)

- [ ] All code has docstrings
- [ ] All new functions have at least one test
- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] CI is green on the PR
- [ ] Code reviewed by the other person
- [ ] PR merged to `main`

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| XGBoost/LightGBM install issues | Pin specific versions in `pyproject.toml`, test in CI across OS |
| Plotly charts too complex | Start with basic charts, enhance interactivity iteratively |
| Scope creep | Strictly follow this plan. New ideas go to a `BACKLOG.md` file |
| One person blocked | Both know enough of each layer to help — that's why work is cross-cutting |
| SQLite concurrency | Single-user local use, not a concern for Phase 1 |

---

## Post-Phase 1: What Comes Next

After the 6-week MVP is complete and stable:

- **Phase 1.5** — Add MLP (via scikit-learn) as a lightweight neural net option
- **Phase 2** — Agentic layer (LLM-backed chatbot, model recommendation, hyperparameter suggestions)
- **Phase 3** — Desktop packaging (.exe, .dmg, AppImage)
- **Phase 4** — GPU models (PyTorch/TensorFlow: CNN, LSTM, Transformers)
