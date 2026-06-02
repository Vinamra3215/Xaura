# XAURA — 6-Week Implementation Plan (Phase 1 + Phase 2)

> **eXtendable Automated Unified Research & Analytics**
> A Python-based intelligent ML library with dataset-aware defaults, experiment tracking, a local web UI, and an agentic conversational interface.

---

## Overview

This document outlines the complete 6-week build plan covering **both phases** of XAURA:

- **Phase 1 (Weeks 1–4):** Core library, models, visualisation, experiment tracking, FastAPI server, web UI, CLI
- **Phase 2 (Weeks 5–6):** Agentic layer — conversational interface, data ingestion, model recommendation, plain-language explanations, hyperparameter suggestions

All Phase 1 models are **CPU-only** (scikit-learn, XGBoost, LightGBM). Phase 2 adds an LLM-backed agent that orchestrates Phase 1 functions via natural language.

Two contributors (**Person A** and **Person B**) work in parallel. Tasks are divided so **both touch every layer** — core library, store, visualisation, server/UI, agent, and tests.

---

## Tech Stack Summary

### Phase 1 — Core Library & UI

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

### Phase 2 — Agentic Layer

| Layer | Technology |
|---|---|
| LLM Backend | Google Gemini API / OpenAI API (switchable) |
| Conversation Management | Custom session handler (in-memory + SQLite) |
| Data Ingestion | pandas (CSV/Excel/Parquet/JSON), sqlalchemy (DB connections), requests (URL download) |
| Agent Framework | Custom lightweight agent (no LangChain — keeps it simple and learnable) |
| Chat UI | WebSocket via FastAPI + vanilla JS chat interface |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                         │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Web UI     │  │  Chat UI   │  │   CLI    │  │ Python API │  │
│  │ (Dashboard) │  │  (Agent)   │  │          │  │            │  │
│  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  └─────┬──────┘  │
└────────┼───────────────┼──────────────┼──────────────┼──────────┘
         │               │              │              │
         ▼               ▼              ▼              ▼
┌────────────────┐  ┌─────────────────────────────────────────────┐
│  FastAPI Server │  │            AGENT LAYER (Phase 2)            │
│  REST Routes    │  │  • Data ingestion (file/URL/DB)             │
│  WebSocket      │  │  • Model recommendation                    │
│  Static Files   │  │  • Result explanation                      │
│                 │  │  • Hyperparameter suggestion                │
│                 │  │  • Conversation history                     │
│                 │  │                                             │
│                 │  │  Calls Phase 1 functions — never bypasses   │
└────────┬────────┘  └──────────────────┬──────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CORE LIBRARY (Phase 1)                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐      │
│  │ Profiler  │  │  Models   │  │   Viz     │  │  Export    │      │
│  │profile()  │  │run_model()│  │plotly/mpl │  │zip/csv/png│      │
│  │DataProfile│  │Result     │  │           │  │           │      │
│  └──────────┘  └──────────┘  └───────────┘  └───────────┘      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        STORE (SQLite)                            │
│         Experiments · Conversation History · Data Cache          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure (Final — Both Phases)

```
xaura/
├── pyproject.toml
├── README.md
├── IMPLEMENTATION_PLAN.md
├── LICENSE
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   └── xaura/
│       ├── __init__.py               # Public API
│       ├── cli.py                    # CLI commands
│       │
│       ├── profiler/                 # Week 1
│       │   ├── __init__.py
│       │   ├── profiler.py
│       │   └── dataprofile.py
│       │
│       ├── models/                   # Week 2
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── defaults.py
│       │   ├── classifiers/
│       │   │   ├── logistic.py
│       │   │   ├── random_forest.py
│       │   │   ├── xgboost_cls.py
│       │   │   └── lightgbm_cls.py
│       │   ├── regressors/
│       │   │   ├── linear.py
│       │   │   ├── ridge_lasso.py
│       │   │   ├── random_forest_reg.py
│       │   │   └── xgboost_reg.py
│       │   └── clusterers/
│       │       ├── kmeans.py
│       │       ├── dbscan.py
│       │       └── hierarchical.py
│       │
│       ├── visualisation/            # Week 3
│       │   ├── __init__.py
│       │   ├── plotly_charts.py
│       │   └── matplotlib_charts.py
│       │
│       ├── store/                    # Week 1
│       │   ├── __init__.py
│       │   └── sqlite_store.py
│       │
│       ├── export/                   # Week 3
│       │   ├── __init__.py
│       │   └── exporter.py
│       │
│       ├── server/                   # Week 4
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── routes/
│       │   │   ├── profile_routes.py
│       │   │   ├── model_routes.py
│       │   │   ├── experiment_routes.py
│       │   │   ├── export_routes.py
│       │   │   └── agent_routes.py   # Week 5
│       │   ├── static/
│       │   │   ├── css/style.css
│       │   │   └── js/
│       │   │       ├── app.js
│       │   │       ├── plots.js
│       │   │       ├── experiments.js
│       │   │       └── chat.js       # Week 5
│       │   └── templates/
│       │       ├── base.html
│       │       ├── index.html
│       │       ├── profile.html
│       │       ├── run.html
│       │       ├── experiments.html
│       │       └── chat.html         # Week 5
│       │
│       └── agent/                    # Weeks 5-6
│           ├── __init__.py
│           ├── engine.py             # Core agent orchestrator
│           ├── llm_client.py         # LLM API wrapper (Gemini/OpenAI)
│           ├── ingestion.py          # Multi-source data loading
│           ├── recommender.py        # Model recommendation logic
│           ├── explainer.py          # Plain-language result explanation
│           ├── tuner.py              # Hyperparameter suggestion engine
│           ├── prompts.py            # System prompts & prompt templates
│           └── session.py            # Conversation history manager
│
└── tests/
    ├── conftest.py
    ├── test_profiler.py
    ├── test_classifiers.py
    ├── test_regressors.py
    ├── test_clusterers.py
    ├── test_store.py
    ├── test_export.py
    ├── test_api.py
    ├── test_agent.py                 # Week 6
    ├── test_ingestion.py             # Week 6
    └── test_e2e.py                   # Week 6
```

---

## Week 1 — Project Foundation, Profiling & Store

### Goals
- Project is installable via `pip install -e .`
- `profile()` works end-to-end on any DataFrame/CSV
- SQLite store handles full CRUD
- CI runs on every PR

### Person A

| Task | Details |
|---|---|
| **Project setup** | `pyproject.toml` with all dependencies (including Phase 2 deps as optional: `google-generativeai`, `openai`, `sqlalchemy`, `requests`). `src/xaura/` layout with all `__init__.py` files. |
| **`.gitignore` + CI** | Python `.gitignore`. GitHub Actions: test + lint for Python 3.10–3.12. |
| **DataProfile dataclass** | `dataprofile.py` — all fields, properties (`is_imbalanced`, `is_small`, `has_missing`), `summary()` method. |
| **Profiler core** | `profile()` — shape detection, feature type inference, basic statistics (mean, std, min, max, skew). |
| **Tests** | `conftest.py` (sample datasets: classification, regression, clustering, edge cases). `test_profiler.py`. |

### Person B

| Task | Details |
|---|---|
| **LICENSE + pre-commit** | MIT License. `pre-commit-config.yaml` with ruff + black. |
| **Profiler extensions** | Class balance detection, correlation matrix, high-correlation flagging (|r| > 0.85), missing value analysis, warning generation. |
| **Target detection** | Heuristic to identify target column + infer task type (classification vs regression). |
| **SQLite store** | `sqlite_store.py` — schema, `init_db()`, `create_run()`, `get_run()`, `list_runs()`, `delete_run()`, `get_metrics_comparison()`. |
| **Tests** | `test_store.py` — all CRUD, edge cases, comparison queries. |

### Week 1 Deliverables
- `xaura.profile(df)` returns a complete `DataProfile`
- `store.create_run()` / `get_run()` / `list_runs()` / `delete_run()` all work
- CI is green

---

## Week 2 — All Models (Classifiers + Regressors + Clusterers)

### Goals
- All 11 models work with dataset-aware defaults
- Every run auto-logs to SQLite
- Unified `run_model(name, data, profile)` dispatcher

### Person A

| Task | Details |
|---|---|
| **BaseModel + Result** | `base.py` — abstract base with `fit()`, `predict()`, `evaluate()`. `Result` dataclass. |
| **Model registry** | `registry.py` — `run_model()` dispatcher, `list_models()`. Auto-logging to SQLite after every run. |
| **Logistic Regression** | `classifiers/logistic.py` |
| **Random Forest Classifier** | `classifiers/random_forest.py` |
| **Linear Regression** | `regressors/linear.py` |
| **Ridge / Lasso** | `regressors/ridge_lasso.py` |
| **K-Means** | `clusterers/kmeans.py` — includes elbow method for auto-k. |
| **Tests** | Tests for all of A's models + profile → run → result integration test. |

### Person B

| Task | Details |
|---|---|
| **Defaults engine** | `defaults.py` — reads DataProfile → computes model config (regularisation, class weights, CV folds, metric selection, early stopping). |
| **XGBoost Classifier** | `classifiers/xgboost_cls.py` |
| **LightGBM Classifier** | `classifiers/lightgbm_cls.py` |
| **Random Forest Regressor** | `regressors/random_forest_reg.py` |
| **XGBoost Regressor** | `regressors/xgboost_reg.py` |
| **DBSCAN** | `clusterers/dbscan.py` — eps estimation from DataProfile. |
| **Hierarchical Clustering** | `clusterers/hierarchical.py` — Agglomerative with dendrogram. |
| **Tests** | Tests for all of B's models + defaults engine test. |

### Week 2 Deliverables
- `run_model("rf_classifier", df, profile)` works for all 11 models
- Dataset-aware defaults auto-computed
- Every run auto-logged to SQLite

---

## Week 3 — Visualisation, Export & CLI

### Goals
- Interactive Plotly charts for all model types
- Static Matplotlib exports (PNG/PDF)
- ZIP bundle + CSV log export
- CLI fully functional

### Person A

| Task | Details |
|---|---|
| **Plotly: Classification** | `plotly_charts.py` — confusion matrix heatmap, ROC curve (per-class), PR curve, feature importance bar chart. |
| **Plotly: Common** | Dataset profile summary panel, metrics card, config panel (shared across all model types). |
| **Matplotlib: Classification** | `matplotlib_charts.py` — static PNG/PDF versions of classification plots. |
| **Export: ZIP bundle** | `exporter.py` — packages weights (joblib) + config (JSON) + metrics (JSON) + profile into ZIP. |
| **CLI: `xaura profile`** | Profile a CSV, print summary to terminal. |
| **CLI: `xaura run`** | Run a model with optional `--config` JSON. |
| **Tests** | Chart output validation + export tests. |

### Person B

| Task | Details |
|---|---|
| **Plotly: Regression** | Residuals vs fitted, Q-Q plot, predicted vs actual scatter, residual distribution histogram. |
| **Plotly: Clustering** | Cluster scatter (PCA 2D), silhouette plot, elbow curve, dendrogram. |
| **Matplotlib: Regression + Clustering** | Static versions for export. |
| **Export: CSV log** | Export full experiment log as CSV. |
| **Export: Plot export** | Export individual/all plots as PNG/PDF. |
| **CLI: `xaura serve`** | Start FastAPI server. |
| **CLI: `xaura export`** | Export run bundle by run_id. |
| **Tests** | Regression/clustering vis tests + CLI smoke tests. |

### Week 3 Deliverables
- All Plotly charts generate valid JSON
- `xaura profile data.csv` / `xaura run rf_classifier data.csv` / `xaura serve` all work
- Export produces valid ZIP/CSV/PNG

---

## Week 4 — FastAPI Server & Web UI

### Goals
- Full web dashboard at `localhost:8000`
- Upload → Profile → Run → View Results → Export flow works end-to-end
- Experiment log with sorting, filtering, comparison

### Person A

| Task | Details |
|---|---|
| **FastAPI: app.py** | Application setup, Jinja2 config, static file mounting, CORS. |
| **Profile routes** | `POST /api/profile` (upload CSV → DataProfile), `GET /api/profile/{id}`. |
| **Model routes** | `POST /api/run`, `GET /api/models`. |
| **`base.html`** | Jinja2 base: nav bar, footer, CDN imports (Plotly.js), CSS/JS includes. |
| **`index.html`** | Landing page: drag-and-drop CSV upload, project description. |
| **`profile.html`** | Profile view: stats table, feature types, missing heatmap, correlation matrix, warnings. |
| **`app.js`** | File upload (FormData → fetch), navigation, loading spinners. |
| **`plots.js`** | Receives Plotly JSON from API, renders into DOM containers. |

### Person B

| Task | Details |
|---|---|
| **Experiment routes** | `GET /api/experiments`, `GET /api/experiments/{id}`, `DELETE`, `GET /api/experiments/compare`. |
| **Export routes** | `GET /api/export/{id}/zip`, `GET /api/export/{id}/plots`, `GET /api/export/log/csv`. |
| **`run.html`** | Model runner: model dropdown, config editor, run button, results panel (metrics + plots). |
| **`experiments.html`** | Experiment log: sortable table, search, click-to-expand, side-by-side diff, delete. |
| **`experiments.js`** | Table rendering, sorting, filtering, comparison, export buttons. |
| **`style.css`** | Full stylesheet — pair program with Person A. Dark mode, clean, functional. |
| **API tests** | `test_api.py` — all endpoints tested via httpx TestClient. |

### Week 4 Deliverables
- `xaura serve` → open `localhost:8000` → full working dashboard
- Upload CSV → see profile → run model → view plots/metrics → export
- Experiment history with comparison

---

## Week 5 — Agent Core & Chat UI

### Goals
- Agent engine orchestrates Phase 1 functions via natural language
- Multi-source data ingestion (file, URL, DB)
- Model recommendation from DataProfile
- Chat UI with WebSocket for real-time conversation

### Person A

| Task | Details |
|---|---|
| **LLM client** | `agent/llm_client.py` — unified wrapper for Gemini API and OpenAI API. Switchable via config/env var. Handles API keys, rate limiting, retries. |
| **Prompt templates** | `agent/prompts.py` — system prompt (defines agent personality, capabilities, constraints), profiling prompt, recommendation prompt, explanation prompt, tuning prompt. |
| **Agent engine** | `agent/engine.py` — core orchestrator. Receives user message → parses intent → calls appropriate Phase 1 function → formats response. Maintains tool-calling loop. |
| **Session manager** | `agent/session.py` — conversation history per session. Store in SQLite (new table). Load/save/list sessions. |
| **Chat UI template** | `templates/chat.html` — chat interface with message bubbles, input box, send button, session selector. |
| **Chat JS** | `static/js/chat.js` — WebSocket connection, message rendering, auto-scroll, typing indicator. |

### Person B

| Task | Details |
|---|---|
| **Data ingestion** | `agent/ingestion.py` — load from: local file path (CSV/Excel/Parquet/JSON with auto-detection), URL (HTTP download + detect type), database connection string (SQLAlchemy: SQLite/PostgreSQL/MySQL → SELECT → DataFrame). Validates and confirms what was loaded. |
| **Model recommender** | `agent/recommender.py` — rule-based layer: reads DataProfile → ranks models by suitability. Rules: imbalanced → XGBoost/LightGBM; small + few features → Logistic; many features → Ridge/Lasso; no target → clustering. LLM layer: generates one-line rationale per recommendation. |
| **Agent routes** | `server/routes/agent_routes.py` — `WebSocket /ws/chat` (real-time conversation), `POST /api/agent/ingest` (data ingestion), `GET /api/agent/sessions` (list sessions), `GET /api/agent/sessions/{id}` (load session). |
| **Nav integration** | Update `base.html` to add "Chat" link in nav bar. Ensure smooth navigation between dashboard and chat. |
| **Tests** | `test_ingestion.py` — file/URL/DB ingestion with mocked sources. |

### Week 5 Deliverables
- Agent can: accept data from file/URL/DB, profile it, recommend models
- Chat UI works via WebSocket at `/chat`
- Conversation history persisted per session

---

## Week 6 — Agent Intelligence, Polish & Release

### Goals
- Agent explains results in plain language
- Agent suggests hyperparameters based on run analysis
- Full end-to-end agent flow tested
- Documentation complete, package ready for release

### Person A

| Task | Details |
|---|---|
| **Result explainer** | `agent/explainer.py` — after a model run, generates plain-language interpretation. Rule layer: identifies key patterns (overfitting, underfitting, class confusion, high/low recall). LLM layer: turns patterns into natural language + suggests next steps. |
| **Hyperparameter tuner** | `agent/tuner.py` — after first run, analyses metrics + DataProfile → suggests 2-3 hyperparameter configs with rationale. Rules: train/val gap → more regularisation; slow convergence → lower LR; class confusion → adjust threshold/weights. LLM: generates Config A/B/C with explanations. |
| **Agent conversation flow** | Full end-to-end: user provides data → agent profiles → recommends model → user picks → agent runs → explains results → suggests hyperparams → user picks → agent re-runs → compares. |
| **End-to-end tests** | `test_e2e.py` — full flow from data upload through agent conversation to export. |
| **README final** | Final README polish: add agent section, chat UI screenshot, full feature list. |

### Person B

| Task | Details |
|---|---|
| **Agent multi-run comparison** | Agent can compare multiple runs and explain which performed better and why. |
| **Error handling & guardrails** | Agent gracefully handles: invalid data, unsupported model names, API failures, empty datasets. Never exposes raw errors to user. |
| **Edge case tests** | `test_agent.py` — malformed inputs, API timeout mocking, conversation with ambiguous requests, session persistence across restarts. |
| **User guide** | `docs/user_guide.md` — tutorial covering both dashboard and agent workflows. |
| **Contributing guide** | `CONTRIBUTING.md` — how to add models, how to modify prompts, coding standards. |
| **CI/CD final** | All tests pass in CI. Add coverage badge. Verify `pip install` works from clean environment. |

### Week 6 Deliverables
- Agent explains results: *"Your RF achieved 89% accuracy but recall on minority class is only 67%..."*
- Agent suggests hyperparams: *"Config A — dropout: 0.4, weight_decay: 1e-4. Rationale: stronger regularisation..."*
- Full conversation flow works
- Documentation complete
- Package installable via `pip install`

---

## Agent Conversation Flow (Week 5–6)

```
User: "I have a dataset at /home/user/sales.csv"
  │
  ▼
Agent: [calls ingestion.load("/home/user/sales.csv")]
Agent: "Loaded sales.csv — 8,400 rows, 12 columns. Running profiling..."
  │
  ▼
Agent: [calls profile(df)]
Agent: "Here's your data profile:
        • 12 numeric features, binary target
        • Class imbalance: 3.2:1
        • No missing values
        • 2 high-correlation pairs flagged"
  │
  ▼
Agent: [calls recommender.recommend(profile)]
Agent: "Based on this profile, recommended models:
        1. XGBoost — handles imbalance well, strong on tabular data
        2. Random Forest — good baseline, interpretable
        3. Logistic Regression — fast, interpretable for low feature count
        Which would you like to run?"
  │
  ▼
User: "Run XGBoost"
  │
  ▼
Agent: [calls run_model("xgb_classifier", df, profile)]
Agent: [calls explainer.explain(result, profile)]
Agent: "Your XGBoost achieved 89.4% accuracy and F1 of 0.83.
        However, minority class recall is only 67% — the model
        misses 1 in 3 positive cases.
        
        Suggested next steps:
        • Try a lower classification threshold
        • Increase scale_pos_weight
        • Consider SMOTE oversampling"
  │
  ▼
Agent: [calls tuner.suggest(result, profile)]
Agent: "Based on the train/val gap, here are 3 configs to try:
        Config A — n_estimators: 500, max_depth: 4, scale_pos_weight: 5
        Config B — n_estimators: 300, max_depth: 6, learning_rate: 0.01
        Config C — n_estimators: 400, max_depth: 5, subsample: 0.8
        Run all three?"
  │
  ▼
User: "Yes, run all"
  │
  ▼
Agent: [runs all 3, compares results]
Agent: "Config B performed best — F1: 0.88, minority recall: 78%.
        All 3 runs are logged. View comparison in the experiments tab."
```

---

## Weekly Rituals

| Activity | When | Details |
|---|---|---|
| **Code Review** | End of each week | Both review each other's PRs |
| **Cross-Testing** | End of each week | Each person writes 1-2 tests for the other's code |
| **Demo** | Friday | Both demo their week's work, explain design decisions |
| **Retro** | Friday | What went well, what's blocking, what to adjust |

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
| LLM API costs during development | Use Gemini Flash (free tier) for dev, switch to Pro for release |
| LLM API latency | Show typing indicator, stream responses via WebSocket |
| XGBoost/LightGBM install issues | Pin versions, test in CI across OS |
| Scope creep | New ideas go to `BACKLOG.md`, not into the sprint |
| One person blocked | Both know enough of each layer to help |
| Agent hallucinating bad advice | Rule-based layer validates before LLM generates text |
| API key security | Use env vars, never commit keys, add `.env` to `.gitignore` |

---

## Post-6-Weeks: Future Enhancements

After the 6-week build is complete:

- **v1.1** — Add MLP (scikit-learn) as a CPU neural net option
- **v1.2** — Agent memory — remembers user preferences across sessions
- **v2.0** — GPU models (PyTorch/TensorFlow: CNN, LSTM, Transformers)
- **v3.0** — Desktop packaging (.exe, .dmg, AppImage) — Ollama-style distribution
- **v4.0** — Multi-user support, cloud deployment option
