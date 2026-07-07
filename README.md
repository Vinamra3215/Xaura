<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/tests-423%20passed-brightgreen?style=for-the-badge" />
</p>

# XAURA — eXtendable Automated Unified Research & Analytics

> An intelligent, dataset-aware Python ML library with built-in profiling, experiment tracking, model-specific visualisations, and a local web UI — all in one `pip install`.

XAURA is designed to make machine learning workflows **faster, smarter, and fully reproducible**. Instead of writing boilerplate for every project, you call `profile()` to understand your data and `run_model()` to train with dataset-aware defaults. Every run is automatically logged, visualised, and exportable.

---

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Python API](#python-api)
  - [Web UI](#web-ui)
  - [CLI](#cli)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Install XAURA directly from PyPI:

```bash
pip install xaura
```

### Install from Source

If you want to contribute or run the latest development version:

```bash
git clone https://github.com/Vinamra3215/Xaura.git
cd Xaura
pip install -e ".[dev]"
```

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
result = run_model("rf_classifier", df, data_profile, target_col="target")

# Step 3: Inspect results
print(result.metrics)       # {'accuracy': 0.91, 'f1': 0.85, 'recall': 0.78, ...}
print(result.config)        # Full config with all defaults resolved
print(result.run_id)        # 'a3f8c21d-...' — logged to SQLite automatically

# Step 4: Override defaults if needed
result2 = run_model("xgboost_cls", df, data_profile, target_col="target", config={
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.01
})
```

### Web UI

XAURA comes with a beautiful local web interface. Start it by running:

```bash
xaura serve
```
Then open `http://localhost:7070` in your browser.

- **Upload page** — drag-and-drop CSV upload
- **Profile view** — dataset summary with interactive charts
- **Model runner** — select model, configure params, run, view results
- **Experiment log** — sortable/filterable table of all past runs
- **Run comparison** — side-by-side diff of two runs
- **Export buttons** — one-click download of plots, bundles, logs

### CLI

You can also use XAURA directly from the terminal:

```bash
xaura profile data.csv
xaura run rf_classifier data.csv --config '{"n_estimators": 200}'
xaura export a3f8c21d
```

---

## Key Features

- 🔍 **Automatic Dataset Profiling** — shape, types, class balance, correlations, missing values, and warnings
- 🎯 **Dataset-Aware Defaults** — hyperparameters adapt based on your data's characteristics
- 📊 **Model-Specific Visualisations** — confusion matrices for classifiers, residual plots for regressors, silhouette plots for clusterers
- 💾 **Silent Experiment Tracking** — every run auto-logged to SQLite with full reproducibility
- 🌐 **Local Web UI** — FastAPI-powered dashboard served at `localhost:7070`
- 📦 **One-Click Export** — PNG plots, JSON configs, ZIP run bundles, CSV experiment logs
- 🖥️ **CLI Interface** — profile, run models, and export from the terminal
- ⚡ **CPU-Optimised** — all models run efficiently on CPU (no GPU required)

---

## Supported Models

#### Classification
| Model | Identifier |
|---|---|
| Logistic Regression | `logistic_regression` |
| Random Forest | `rf_classifier` |
| XGBoost | `xgboost_cls` |
| LightGBM | `lightgbm_cls` |

#### Regression
| Model | Identifier |
|---|---|
| Linear Regression | `linear_regression` |
| Ridge | `ridge` |
| Lasso | `lasso` |
| Random Forest | `random_forest_reg` |
| XGBoost | `xgboost_reg` |

#### Clustering
| Model | Identifier |
|---|---|
| K-Means | `kmeans` |
| DBSCAN | `dbscan` |
| Agglomerative | `hierarchical` |

---

## API Reference

### `profile(data) → DataProfile`

| Parameter | Type | Description |
|---|---|---|
| `data` | `pd.DataFrame`, `str`, `np.ndarray` | Dataset or path to CSV |

### `run_model(model_name, data, profile, config=None, target_col=None) → Result`

| Parameter | Type | Description |
|---|---|---|
| `model_name` | `str` | Model identifier (e.g., `"rf_classifier"`) |
| `data` | `pd.DataFrame` | Dataset |
| `profile` | `DataProfile` | From `profile()` call |
| `config` | `dict`, optional | Hyperparameter overrides |
| `target_col` | `str`, optional | The target column to predict |

### `Result` Object

| Attribute | Type | Description |
|---|---|---|
| `metrics` | `dict` | Evaluation metrics |
| `plots` | `list` | Plotly JSON chart objects |
| `model_object` | `object` | Trained model |
| `run_id` | `str` | UUID in experiment log |
| `config` | `dict` | Full resolved config |

---

## Architecture

XAURA is built with strict layer separation. Each component has a clear responsibility:

1. **User Interfaces:** Web UI (FastAPI/JS), CLI (Click), Python API
2. **Core Library:** Profiler, Model Wrappers, Registry, Visualisation Engines
3. **Storage:** Local SQLite database for zero-config experiment tracking

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

<p align="center">
  <strong>XAURA</strong> — Because ML should be intelligent about your data, not just your model.
</p>
