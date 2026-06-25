"""Model Registry — central dispatcher for running models.

The registry maintains a global dictionary mapping model names to their
classes. When you call run_model("rf_classifier", df, profile), it:
    1. Looks up "rf_classifier" in the registry
    2. Instantiates the model class
    3. Calls .run() to get a Result
    4. Auto-logs the run to SQLite
    5. Returns the Result

Models register themselves using the @register_model decorator. This
happens at import time — when you import the classifiers/regressors
packages, the decorators fire and models are added to the registry.

Usage:
    from xaura import run_model, list_models

    result = run_model("rf_classifier", df, profile)
    print(result.metrics)

    # List all available models
    for m in list_models():
        print(m["name"], m["task_type"])
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from xaura.models.base import BaseModel, Result
from xaura.profiler.dataprofile import DataProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(model_class: type[BaseModel]) -> type[BaseModel]:
    """Decorator to register a model class in the global registry.

    Reads the class-level `name` attribute and uses it as the key.
    Logs a warning if a duplicate name is registered (last one wins).

    Usage:
        @register_model
        class MyModel(BaseModel):
            name = "my_model"
            ...

    Args:
        model_class: A subclass of BaseModel with a `name` attribute.

    Returns:
        The model class (unchanged), so it can be used as a decorator.
    """
    name = model_class.name
    if not name:
        raise ValueError(f"{model_class.__name__} must have a non-empty 'name' attribute")

    if name in _MODEL_REGISTRY:
        logger.warning(
            "Model '%s' is already registered (overwriting with %s)",
            name,
            model_class.__name__,
        )

    _MODEL_REGISTRY[name] = model_class
    return model_class


# ---------------------------------------------------------------------------
# run_model — the main user-facing function
# ---------------------------------------------------------------------------


def run_model(
    model_name: str,
    data: pd.DataFrame,
    profile: DataProfile,
    config: dict[str, Any] | None = None,
    target_col: str | None = None,
    test_size: float = 0.2,
    db_path: str | None = None,
    auto_log: bool = True,
) -> Result:
    """Run a model by name and return a Result.

    This is the main entry point for running models. It:
        1. Looks up the model class by name
        2. Instantiates it
        3. Calls .run() to execute the full pipeline
        4. Auto-logs the run to SQLite (unless auto_log=False)
        5. Returns the Result

    Args:
        model_name: Name of the model (e.g. "rf_classifier", "lasso").
            Use list_models() to see all available names.
        data: Input DataFrame with features and target column.
        profile: DataProfile of the dataset (from xaura.profile()).
        config: Optional hyperparameter overrides. Merged with defaults.
        target_col: Name of the target column. Auto-detected if None.
        test_size: Fraction of data for test set (default 0.2).
        db_path: Path to SQLite database for logging. Uses default if None.
        auto_log: If True (default), auto-log the run to SQLite.

    Returns:
        A Result object with predictions, metrics, config, etc.

    Raises:
        KeyError: If model_name is not in the registry.
        ValueError: If the data is invalid for the model type.
    """
    # 1. Look up model class
    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Model '{model_name}' not found. Available models: {available}")

    model_class = _MODEL_REGISTRY[model_name]

    # 2. Instantiate
    model = model_class()

    # 3. Run the full pipeline
    result = model.run(
        df=data,
        profile=profile,
        config=config,
        target_col=target_col,
        test_size=test_size,
    )

    # 4. Auto-log to SQLite
    if auto_log:
        _auto_log(result, db_path)

    # 5. Return
    return result


# ---------------------------------------------------------------------------
# list_models — discover available models
# ---------------------------------------------------------------------------


def list_models(task_type: str | None = None) -> list[dict[str, str]]:
    """List all registered models.

    Returns a list of dicts, each containing the model's name,
    display_name, and task_type. Optionally filter by task_type.

    Args:
        task_type: Filter by task type ("classification", "regression",
            "clustering"). If None, returns all models.

    Returns:
        List of dicts with keys: name, display_name, task_type.

    Example:
        >>> list_models()
        [
            {"name": "logistic_regression", "display_name": "Logistic Regression", "task_type": "classification"},
            {"name": "rf_classifier", "display_name": "Random Forest Classifier", "task_type": "classification"},
            ...
        ]
    """
    models = []
    for name, model_class in sorted(_MODEL_REGISTRY.items()):
        info = {
            "name": model_class.name,
            "display_name": model_class.display_name,
            "task_type": model_class.task_type,
        }
        if task_type is None or model_class.task_type == task_type:
            models.append(info)
    return models


# ---------------------------------------------------------------------------
# Auto-logging helper
# ---------------------------------------------------------------------------


def _auto_log(result: Result, db_path: str | None = None) -> str | None:
    """Auto-log a model run to SQLite.

    Silently skips logging if the store is not available or if
    logging fails (we never want logging to crash a model run).

    Args:
        result: The Result from a model run.
        db_path: Path to SQLite database. Uses default if None.

    Returns:
        The run_id string if logged, None if skipped.
    """
    try:
        from xaura.store import create_run, init_db

        # Ensure DB exists
        init_db(db_path)

        # Build the run_data dict matching the store's expected format
        run_data = {
            "model_name": result.model_name,
            "task_type": result.task_type,
            "config": result.config,
            "metrics": result.metrics,
            "duration_seconds": result.train_time_seconds,
            "dataset_name": result.dataset_hash[:16] if result.dataset_hash else "",
        }

        run_id = create_run(run_data, db_path)
        logger.info("Run logged to SQLite: %s", run_id)
        return run_id

    except Exception as exc:
        # Never let logging failures crash the model run
        logger.warning("Auto-logging failed (non-fatal): %s", exc)
        return None
