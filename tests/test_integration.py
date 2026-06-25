"""Integration tests — end-to-end pipeline verification.

Tests the full XAURA workflow:
    profile(df) → get_defaults(profile, model) → run_model() → Result → SQLite store

These tests verify that all components work together correctly:
- Profiler produces DataProfiles that defaults engine can read
- Defaults engine produces configs that models accept
- Models produce Results that the SQLite store can log
- The registry dispatches correctly for all 4 Person B models
- User config overrides flow through the entire pipeline
"""

import numpy as np
import pandas as pd
import pytest

from xaura.models.base import Result

# Import models to trigger @register_model
from xaura.models.classifiers.lightgbm_cls import LightGBMClassifier  # noqa: F401
from xaura.models.classifiers.xgboost_cls import XGBoostClassifier  # noqa: F401
from xaura.models.defaults import get_defaults
from xaura.models.registry import list_models, run_model
from xaura.models.regressors.random_forest_reg import RandomForestRegressor  # noqa: F401
from xaura.models.regressors.xgboost_reg import XGBoostRegressor  # noqa: F401
from xaura.profiler.profiler import profile
from xaura.store import create_run, get_run, init_db, list_runs

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def classification_df():
    """Binary classification dataset."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    return pd.DataFrame(
        {
            "feature_a": x1,
            "feature_b": x2,
            "feature_c": np.random.uniform(0, 10, n),
            "target": (x1 + x2 > 0).astype(int),
        }
    )


@pytest.fixture
def regression_df():
    """Regression dataset with clear linear signal."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    return pd.DataFrame(
        {
            "feature_a": x1,
            "feature_b": x2,
            "feature_c": np.random.uniform(0, 100, n),
            "target": 3 * x1 + 2 * x2 + np.random.randn(n) * 0.5,
        }
    )


@pytest.fixture
def mixed_df():
    """Dataset with both numeric and categorical features."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.uniform(20000, 200000, n),
            "region": np.random.choice(["north", "south", "east", "west"], n),
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n),
        }
    )


@pytest.fixture
def db_path(tmp_path):
    """Temporary SQLite database path."""
    path = tmp_path / "test_integration.db"
    init_db(str(path))
    return str(path)


# ─────────────────────────────────────────────────────────────
# Full pipeline tests: profile → defaults → run → Result
# ─────────────────────────────────────────────────────────────


class TestFullPipelineClassification:
    """End-to-end tests for classification models."""

    @pytest.mark.parametrize("model_name", ["xgboost_cls", "lightgbm_cls"])
    def test_profile_to_result(self, classification_df, model_name):
        """profile() → run_model() → Result works end-to-end."""
        dp = profile(classification_df)
        result = run_model(model_name, classification_df, dp, auto_log=False)

        # Result is valid
        assert isinstance(result, Result)
        assert result.model_name == model_name
        assert result.task_type == "classification"

        # Metrics are populated
        assert result.metrics["accuracy"] > 0
        assert result.metrics["f1"] > 0

        # Predictions match test set size
        assert len(result.predictions) == len(result.y_test)

        # Config was generated from defaults
        assert len(result.config) > 0

    @pytest.mark.parametrize("model_name", ["xgboost_cls", "lightgbm_cls"])
    def test_defaults_match_config(self, classification_df, model_name):
        """get_defaults() output should be a subset of the final config."""
        dp = profile(classification_df)
        defaults = get_defaults(dp, model_name)
        result = run_model(model_name, classification_df, dp, auto_log=False)

        # Every default key should be in the final config
        for key in defaults:
            assert key in result.config, f"Default key '{key}' missing from result config"


class TestFullPipelineRegression:
    """End-to-end tests for regression models."""

    @pytest.mark.parametrize("model_name", ["random_forest_reg", "xgboost_reg"])
    def test_profile_to_result(self, regression_df, model_name):
        """profile() → run_model() → Result works end-to-end."""
        dp = profile(regression_df)
        result = run_model(model_name, regression_df, dp, auto_log=False)

        assert isinstance(result, Result)
        assert result.model_name == model_name
        assert result.task_type == "regression"

        # Regression metrics
        assert "mse" in result.metrics
        assert "rmse" in result.metrics
        assert "mae" in result.metrics
        assert "r2" in result.metrics

        # R² should be decent for this linear dataset
        assert result.metrics["r2"] > 0.5

    @pytest.mark.parametrize("model_name", ["random_forest_reg", "xgboost_reg"])
    def test_no_probabilities_for_regression(self, regression_df, model_name):
        """Regressors should never produce probabilities."""
        dp = profile(regression_df)
        result = run_model(model_name, regression_df, dp, auto_log=False)
        assert result.probabilities is None


# ─────────────────────────────────────────────────────────────
# Store integration: run → log → retrieve
# ─────────────────────────────────────────────────────────────


class TestStoreIntegration:
    """Tests that model results can be logged to and retrieved from SQLite."""

    def test_log_classification_result(self, classification_df, db_path):
        """Run a classifier, log to store, retrieve and verify."""
        dp = profile(classification_df)
        result = run_model("xgboost_cls", classification_df, dp, auto_log=False)

        # Manually log to store
        run_data = {
            "model_name": result.model_name,
            "task_type": result.task_type,
            "config": result.config,
            "metrics": result.metrics,
            "duration_seconds": result.train_time_seconds,
            "dataset_name": "test_classification",
        }
        run_id = create_run(run_data, db_path)

        # Retrieve and verify
        stored = get_run(run_id, db_path)
        assert stored is not None
        assert stored["model_name"] == "xgboost_cls"
        assert stored["task_type"] == "classification"
        assert stored["metrics"]["accuracy"] == result.metrics["accuracy"]
        assert stored["config"]["n_estimators"] == result.config["n_estimators"]

    def test_log_regression_result(self, regression_df, db_path):
        """Run a regressor, log to store, retrieve and verify."""
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)

        run_data = {
            "model_name": result.model_name,
            "task_type": result.task_type,
            "config": result.config,
            "metrics": result.metrics,
            "duration_seconds": result.train_time_seconds,
            "dataset_name": "test_regression",
        }
        run_id = create_run(run_data, db_path)

        stored = get_run(run_id, db_path)
        assert stored is not None
        assert stored["model_name"] == "random_forest_reg"
        assert stored["metrics"]["r2"] == result.metrics["r2"]

    def test_log_multiple_models_and_list(self, classification_df, db_path):
        """Run multiple models, log all, list and filter."""
        dp = profile(classification_df)

        logged_ids = []
        for model_name in ["xgboost_cls", "lightgbm_cls"]:
            result = run_model(model_name, classification_df, dp, auto_log=False)
            run_data = {
                "model_name": result.model_name,
                "task_type": result.task_type,
                "config": result.config,
                "metrics": result.metrics,
                "duration_seconds": result.train_time_seconds,
            }
            run_id = create_run(run_data, db_path)
            logged_ids.append(run_id)

        # List all runs
        all_runs = list_runs(db_path=db_path)
        assert len(all_runs) >= 2

        # Filter by model name
        xgb_runs = list_runs(filters={"model_name": "xgboost_cls"}, db_path=db_path)
        assert len(xgb_runs) >= 1
        assert all(r["model_name"] == "xgboost_cls" for r in xgb_runs)

    def test_metrics_survive_json_roundtrip(self, classification_df, db_path):
        """Metrics should survive serialisation to/from SQLite JSON."""
        dp = profile(classification_df)
        result = run_model("lightgbm_cls", classification_df, dp, auto_log=False)

        run_data = {
            "model_name": result.model_name,
            "task_type": result.task_type,
            "config": result.config,
            "metrics": result.metrics,
        }
        run_id = create_run(run_data, db_path)
        stored = get_run(run_id, db_path)

        # Every metric should roundtrip exactly
        for key, value in result.metrics.items():
            assert stored["metrics"][key] == pytest.approx(
                value
            ), f"Metric '{key}' changed after roundtrip"


# ─────────────────────────────────────────────────────────────
# Mixed data / edge case integration
# ─────────────────────────────────────────────────────────────


class TestMixedDataIntegration:
    """Tests with datasets containing categorical + numeric features."""

    @pytest.mark.parametrize("model_name", ["xgboost_cls", "lightgbm_cls"])
    def test_categorical_encoding_pipeline(self, mixed_df, model_name):
        """Categorical features should be encoded transparently."""
        dp = profile(mixed_df)
        result = run_model(model_name, mixed_df, dp, auto_log=False)

        assert isinstance(result, Result)
        assert result.metrics["accuracy"] > 0

    def test_all_four_models_on_same_data(self, classification_df, regression_df):
        """Run all 4 Person B models and verify they all return Results."""
        cls_dp = profile(classification_df)
        reg_dp = profile(regression_df)

        results = {}

        # Classifiers
        for name in ["xgboost_cls", "lightgbm_cls"]:
            results[name] = run_model(name, classification_df, cls_dp, auto_log=False)

        # Regressors
        for name in ["random_forest_reg", "xgboost_reg"]:
            results[name] = run_model(name, regression_df, reg_dp, auto_log=False)

        # All should be valid Results
        for name, result in results.items():
            assert isinstance(result, Result), f"{name} didn't return a Result"
            assert result.model_name == name
            assert result.train_time_seconds > 0
            assert len(result.metrics) > 0


# ─────────────────────────────────────────────────────────────
# Config override integration
# ─────────────────────────────────────────────────────────────


class TestConfigOverrideIntegration:
    """Tests that user config overrides flow through the whole pipeline."""

    def test_override_merges_with_defaults(self, classification_df):
        """User overrides should merge with (not replace) defaults."""
        dp = profile(classification_df)

        # Run with one override
        result = run_model(
            "xgboost_cls",
            classification_df,
            dp,
            config={"n_estimators": 5},
            auto_log=False,
        )

        # Override applied
        assert result.config["n_estimators"] == 5

        # Other defaults still present (learning_rate comes from defaults)
        assert "learning_rate" in result.config

    def test_override_persists_to_store(self, classification_df, db_path):
        """Overridden config should be stored correctly in SQLite."""
        dp = profile(classification_df)
        result = run_model(
            "lightgbm_cls",
            classification_df,
            dp,
            config={"n_estimators": 7, "num_leaves": 4},
            auto_log=False,
        )

        run_data = {
            "model_name": result.model_name,
            "task_type": result.task_type,
            "config": result.config,
            "metrics": result.metrics,
        }
        run_id = create_run(run_data, db_path)
        stored = get_run(run_id, db_path)

        assert stored["config"]["n_estimators"] == 7
        assert stored["config"]["num_leaves"] == 4


# ─────────────────────────────────────────────────────────────
# Registry completeness
# ─────────────────────────────────────────────────────────────


class TestRegistryCompleteness:
    """Tests that the registry is complete and consistent."""

    def test_all_person_b_models_registered(self):
        """All 4 Person B models should be in the registry."""
        models = list_models()
        names = {m["name"] for m in models}
        expected = {"xgboost_cls", "lightgbm_cls", "random_forest_reg", "xgboost_reg"}
        assert expected.issubset(names)

    def test_classification_filter(self):
        cls_models = list_models(task_type="classification")
        names = {m["name"] for m in cls_models}
        assert "xgboost_cls" in names
        assert "lightgbm_cls" in names
        assert "random_forest_reg" not in names
        assert "xgboost_reg" not in names

    def test_regression_filter(self):
        reg_models = list_models(task_type="regression")
        names = {m["name"] for m in reg_models}
        assert "random_forest_reg" in names
        assert "xgboost_reg" in names
        assert "xgboost_cls" not in names

    def test_unknown_model_raises(self, classification_df):
        dp = profile(classification_df)
        with pytest.raises(KeyError, match="not found"):
            run_model("fake_model", classification_df, dp, auto_log=False)

    def test_each_model_has_display_name(self):
        for m in list_models():
            assert m["display_name"] != "", f"{m['name']} has empty display_name"
