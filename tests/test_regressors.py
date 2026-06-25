"""Tests for Person B's regressors — Random Forest and XGBoost.

Tests the full pipeline via run_model(), verifying that both
regressors:
- Register correctly in the registry
- Return valid Result objects with regression metrics
- Handle user config overrides
- Work with various data shapes
- Handle categorical features via BaseModel's encoding
"""

import numpy as np
import pandas as pd
import pytest

from xaura.models.base import Result
from xaura.models.registry import list_models, run_model

# Import to trigger @register_model decorators
from xaura.models.regressors.random_forest_reg import RandomForestRegressor  # noqa: F401
from xaura.models.regressors.xgboost_reg import XGBoostRegressor  # noqa: F401
from xaura.profiler.profiler import profile

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def regression_df():
    """Regression dataset (300 rows, continuous target)."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": np.random.uniform(0, 100, n),
            "target": 3 * x1 + 2 * x2 + np.random.randn(n) * 0.5,
        }
    )


@pytest.fixture
def regression_with_cat_df():
    """Regression dataset with a categorical feature."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    return pd.DataFrame(
        {
            "x1": x1,
            "region": np.random.choice(["north", "south", "east", "west"], n),
            "target": 2 * x1 + np.random.randn(n) * 0.3,
        }
    )


@pytest.fixture
def small_regression_df():
    """Small regression dataset (100 rows)."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": np.random.randn(n),
            "target": 5 * x1 + np.random.randn(n),
        }
    )


# ─────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────


class TestRegressorRegistry:
    """Tests that regressors register correctly."""

    def test_rf_regressor_registered(self):
        models = list_models()
        names = [m["name"] for m in models]
        assert "random_forest_reg" in names

    def test_xgb_regressor_registered(self):
        models = list_models()
        names = [m["name"] for m in models]
        assert "xgboost_reg" in names

    def test_both_are_regression(self):
        reg_models = list_models(task_type="regression")
        names = [m["name"] for m in reg_models]
        assert "random_forest_reg" in names
        assert "xgboost_reg" in names

    def test_display_names(self):
        models = {m["name"]: m for m in list_models()}
        assert models["random_forest_reg"]["display_name"] == "Random Forest Regressor"
        assert models["xgboost_reg"]["display_name"] == "XGBoost Regressor"


# ─────────────────────────────────────────────────────────────
# Random Forest Regressor tests
# ─────────────────────────────────────────────────────────────


class TestRandomForestRegressor:
    """Tests for Random Forest Regressor via run_model()."""

    def test_basic_regression(self, regression_df):
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        assert isinstance(result, Result)
        assert result.model_name == "random_forest_reg"
        assert result.task_type == "regression"

    def test_returns_regression_metrics(self, regression_df):
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        for key in ["mse", "rmse", "mae", "r2"]:
            assert key in result.metrics, f"Missing metric: {key}"

    def test_mse_rmse_relationship(self, regression_df):
        """RMSE should be sqrt(MSE)."""
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        assert abs(result.metrics["rmse"] - result.metrics["mse"] ** 0.5) < 1e-6

    def test_r2_reasonable(self, regression_df):
        """R² should be reasonable for a linear-ish dataset."""
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        # Our test data has a strong linear signal, R² should be decent
        assert result.metrics["r2"] > 0.5

    def test_predictions_shape(self, regression_df):
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        assert len(result.predictions) == len(result.y_test)

    def test_no_probabilities(self, regression_df):
        """Regressors should NOT have probabilities."""
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        assert result.probabilities is None

    def test_feature_importances(self, regression_df):
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        assert result.feature_importances is not None

    def test_config_overrides(self, regression_df):
        dp = profile(regression_df)
        result = run_model(
            "random_forest_reg",
            regression_df,
            dp,
            config={"n_estimators": 20, "max_depth": 3},
            auto_log=False,
        )
        assert result.config["n_estimators"] == 20
        assert result.config["max_depth"] == 3

    def test_categorical_features(self, regression_with_cat_df):
        dp = profile(regression_with_cat_df)
        result = run_model("random_forest_reg", regression_with_cat_df, dp, auto_log=False)
        assert isinstance(result, Result)

    def test_train_time_recorded(self, regression_df):
        dp = profile(regression_df)
        result = run_model("random_forest_reg", regression_df, dp, auto_log=False)
        assert result.train_time_seconds > 0

    def test_small_dataset(self, small_regression_df):
        """Should work on small datasets without errors."""
        dp = profile(small_regression_df)
        result = run_model("random_forest_reg", small_regression_df, dp, auto_log=False)
        assert isinstance(result, Result)


# ─────────────────────────────────────────────────────────────
# XGBoost Regressor tests
# ─────────────────────────────────────────────────────────────


class TestXGBoostRegressor:
    """Tests for XGBoost Regressor via run_model()."""

    def test_basic_regression(self, regression_df):
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        assert isinstance(result, Result)
        assert result.model_name == "xgboost_reg"
        assert result.task_type == "regression"

    def test_returns_regression_metrics(self, regression_df):
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        for key in ["mse", "rmse", "mae", "r2"]:
            assert key in result.metrics

    def test_r2_reasonable(self, regression_df):
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        assert result.metrics["r2"] > 0.5

    def test_predictions_continuous(self, regression_df):
        """Predictions should be continuous (not integer labels)."""
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        # At least some predictions should be non-integer
        non_integer = sum(1 for p in result.predictions if p != int(p))
        assert non_integer > 0

    def test_feature_importances(self, regression_df):
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        assert result.feature_importances is not None

    def test_config_overrides(self, regression_df):
        dp = profile(regression_df)
        result = run_model(
            "xgboost_reg",
            regression_df,
            dp,
            config={"n_estimators": 10, "max_depth": 2},
            auto_log=False,
        )
        assert result.config["n_estimators"] == 10
        assert result.config["max_depth"] == 2

    def test_categorical_features(self, regression_with_cat_df):
        dp = profile(regression_with_cat_df)
        result = run_model("xgboost_reg", regression_with_cat_df, dp, auto_log=False)
        assert isinstance(result, Result)

    def test_dataset_hash_recorded(self, regression_df):
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        assert result.dataset_hash != ""
        assert result.dataset_hash == dp.dataset_hash

    def test_train_test_splits_stored(self, regression_df):
        dp = profile(regression_df)
        result = run_model("xgboost_reg", regression_df, dp, auto_log=False)
        assert result.X_train is not None
        assert result.X_test is not None
        assert len(result.X_train) + len(result.X_test) == len(regression_df)
