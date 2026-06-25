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
from xaura.models.registry import run_model

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


@pytest.fixture
def reg_df():
    """Regression dataset (300 rows, 4 features + continuous target)."""
    np.random.seed(42)
    n = 300
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.random.uniform(0, 10, n)
    f4 = np.random.randn(n)
    # Target is a linear combination + noise
    target = 3.0 * f1 - 2.0 * f2 + 0.5 * f3 + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
            "target": target,
        }
    )


@pytest.fixture
def reg_profile(reg_df):
    """DataProfile for the regression dataset."""
    return profile(reg_df)


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------


class TestLinearRegression:
    """Tests for LinearRegressionModel."""

    def test_run_returns_result(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_regression_metrics(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        for metric in ["mse", "rmse", "mae", "r2"]:
            assert metric in result.metrics

    def test_r2_reasonable(self, reg_df, reg_profile):
        """R² should be decent since target is a linear combination."""
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        # Data is linear + small noise, so R² should be high
        assert result.metrics["r2"] > 0.5

    def test_mse_positive(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.metrics["mse"] >= 0.0
        assert result.metrics["rmse"] >= 0.0
        assert result.metrics["mae"] >= 0.0

    def test_rmse_is_sqrt_of_mse(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert abs(result.metrics["rmse"] - np.sqrt(result.metrics["mse"])) < 1e-6

    def test_model_name_and_type(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.model_name == "linear_regression"
        assert result.task_type == "regression"

    def test_feature_importances(self, reg_df, reg_profile):
        """Linear Regression has coef_ → feature importances."""
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4  # 4 features

    def test_train_test_splits(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.X_train is not None
        assert result.X_test is not None
        assert result.y_train is not None
        assert result.y_test is not None
        assert len(result.X_train) + len(result.X_test) == 300

    def test_no_probabilities(self, reg_df, reg_profile):
        """Regression models should NOT produce probabilities."""
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.probabilities is None

    def test_custom_config(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            config={"fit_intercept": False},
            target_col="target",
            auto_log=False,
        )
        assert result.config["fit_intercept"] is False

    def test_train_time_recorded(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------


class TestRidge:
    """Tests for RidgeModel."""

    def test_run_returns_result(self, reg_df, reg_profile):
        result = run_model(
            "ridge",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_regression_metrics(self, reg_df, reg_profile):
        result = run_model(
            "ridge",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        for metric in ["mse", "rmse", "mae", "r2"]:
            assert metric in result.metrics

    def test_r2_reasonable(self, reg_df, reg_profile):
        result = run_model(
            "ridge",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.metrics["r2"] > 0.5

    def test_feature_importances(self, reg_df, reg_profile):
        """Ridge has coef_ → feature importances."""
        result = run_model(
            "ridge",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4

    def test_custom_config(self, reg_df, reg_profile):
        result = run_model(
            "ridge",
            reg_df,
            reg_profile,
            config={"alpha": 0.001},
            target_col="target",
            auto_log=False,
        )
        assert result.config["alpha"] == 0.001

    def test_model_object_saved(self, reg_df, reg_profile):
        result = run_model(
            "ridge",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "predict")

    def test_high_alpha_reduces_coefficients(self, reg_df, reg_profile):
        """With very high alpha, coefficients should be smaller."""
        result_low = run_model(
            "ridge",
            reg_df,
            reg_profile,
            config={"alpha": 0.001},
            target_col="target",
            auto_log=False,
        )
        result_high = run_model(
            "ridge",
            reg_df,
            reg_profile,
            config={"alpha": 1000.0},
            target_col="target",
            auto_log=False,
        )
        # High alpha → smaller coefficients (more shrinkage)
        low_coef_sum = np.sum(np.abs(result_low.model_object.coef_))
        high_coef_sum = np.sum(np.abs(result_high.model_object.coef_))
        assert high_coef_sum < low_coef_sum


# ---------------------------------------------------------------------------
# Lasso Regression
# ---------------------------------------------------------------------------


class TestLasso:
    """Tests for LassoModel."""

    def test_run_returns_result(self, reg_df, reg_profile):
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_regression_metrics(self, reg_df, reg_profile):
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        for metric in ["mse", "rmse", "mae", "r2"]:
            assert metric in result.metrics

    def test_r2_reasonable(self, reg_df, reg_profile):
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        # Lasso with default alpha may zero out some features,
        # but R² should still be positive on this dataset
        assert result.metrics["r2"] > 0.0

    def test_feature_importances(self, reg_df, reg_profile):
        """Lasso has coef_ → feature importances."""
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4

    def test_custom_config(self, reg_df, reg_profile):
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            config={"alpha": 0.01, "max_iter": 10000},
            target_col="target",
            auto_log=False,
        )
        assert result.config["alpha"] == 0.01
        assert result.config["max_iter"] == 10000

    def test_lasso_can_zero_out_features(self, reg_df, reg_profile):
        """With very high alpha, Lasso should zero out some coefficients."""
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            config={"alpha": 10.0, "max_iter": 5000},
            target_col="target",
            auto_log=False,
        )
        # At least one coefficient should be exactly zero
        n_zero = np.sum(result.model_object.coef_ == 0.0)
        assert n_zero >= 1, "Lasso with high alpha should zero out at least one feature"

    def test_model_object_saved(self, reg_df, reg_profile):
        result = run_model(
            "lasso",
            reg_df,
            reg_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "predict")


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
