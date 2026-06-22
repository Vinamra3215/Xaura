"""Tests for XAURA regressors (Linear, Ridge, Lasso)."""

import numpy as np
import pandas as pd
import pytest

from xaura import profile, run_model
from xaura.models import list_models, Result

# Ensure regressors are registered
import xaura.models.regressors  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    return pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "target": target,
    })


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
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert isinstance(result, Result)

    def test_regression_metrics(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        for metric in ["mse", "rmse", "mae", "r2"]:
            assert metric in result.metrics

    def test_r2_reasonable(self, reg_df, reg_profile):
        """R² should be decent since target is a linear combination."""
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        # Data is linear + small noise, so R² should be high
        assert result.metrics["r2"] > 0.5

    def test_mse_positive(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.metrics["mse"] >= 0.0
        assert result.metrics["rmse"] >= 0.0
        assert result.metrics["mae"] >= 0.0

    def test_rmse_is_sqrt_of_mse(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert abs(result.metrics["rmse"] - np.sqrt(result.metrics["mse"])) < 1e-6

    def test_model_name_and_type(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.model_name == "linear_regression"
        assert result.task_type == "regression"

    def test_feature_importances(self, reg_df, reg_profile):
        """Linear Regression has coef_ → feature importances."""
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4  # 4 features

    def test_train_test_splits(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.X_train is not None
        assert result.X_test is not None
        assert result.y_train is not None
        assert result.y_test is not None
        assert len(result.X_train) + len(result.X_test) == 300

    def test_no_probabilities(self, reg_df, reg_profile):
        """Regression models should NOT produce probabilities."""
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.probabilities is None

    def test_custom_config(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            config={"fit_intercept": False},
            target_col="target", auto_log=False,
        )
        assert result.config["fit_intercept"] is False

    def test_train_time_recorded(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, reg_df, reg_profile):
        result = run_model(
            "linear_regression", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------


class TestRidge:
    """Tests for RidgeModel."""

    def test_run_returns_result(self, reg_df, reg_profile):
        result = run_model(
            "ridge", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert isinstance(result, Result)

    def test_regression_metrics(self, reg_df, reg_profile):
        result = run_model(
            "ridge", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        for metric in ["mse", "rmse", "mae", "r2"]:
            assert metric in result.metrics

    def test_r2_reasonable(self, reg_df, reg_profile):
        result = run_model(
            "ridge", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.metrics["r2"] > 0.5

    def test_feature_importances(self, reg_df, reg_profile):
        """Ridge has coef_ → feature importances."""
        result = run_model(
            "ridge", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4

    def test_custom_config(self, reg_df, reg_profile):
        result = run_model(
            "ridge", reg_df, reg_profile,
            config={"alpha": 0.001},
            target_col="target", auto_log=False,
        )
        assert result.config["alpha"] == 0.001

    def test_model_object_saved(self, reg_df, reg_profile):
        result = run_model(
            "ridge", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "predict")

    def test_high_alpha_reduces_coefficients(self, reg_df, reg_profile):
        """With very high alpha, coefficients should be smaller."""
        result_low = run_model(
            "ridge", reg_df, reg_profile,
            config={"alpha": 0.001},
            target_col="target", auto_log=False,
        )
        result_high = run_model(
            "ridge", reg_df, reg_profile,
            config={"alpha": 1000.0},
            target_col="target", auto_log=False,
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
            "lasso", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert isinstance(result, Result)

    def test_regression_metrics(self, reg_df, reg_profile):
        result = run_model(
            "lasso", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        for metric in ["mse", "rmse", "mae", "r2"]:
            assert metric in result.metrics

    def test_r2_reasonable(self, reg_df, reg_profile):
        result = run_model(
            "lasso", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        # Lasso with default alpha may zero out some features,
        # but R² should still be positive on this dataset
        assert result.metrics["r2"] > 0.0

    def test_feature_importances(self, reg_df, reg_profile):
        """Lasso has coef_ → feature importances."""
        result = run_model(
            "lasso", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4

    def test_custom_config(self, reg_df, reg_profile):
        result = run_model(
            "lasso", reg_df, reg_profile,
            config={"alpha": 0.01, "max_iter": 10000},
            target_col="target", auto_log=False,
        )
        assert result.config["alpha"] == 0.01
        assert result.config["max_iter"] == 10000

    def test_lasso_can_zero_out_features(self, reg_df, reg_profile):
        """With very high alpha, Lasso should zero out some coefficients."""
        result = run_model(
            "lasso", reg_df, reg_profile,
            config={"alpha": 10.0, "max_iter": 5000},
            target_col="target", auto_log=False,
        )
        # At least one coefficient should be exactly zero
        n_zero = np.sum(result.model_object.coef_ == 0.0)
        assert n_zero >= 1, "Lasso with high alpha should zero out at least one feature"

    def test_model_object_saved(self, reg_df, reg_profile):
        result = run_model(
            "lasso", reg_df, reg_profile,
            target_col="target", auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "predict")


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegressorRegistry:
    """Tests for model registry with regressors."""

    def test_regressors_registered(self):
        models = list_models(task_type="regression")
        names = [m["name"] for m in models]
        assert "linear_regression" in names
        assert "ridge" in names
        assert "lasso" in names

    def test_list_models_has_display_name(self):
        models = list_models(task_type="regression")
        for m in models:
            assert "display_name" in m
            assert len(m["display_name"]) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not found"):
            run_model("nonexistent_regressor", pd.DataFrame(), None)

    def test_full_integration(self, reg_df):
        """End-to-end: profile → run_model → check result for all 3."""
        p = profile(reg_df)
        for model_name in ["linear_regression", "ridge", "lasso"]:
            result = run_model(
                model_name, reg_df, p,
                target_col="target", auto_log=False,
            )
            assert isinstance(result, Result)
            assert result.task_type == "regression"
            assert "r2" in result.metrics
            assert result.dataset_hash == p.dataset_hash
