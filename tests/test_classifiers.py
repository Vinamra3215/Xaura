"""Tests for Person B's classifiers — XGBoost and LightGBM.

Tests the full pipeline via run_model() and direct .run(),
verifying that both classifiers:
- Register correctly in the registry
- Return valid Result objects with expected metrics
- Handle user config overrides
- Work with binary and multiclass data
- Handle categorical features via BaseModel's encoding
"""

import numpy as np
import pandas as pd
import pytest

from xaura.models.base import Result
from xaura.models.classifiers.lightgbm_cls import LightGBMClassifier  # noqa: F401

# Import to trigger @register_model decorators
from xaura.models.classifiers.xgboost_cls import XGBoostClassifier  # noqa: F401
from xaura.models.registry import list_models, run_model
from xaura.profiler.profiler import profile

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def binary_df():
    """Binary classification dataset (200 rows)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.uniform(0, 10, n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def multiclass_df():
    """Multiclass classification dataset (300 rows, 3 classes)."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "target": np.random.choice([0, 1, 2], n),
        }
    )


@pytest.fixture
def categorical_df():
    """Classification dataset with a categorical feature."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "num1": np.random.randn(n),
            "cat1": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice([0, 1], n),
        }
    )


# ─────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────


class TestClassifierRegistry:
    """Tests that classifiers register correctly."""

    def test_xgboost_registered(self):
        models = list_models()
        names = [m["name"] for m in models]
        assert "xgboost_cls" in names

    def test_lightgbm_registered(self):
        models = list_models()
        names = [m["name"] for m in models]
        assert "lightgbm_cls" in names

    def test_both_are_classification(self):
        cls_models = list_models(task_type="classification")
        names = [m["name"] for m in cls_models]
        assert "xgboost_cls" in names
        assert "lightgbm_cls" in names

    def test_display_names(self):
        models = {m["name"]: m for m in list_models()}
        assert models["xgboost_cls"]["display_name"] == "XGBoost Classifier"
        assert models["lightgbm_cls"]["display_name"] == "LightGBM Classifier"


# ─────────────────────────────────────────────────────────────
# XGBoost Classifier tests
# ─────────────────────────────────────────────────────────────


class TestXGBoostClassifier:
    """Tests for XGBoost Classifier via run_model()."""

    def test_binary_classification(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert isinstance(result, Result)
        assert result.model_name == "xgboost_cls"
        assert result.task_type == "classification"

    def test_returns_expected_metrics(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in result.metrics, f"Missing metric: {key}"
            assert 0.0 <= result.metrics[key] <= 1.0

    def test_roc_auc_for_binary(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert "roc_auc" in result.metrics

    def test_predictions_shape(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert result.predictions is not None
        assert len(result.predictions) == len(result.y_test)

    def test_probabilities_present(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert result.probabilities is not None
        assert result.probabilities.shape[1] == 2  # binary

    def test_feature_importances(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert result.feature_importances is not None

    def test_train_time_recorded(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert result.train_time_seconds > 0

    def test_config_overrides(self, binary_df):
        dp = profile(binary_df)
        result = run_model(
            "xgboost_cls",
            binary_df,
            dp,
            config={"n_estimators": 10, "max_depth": 2},
            auto_log=False,
        )
        assert result.config["n_estimators"] == 10
        assert result.config["max_depth"] == 2

    def test_multiclass(self, multiclass_df):
        dp = profile(multiclass_df)
        result = run_model("xgboost_cls", multiclass_df, dp, auto_log=False)
        assert result.metrics["accuracy"] >= 0.0
        assert result.probabilities.shape[1] == 3  # 3 classes

    def test_categorical_features(self, categorical_df):
        """Categorical features should be encoded by BaseModel."""
        dp = profile(categorical_df)
        result = run_model("xgboost_cls", categorical_df, dp, auto_log=False)
        assert isinstance(result, Result)

    def test_train_test_splits_stored(self, binary_df):
        dp = profile(binary_df)
        result = run_model("xgboost_cls", binary_df, dp, auto_log=False)
        assert result.X_train is not None
        assert result.X_test is not None
        assert result.y_train is not None
        assert result.y_test is not None


# ─────────────────────────────────────────────────────────────
# LightGBM Classifier tests
# ─────────────────────────────────────────────────────────────


class TestLightGBMClassifier:
    """Tests for LightGBM Classifier via run_model()."""

    def test_binary_classification(self, binary_df):
        dp = profile(binary_df)
        result = run_model("lightgbm_cls", binary_df, dp, auto_log=False)
        assert isinstance(result, Result)
        assert result.model_name == "lightgbm_cls"
        assert result.task_type == "classification"

    def test_returns_expected_metrics(self, binary_df):
        dp = profile(binary_df)
        result = run_model("lightgbm_cls", binary_df, dp, auto_log=False)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in result.metrics
            assert 0.0 <= result.metrics[key] <= 1.0

    def test_roc_auc_for_binary(self, binary_df):
        dp = profile(binary_df)
        result = run_model("lightgbm_cls", binary_df, dp, auto_log=False)
        assert "roc_auc" in result.metrics

    def test_predictions_shape(self, binary_df):
        dp = profile(binary_df)
        result = run_model("lightgbm_cls", binary_df, dp, auto_log=False)
        assert len(result.predictions) == len(result.y_test)

    def test_feature_importances(self, binary_df):
        dp = profile(binary_df)
        result = run_model("lightgbm_cls", binary_df, dp, auto_log=False)
        assert result.feature_importances is not None

    def test_config_overrides(self, binary_df):
        dp = profile(binary_df)
        result = run_model(
            "lightgbm_cls",
            binary_df,
            dp,
            config={"n_estimators": 10, "num_leaves": 8},
            auto_log=False,
        )
        assert result.config["n_estimators"] == 10
        assert result.config["num_leaves"] == 8

    def test_multiclass(self, multiclass_df):
        dp = profile(multiclass_df)
        result = run_model("lightgbm_cls", multiclass_df, dp, auto_log=False)
        assert result.metrics["accuracy"] >= 0.0

    def test_categorical_features(self, categorical_df):
        dp = profile(categorical_df)
        result = run_model("lightgbm_cls", categorical_df, dp, auto_log=False)
        assert isinstance(result, Result)

    def test_dataset_hash_recorded(self, binary_df):
        dp = profile(binary_df)
        result = run_model("lightgbm_cls", binary_df, dp, auto_log=False)
        assert result.dataset_hash != ""
        assert result.dataset_hash == dp.dataset_hash


@pytest.fixture
def clf_df():
    """Binary classification dataset (200 rows, 4 features + target)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.uniform(0, 10, n),
            "f4": np.random.randn(n),
            "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def clf_profile(clf_df):
    """DataProfile for the classification dataset."""
    return profile(clf_df)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------


class TestLogisticRegression:
    """Tests for LogisticRegressionModel."""

    def test_run_returns_result(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_result_has_predictions(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert len(result.predictions) > 0
        assert result.predictions.dtype in [np.int64, np.int32, np.float64]

    def test_result_has_probabilities(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.probabilities is not None
        assert result.probabilities.shape[1] == 2  # binary

    def test_classification_metrics(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert metric in result.metrics
            assert 0.0 <= result.metrics[metric] <= 1.0

    def test_roc_auc_present(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert "roc_auc" in result.metrics

    def test_model_name_and_type(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.model_name == "logistic_regression"
        assert result.task_type == "classification"

    def test_feature_importances(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        # Logistic regression has coef_ → feature importances
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4  # 4 features

    def test_train_test_splits(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.X_train is not None
        assert result.X_test is not None
        assert result.y_train is not None
        assert result.y_test is not None
        assert len(result.X_train) + len(result.X_test) == 200

    def test_custom_config(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            config={"C": 0.01, "max_iter": 500},
            target_col="target",
            auto_log=False,
        )
        assert result.config["C"] == 0.01
        assert result.config["max_iter"] == 500

    def test_multiclass(self, multiclass_df):
        p = profile(multiclass_df)
        result = run_model(
            "logistic_regression",
            multiclass_df,
            p,
            target_col="target",
            auto_log=False,
        )
        assert result.probabilities.shape[1] == 3  # 3 classes

    def test_train_time_recorded(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------


class TestRandomForestClassifier:
    """Tests for RandomForestClassifierModel."""

    def test_run_returns_result(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_classification_metrics(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert metric in result.metrics
            assert 0.0 <= result.metrics[metric] <= 1.0

    def test_feature_importances(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        # Random Forest always has feature_importances_
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4

    def test_probabilities(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.probabilities is not None

    def test_custom_config(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier",
            clf_df,
            clf_profile,
            config={"n_estimators": 10, "max_depth": 3},
            target_col="target",
            auto_log=False,
        )
        assert result.config["n_estimators"] == 10
        assert result.config["max_depth"] == 3

    def test_model_object_saved(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier",
            clf_df,
            clf_profile,
            target_col="target",
            auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "predict")

    def test_multiclass(self, multiclass_df):
        p = profile(multiclass_df)
        result = run_model(
            "rf_classifier",
            multiclass_df,
            p,
            target_col="target",
            auto_log=False,
        )
        assert result.probabilities.shape[1] == 3


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
