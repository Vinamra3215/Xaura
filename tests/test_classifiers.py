"""Tests for XAURA classifiers (Logistic Regression + Random Forest)."""

import numpy as np
import pandas as pd
import pytest

from xaura import profile, run_model
from xaura.models import list_models, Result

# Ensure classifiers are registered
import xaura.models.classifiers  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf_df():
    """Binary classification dataset (200 rows, 4 features + target)."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.uniform(0, 10, n),
        "f4": np.random.randn(n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })


@pytest.fixture
def clf_profile(clf_df):
    """DataProfile for the classification dataset."""
    return profile(clf_df)


@pytest.fixture
def multiclass_df():
    """Multi-class classification dataset (3 classes)."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.choice([0, 1, 2], n),
    })


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------


class TestLogisticRegression:
    """Tests for LogisticRegressionModel."""

    def test_run_returns_result(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert isinstance(result, Result)

    def test_result_has_predictions(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert len(result.predictions) > 0
        assert result.predictions.dtype in [np.int64, np.int32, np.float64]

    def test_result_has_probabilities(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert result.probabilities is not None
        assert result.probabilities.shape[1] == 2  # binary

    def test_classification_metrics(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert metric in result.metrics
            assert 0.0 <= result.metrics[metric] <= 1.0

    def test_roc_auc_present(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert "roc_auc" in result.metrics

    def test_model_name_and_type(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert result.model_name == "logistic_regression"
        assert result.task_type == "classification"

    def test_feature_importances(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        # Logistic regression has coef_ → feature importances
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4  # 4 features

    def test_train_test_splits(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert result.X_train is not None
        assert result.X_test is not None
        assert result.y_train is not None
        assert result.y_test is not None
        assert len(result.X_train) + len(result.X_test) == 200

    def test_custom_config(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            config={"C": 0.01, "max_iter": 500},
            target_col="target", auto_log=False,
        )
        assert result.config["C"] == 0.01
        assert result.config["max_iter"] == 500

    def test_multiclass(self, multiclass_df):
        p = profile(multiclass_df)
        result = run_model(
            "logistic_regression", multiclass_df, p,
            target_col="target", auto_log=False,
        )
        assert result.probabilities.shape[1] == 3  # 3 classes

    def test_train_time_recorded(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, clf_df, clf_profile):
        result = run_model(
            "logistic_regression", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------


class TestRandomForestClassifier:
    """Tests for RandomForestClassifierModel."""

    def test_run_returns_result(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert isinstance(result, Result)

    def test_classification_metrics(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert metric in result.metrics
            assert 0.0 <= result.metrics[metric] <= 1.0

    def test_feature_importances(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        # Random Forest always has feature_importances_
        assert result.feature_importances is not None
        assert len(result.feature_importances) == 4

    def test_probabilities(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert result.probabilities is not None

    def test_custom_config(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier", clf_df, clf_profile,
            config={"n_estimators": 10, "max_depth": 3},
            target_col="target", auto_log=False,
        )
        assert result.config["n_estimators"] == 10
        assert result.config["max_depth"] == 3

    def test_model_object_saved(self, clf_df, clf_profile):
        result = run_model(
            "rf_classifier", clf_df, clf_profile,
            target_col="target", auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "predict")

    def test_multiclass(self, multiclass_df):
        p = profile(multiclass_df)
        result = run_model(
            "rf_classifier", multiclass_df, p,
            target_col="target", auto_log=False,
        )
        assert result.probabilities.shape[1] == 3


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests for model registry with classifiers."""

    def test_classifiers_registered(self):
        models = list_models(task_type="classification")
        names = [m["name"] for m in models]
        assert "logistic_regression" in names
        assert "rf_classifier" in names

    def test_list_models_has_display_name(self):
        models = list_models(task_type="classification")
        for m in models:
            assert "display_name" in m
            assert len(m["display_name"]) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not found"):
            run_model("nonexistent_model", pd.DataFrame(), None)

    def test_full_integration(self, clf_df):
        """End-to-end: profile → run_model → check result."""
        p = profile(clf_df)
        result = run_model(
            "rf_classifier", clf_df, p,
            target_col="target", auto_log=False,
        )
        assert isinstance(result, Result)
        assert result.metrics["accuracy"] > 0.0
        assert result.dataset_hash == p.dataset_hash
