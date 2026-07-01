"""Tests for XAURA Plotly visualisation charts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Ensure models are registered
import xaura.models.classifiers  # noqa: F401
import xaura.models.regressors  # noqa: F401
from xaura import profile, run_model
from xaura.models import Result
from xaura.visualisation import (
    config_panel,
    confusion_matrix_chart,
    feature_importance_chart,
    metrics_card,
    precision_recall_chart,
    profile_summary_panel,
    roc_curve_chart,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clf_df():
    """Binary classification dataset."""
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
def clf_result(clf_df):
    """Result from running RF classifier on binary dataset."""
    p = profile(clf_df)
    return run_model("rf_classifier", clf_df, p, target_col="target", auto_log=False)


@pytest.fixture
def clf_profile(clf_df):
    """DataProfile for the classification dataset."""
    return profile(clf_df)


@pytest.fixture
def multiclass_df():
    """Multi-class classification dataset (3 classes)."""
    np.random.seed(42)
    n = 150
    return pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.choice([0, 1, 2], n),
        }
    )


@pytest.fixture
def multiclass_result(multiclass_df):
    """Result from running RF classifier on multiclass dataset."""
    p = profile(multiclass_df)
    return run_model("rf_classifier", multiclass_df, p, target_col="target", auto_log=False)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    """Tests for confusion_matrix_chart."""

    def test_returns_figure(self, clf_result):
        fig = confusion_matrix_chart(clf_result)
        assert isinstance(fig, go.Figure)

    def test_heatmap_shape_binary(self, clf_result):
        """Binary classification → 2×2 matrix."""
        fig = confusion_matrix_chart(clf_result)
        heatmap = fig.data[0]
        assert np.array(heatmap.z).shape == (2, 2)

    def test_heatmap_shape_multiclass(self, multiclass_result):
        """3-class classification → 3×3 matrix."""
        fig = confusion_matrix_chart(multiclass_result)
        heatmap = fig.data[0]
        assert np.array(heatmap.z).shape == (3, 3)

    def test_has_title(self, clf_result):
        fig = confusion_matrix_chart(clf_result)
        assert "Confusion Matrix" in fig.layout.title.text

    def test_has_axis_labels(self, clf_result):
        fig = confusion_matrix_chart(clf_result)
        assert fig.layout.xaxis.title.text == "Predicted Label"
        assert fig.layout.yaxis.title.text == "Actual Label"

    def test_json_export(self, clf_result):
        """Chart must be exportable to JSON for web UI."""
        fig = confusion_matrix_chart(clf_result)
        json_str = fig.to_json()
        assert len(json_str) > 0
        assert "Confusion Matrix" in json_str


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------


class TestROCCurve:
    """Tests for roc_curve_chart."""

    def test_returns_figure(self, clf_result):
        fig = roc_curve_chart(clf_result)
        assert isinstance(fig, go.Figure)

    def test_binary_has_two_traces(self, clf_result):
        """Binary: one ROC curve + one diagonal reference line."""
        fig = roc_curve_chart(clf_result)
        assert len(fig.data) == 2

    def test_multiclass_has_per_class_curves(self, multiclass_result):
        """Multiclass: one curve per class + diagonal = 4 traces."""
        fig = roc_curve_chart(multiclass_result)
        assert len(fig.data) == 4  # 3 classes + diagonal

    def test_auc_in_legend(self, clf_result):
        """AUC value should appear in trace name."""
        fig = roc_curve_chart(clf_result)
        roc_trace = fig.data[0]
        assert "AUC" in roc_trace.name

    def test_diagonal_reference(self, clf_result):
        """Should have a diagonal dashed reference line."""
        fig = roc_curve_chart(clf_result)
        diagonal = fig.data[-1]
        assert diagonal.line.dash == "dash"

    def test_raises_without_probabilities(self):
        """Should raise ValueError if no probabilities."""
        result = Result(
            model_name="test",
            task_type="classification",
            predictions=np.array([0, 1]),
            probabilities=None,
        )
        with pytest.raises(ValueError, match="probabilities"):
            roc_curve_chart(result)

    def test_has_title(self, clf_result):
        fig = roc_curve_chart(clf_result)
        assert "ROC" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Precision-Recall Curve
# ---------------------------------------------------------------------------


class TestPrecisionRecall:
    """Tests for precision_recall_chart."""

    def test_returns_figure(self, clf_result):
        fig = precision_recall_chart(clf_result)
        assert isinstance(fig, go.Figure)

    def test_binary_has_one_curve(self, clf_result):
        fig = precision_recall_chart(clf_result)
        assert len(fig.data) == 1

    def test_multiclass_has_per_class_curves(self, multiclass_result):
        fig = precision_recall_chart(multiclass_result)
        assert len(fig.data) == 3  # one per class

    def test_auc_in_legend(self, clf_result):
        fig = precision_recall_chart(clf_result)
        assert "AUC" in fig.data[0].name

    def test_raises_without_probabilities(self):
        result = Result(
            model_name="test",
            task_type="classification",
            predictions=np.array([0, 1]),
            probabilities=None,
        )
        with pytest.raises(ValueError, match="probabilities"):
            precision_recall_chart(result)

    def test_has_title(self, clf_result):
        fig = precision_recall_chart(clf_result)
        assert "Precision" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    """Tests for feature_importance_chart."""

    def test_returns_figure(self, clf_result):
        fig = feature_importance_chart(clf_result)
        assert isinstance(fig, go.Figure)

    def test_bar_count_matches_features(self, clf_result):
        """Should have one bar per feature."""
        fig = feature_importance_chart(clf_result)
        bars = fig.data[0]
        assert len(bars.x) == 4  # 4 features

    def test_bars_sorted_ascending(self, clf_result):
        """Bars sorted ascending (most important at top in horizontal bar)."""
        fig = feature_importance_chart(clf_result)
        values = list(fig.data[0].x)
        assert values == sorted(values)

    def test_raises_without_importances(self):
        result = Result(
            model_name="test",
            task_type="classification",
            predictions=np.array([0, 1]),
            feature_importances=None,
        )
        with pytest.raises(ValueError, match="feature_importances"):
            feature_importance_chart(result)

    def test_has_title(self, clf_result):
        fig = feature_importance_chart(clf_result)
        assert "Feature" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Profile Summary Panel
# ---------------------------------------------------------------------------


class TestProfileSummary:
    """Tests for profile_summary_panel."""

    def test_returns_figure(self, clf_profile):
        fig = profile_summary_panel(clf_profile)
        assert isinstance(fig, go.Figure)

    def test_contains_table(self, clf_profile):
        fig = profile_summary_panel(clf_profile)
        assert isinstance(fig.data[0], go.Table)

    def test_shows_row_count(self, clf_profile):
        fig = profile_summary_panel(clf_profile)
        # Values are in the second column of the table
        values = fig.data[0].cells.values[1]
        # First value should contain the row×col info
        assert "200" in str(values[0])

    def test_has_title(self, clf_profile):
        fig = profile_summary_panel(clf_profile)
        assert "Profile" in fig.layout.title.text


# ---------------------------------------------------------------------------
# Metrics Card
# ---------------------------------------------------------------------------


class TestMetricsCard:
    """Tests for metrics_card."""

    def test_returns_figure(self, clf_result):
        fig = metrics_card(clf_result)
        assert isinstance(fig, go.Figure)

    def test_contains_table(self, clf_result):
        fig = metrics_card(clf_result)
        assert isinstance(fig.data[0], go.Table)

    def test_shows_model_name(self, clf_result):
        fig = metrics_card(clf_result)
        values = fig.data[0].cells.values[1]
        assert "rf_classifier" in values

    def test_shows_accuracy(self, clf_result):
        fig = metrics_card(clf_result)
        names = fig.data[0].cells.values[0]
        assert "ACCURACY" in names


# ---------------------------------------------------------------------------
# Config Panel
# ---------------------------------------------------------------------------


class TestConfigPanel:
    """Tests for config_panel."""

    def test_returns_figure(self, clf_result):
        fig = config_panel(clf_result)
        assert isinstance(fig, go.Figure)

    def test_contains_table(self, clf_result):
        fig = config_panel(clf_result)
        assert isinstance(fig.data[0], go.Table)

    def test_shows_config_keys(self, clf_result):
        fig = config_panel(clf_result)
        keys = list(fig.data[0].cells.values[0])
        # RF classifier should have n_estimators in config
        assert "n_estimators" in keys

    def test_empty_config(self):
        """Should handle empty config gracefully."""
        result = Result(model_name="test", task_type="test", config={})
        fig = config_panel(result)
        assert isinstance(fig, go.Figure)
