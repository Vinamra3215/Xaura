"""Tests for XAURA clusterers (K-Means) + full integration tests."""

import numpy as np
import pandas as pd
import pytest

from xaura import profile, run_model
from xaura.models import list_models, Result
from xaura.models.clusterers.kmeans import KMeansModel

# Ensure all model families are registered
import xaura.models.classifiers  # noqa: F401
import xaura.models.regressors  # noqa: F401
import xaura.models.clusterers  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_df():
    """Clustering dataset (200 rows, 4 numeric features, no target).

    Creates 3 natural clusters using shifted Gaussians so that
    K-Means can find meaningful structure.
    """
    np.random.seed(42)
    # Three clusters centred at different points
    c1 = np.random.randn(70, 4) + np.array([0, 0, 0, 0])
    c2 = np.random.randn(65, 4) + np.array([5, 5, 5, 5])
    c3 = np.random.randn(65, 4) + np.array([-5, 5, -5, 5])
    data = np.vstack([c1, c2, c3])
    return pd.DataFrame(data, columns=["f1", "f2", "f3", "f4"])


@pytest.fixture
def cluster_profile(cluster_df):
    """DataProfile for the clustering dataset."""
    return profile(cluster_df)


# ---------------------------------------------------------------------------
# K-Means basic tests
# ---------------------------------------------------------------------------


class TestKMeans:
    """Tests for KMeansModel."""

    def test_run_returns_result(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_model_name_and_type(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert result.model_name == "kmeans"
        assert result.task_type == "clustering"

    def test_cluster_labels_valid(self, cluster_df, cluster_profile):
        """Labels should be integers in [0, n_clusters)."""
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        unique_labels = set(result.predictions)
        assert unique_labels == {0, 1, 2}

    def test_predictions_length(self, cluster_df, cluster_profile):
        """Should have one label per row."""
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert len(result.predictions) == 200

    def test_silhouette_score(self, cluster_df, cluster_profile):
        """Silhouette should be between -1 and 1, and high for well-separated clusters."""
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert "silhouette" in result.metrics
        assert -1.0 <= result.metrics["silhouette"] <= 1.0
        # Our clusters are well-separated, so silhouette should be decent
        assert result.metrics["silhouette"] > 0.3

    def test_calinski_harabasz(self, cluster_df, cluster_profile):
        """Calinski-Harabasz index should be positive for valid clustering."""
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert "calinski_harabasz" in result.metrics
        assert result.metrics["calinski_harabasz"] > 0

    def test_no_probabilities(self, cluster_df, cluster_profile):
        """Clustering should NOT produce probabilities."""
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert result.probabilities is None

    def test_no_train_test_split(self, cluster_df, cluster_profile):
        """Clustering uses all data — X_train holds the full dataset."""
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        # X_train should hold the full dataset for clustering
        assert result.X_train is not None
        assert len(result.X_train) == 200
        # No test split for clustering
        assert result.X_test is None
        assert result.y_test is None

    def test_custom_config(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            config={"n_clusters": 5, "n_init": 5},
            auto_log=False,
        )
        assert result.config["n_clusters"] == 5
        unique_labels = set(result.predictions)
        assert len(unique_labels) == 5

    def test_model_object_saved(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "cluster_centers_")

    def test_train_time_recorded(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans", cluster_df, cluster_profile,
            auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Elbow method
# ---------------------------------------------------------------------------


class TestElbowMethod:
    """Tests for the elbow method helper."""

    def test_elbow_returns_dict(self, cluster_df):
        X = cluster_df.values
        result = KMeansModel.elbow_method(X)
        assert isinstance(result, dict)
        assert "k_range" in result
        assert "inertias" in result
        assert "suggested_k" in result

    def test_elbow_k_range(self, cluster_df):
        X = cluster_df.values
        result = KMeansModel.elbow_method(X, k_range=range(2, 8))
        assert result["k_range"] == [2, 3, 4, 5, 6, 7]
        assert len(result["inertias"]) == 6

    def test_elbow_inertias_decrease(self, cluster_df):
        """Inertia should monotonically decrease as k increases."""
        X = cluster_df.values
        result = KMeansModel.elbow_method(X)
        inertias = result["inertias"]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_elbow_suggested_k_reasonable(self, cluster_df):
        """For 3 well-separated clusters, suggested k should be around 3."""
        X = cluster_df.values
        result = KMeansModel.elbow_method(X)
        # Should be 2, 3, or 4 — near the true value of 3
        assert 2 <= result["suggested_k"] <= 5

    def test_elbow_suggested_k_in_range(self, cluster_df):
        X = cluster_df.values
        result = KMeansModel.elbow_method(X, k_range=range(2, 11))
        assert result["suggested_k"] in result["k_range"]


# ---------------------------------------------------------------------------
# Registry + full integration
# ---------------------------------------------------------------------------


class TestClustererRegistry:
    """Tests for model registry with clusterers."""

    def test_kmeans_registered(self):
        models = list_models(task_type="clustering")
        names = [m["name"] for m in models]
        assert "kmeans" in names

    def test_all_person_a_models_registered(self):
        """All 6 Person A models should be registered."""
        all_models = list_models()
        names = [m["name"] for m in all_models]
        expected = [
            "logistic_regression",
            "rf_classifier",
            "linear_regression",
            "ridge",
            "lasso",
            "kmeans",
        ]
        for name in expected:
            assert name in names, f"Model '{name}' not found in registry"

    def test_list_models_filter_by_type(self):
        """Filtering by task type should return only matching models."""
        clf = list_models(task_type="classification")
        reg = list_models(task_type="regression")
        clu = list_models(task_type="clustering")
        assert all(m["task_type"] == "classification" for m in clf)
        assert all(m["task_type"] == "regression" for m in reg)
        assert all(m["task_type"] == "clustering" for m in clu)

    def test_full_integration_clustering(self, cluster_df):
        """End-to-end: profile → run_model → check result."""
        p = profile(cluster_df)
        result = run_model("kmeans", cluster_df, p, auto_log=False)
        assert isinstance(result, Result)
        assert result.task_type == "clustering"
        assert "silhouette" in result.metrics
        assert result.dataset_hash == p.dataset_hash

    def test_full_integration_all_model_types(self):
        """Run one model from each type to verify everything works together."""
        np.random.seed(42)

        # Classification
        clf_df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        })
        clf_p = profile(clf_df)
        clf_result = run_model(
            "logistic_regression", clf_df, clf_p,
            target_col="target", auto_log=False,
        )
        assert clf_result.task_type == "classification"
        assert "accuracy" in clf_result.metrics

        # Regression
        reg_df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "target": np.random.randn(100),
        })
        reg_p = profile(reg_df)
        reg_result = run_model(
            "ridge", reg_df, reg_p,
            target_col="target", auto_log=False,
        )
        assert reg_result.task_type == "regression"
        assert "r2" in reg_result.metrics

        # Clustering
        clu_df = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
        })
        clu_p = profile(clu_df)
        clu_result = run_model("kmeans", clu_df, clu_p, auto_log=False)
        assert clu_result.task_type == "clustering"
        assert "silhouette" in clu_result.metrics
