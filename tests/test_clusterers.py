"""Tests for XAURA clusterers (K-Means) + full integration tests."""

import numpy as np
import pandas as pd
import pytest

import xaura.models.classifiers  # noqa: F401
import xaura.models.clusterers  # noqa: F401
import xaura.models.regressors  # noqa: F401
from xaura import profile, run_model
from xaura.models import Result, list_models
from xaura.models.clusterers.hierarchical import HierarchicalModel
from xaura.models.clusterers.kmeans import KMeansModel

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
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_model_name_and_type(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.model_name == "kmeans"
        assert result.task_type == "clustering"

    def test_cluster_labels_valid(self, cluster_df, cluster_profile):
        """Labels should be integers in [0, n_clusters)."""
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        unique_labels = set(result.predictions)
        assert unique_labels == {0, 1, 2}

    def test_predictions_length(self, cluster_df, cluster_profile):
        """Should have one label per row."""
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert len(result.predictions) == 200

    def test_silhouette_score(self, cluster_df, cluster_profile):
        """Silhouette should be between -1 and 1, and high for well-separated clusters."""
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert "silhouette" in result.metrics
        assert -1.0 <= result.metrics["silhouette"] <= 1.0
        # Our clusters are well-separated, so silhouette should be decent
        assert result.metrics["silhouette"] > 0.3

    def test_calinski_harabasz(self, cluster_df, cluster_profile):
        """Calinski-Harabasz index should be positive for valid clustering."""
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert "calinski_harabasz" in result.metrics
        assert result.metrics["calinski_harabasz"] > 0

    def test_no_probabilities(self, cluster_df, cluster_profile):
        """Clustering should NOT produce probabilities."""
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.probabilities is None

    def test_no_train_test_split(self, cluster_df, cluster_profile):
        """Clustering uses all data — X_train holds the full dataset."""
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
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
            "kmeans",
            cluster_df,
            cluster_profile,
            config={"n_clusters": 5, "n_init": 5},
            auto_log=False,
        )
        assert result.config["n_clusters"] == 5
        unique_labels = set(result.predictions)
        assert len(unique_labels) == 5

    def test_model_object_saved(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "cluster_centers_")

    def test_train_time_recorded(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, cluster_df, cluster_profile):
        result = run_model(
            "kmeans",
            cluster_df,
            cluster_profile,
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

    def test_all_person_b_models_registered(self):
        """All 7 Person B models should be registered."""
        all_models = list_models()
        names = [m["name"] for m in all_models]
        expected = [
            "xgboost_cls",
            "lightgbm_cls",
            "random_forest_reg",
            "xgboost_reg",
            "dbscan",
            "hierarchical",
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

    def test_clustering_models_count(self):
        """Should have exactly 3 clustering models."""
        clu = list_models(task_type="clustering")
        names = sorted(m["name"] for m in clu)
        assert names == ["dbscan", "hierarchical", "kmeans"]

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
        clf_df = pd.DataFrame(
            {
                "f1": np.random.randn(100),
                "f2": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )
        clf_p = profile(clf_df)
        clf_result = run_model(
            "logistic_regression",
            clf_df,
            clf_p,
            target_col="target",
            auto_log=False,
        )
        assert clf_result.task_type == "classification"
        assert "accuracy" in clf_result.metrics

        # Regression
        reg_df = pd.DataFrame(
            {
                "f1": np.random.randn(100),
                "f2": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )
        reg_p = profile(reg_df)
        reg_result = run_model(
            "ridge",
            reg_df,
            reg_p,
            target_col="target",
            auto_log=False,
        )
        assert reg_result.task_type == "regression"
        assert "r2" in reg_result.metrics

        # Clustering
        clu_df = pd.DataFrame(
            {
                "f1": np.random.randn(100),
                "f2": np.random.randn(100),
            }
        )
        clu_p = profile(clu_df)
        clu_result = run_model("kmeans", clu_df, clu_p, auto_log=False)
        assert clu_result.task_type == "clustering"
        assert "silhouette" in clu_result.metrics


# ---------------------------------------------------------------------------
# DBSCAN tests
# ---------------------------------------------------------------------------


class TestDBSCAN:
    """Tests for DBSCANModel."""

    def test_run_returns_result(self, cluster_df, cluster_profile):
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_model_name_and_type(self, cluster_df, cluster_profile):
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.model_name == "dbscan"
        assert result.task_type == "clustering"

    def test_predictions_length(self, cluster_df, cluster_profile):
        """Should have one label per row."""
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert len(result.predictions) == 200

    def test_noise_labels_possible(self, cluster_df, cluster_profile):
        """DBSCAN can label points as noise (-1)."""
        # Use an eps that finds clusters but may still produce some noise
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            config={"eps": 1.5, "min_samples": 5},
            auto_log=False,
        )
        # Labels are integers; noise is -1, clusters are 0, 1, 2, ...
        unique = set(result.predictions)
        # At least one cluster must be found (could also have noise)
        non_noise = {lab for lab in unique if lab >= 0}
        assert len(non_noise) >= 1

    def test_well_separated_clusters(self):
        """With proper eps, DBSCAN should find the 3 clusters."""
        np.random.seed(42)
        c1 = np.random.randn(70, 2) + np.array([0, 0])
        c2 = np.random.randn(65, 2) + np.array([10, 10])
        c3 = np.random.randn(65, 2) + np.array([-10, 10])
        data = np.vstack([c1, c2, c3])
        df = pd.DataFrame(data, columns=["f1", "f2"])
        p = profile(df)
        result = run_model(
            "dbscan",
            df,
            p,
            config={"eps": 2.0, "min_samples": 5},
            auto_log=False,
        )
        non_noise = {lab for lab in result.predictions if lab >= 0}
        assert len(non_noise) == 3

    def test_silhouette_present_when_multiple_clusters(self):
        """Silhouette should be computed when DBSCAN finds ≥2 clusters."""
        np.random.seed(42)
        c1 = np.random.randn(70, 2) + np.array([0, 0])
        c2 = np.random.randn(70, 2) + np.array([10, 10])
        data = np.vstack([c1, c2])
        df = pd.DataFrame(data, columns=["f1", "f2"])
        p = profile(df)
        result = run_model(
            "dbscan",
            df,
            p,
            config={"eps": 2.0, "min_samples": 5},
            auto_log=False,
        )
        non_noise = {lab for lab in result.predictions if lab >= 0}
        if len(non_noise) >= 2:
            assert "silhouette" in result.metrics
            assert -1.0 <= result.metrics["silhouette"] <= 1.0

    def test_custom_config(self, cluster_df, cluster_profile):
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            config={"eps": 1.0, "min_samples": 10},
            auto_log=False,
        )
        assert result.config["eps"] == 1.0
        assert result.config["min_samples"] == 10

    def test_no_probabilities(self, cluster_df, cluster_profile):
        """Clustering should NOT produce probabilities."""
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.probabilities is None

    def test_no_train_test_split(self, cluster_df, cluster_profile):
        """Clustering uses all data — X_train holds the full dataset."""
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.X_train is not None
        assert len(result.X_train) == 200
        assert result.X_test is None

    def test_train_time_recorded(self, cluster_df, cluster_profile):
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, cluster_df, cluster_profile):
        result = run_model(
            "dbscan",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Hierarchical Clustering tests
# ---------------------------------------------------------------------------


class TestHierarchical:
    """Tests for HierarchicalModel."""

    def test_run_returns_result(self, cluster_df, cluster_profile):
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert isinstance(result, Result)

    def test_model_name_and_type(self, cluster_df, cluster_profile):
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.model_name == "hierarchical"
        assert result.task_type == "clustering"

    def test_cluster_labels_valid(self, cluster_df, cluster_profile):
        """Labels should be integers in [0, n_clusters)."""
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        unique_labels = set(result.predictions)
        assert unique_labels == {0, 1, 2}

    def test_predictions_length(self, cluster_df, cluster_profile):
        """Should have one label per row."""
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert len(result.predictions) == 200

    def test_silhouette_score(self, cluster_df, cluster_profile):
        """Silhouette should be between -1 and 1, and decent for well-separated data."""
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert "silhouette" in result.metrics
        assert -1.0 <= result.metrics["silhouette"] <= 1.0
        assert result.metrics["silhouette"] > 0.3

    def test_calinski_harabasz(self, cluster_df, cluster_profile):
        """Calinski-Harabasz index should be positive for valid clustering."""
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert "calinski_harabasz" in result.metrics
        assert result.metrics["calinski_harabasz"] > 0

    def test_no_probabilities(self, cluster_df, cluster_profile):
        """Clustering should NOT produce probabilities."""
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.probabilities is None

    def test_no_train_test_split(self, cluster_df, cluster_profile):
        """Clustering uses all data."""
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.X_train is not None
        assert len(result.X_train) == 200
        assert result.X_test is None

    def test_custom_config(self, cluster_df, cluster_profile):
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            config={"n_clusters": 5, "linkage": "complete"},
            auto_log=False,
        )
        assert result.config["n_clusters"] == 5
        assert result.config["linkage"] == "complete"
        unique_labels = set(result.predictions)
        assert len(unique_labels) == 5

    def test_model_object_saved(self, cluster_df, cluster_profile):
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.model_object is not None
        assert hasattr(result.model_object, "labels_")

    def test_train_time_recorded(self, cluster_df, cluster_profile):
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert result.train_time_seconds > 0.0

    def test_dataset_hash(self, cluster_df, cluster_profile):
        result = run_model(
            "hierarchical",
            cluster_df,
            cluster_profile,
            auto_log=False,
        )
        assert len(result.dataset_hash) == 64


# ---------------------------------------------------------------------------
# Dendrogram helper tests
# ---------------------------------------------------------------------------


class TestDendrogram:
    """Tests for the dendrogram linkage matrix helper."""

    def test_linkage_returns_array(self, cluster_df):
        X = cluster_df.values
        Z = HierarchicalModel.compute_linkage_matrix(X)
        assert isinstance(Z, np.ndarray)

    def test_linkage_shape(self, cluster_df):
        """Linkage matrix should be (n_samples-1, 4)."""
        X = cluster_df.values
        Z = HierarchicalModel.compute_linkage_matrix(X)
        assert Z.shape == (len(X) - 1, 4)

    def test_linkage_methods(self, cluster_df):
        """All standard linkage methods should work."""
        X = cluster_df.values
        for method in ["ward", "complete", "average", "single"]:
            Z = HierarchicalModel.compute_linkage_matrix(X, method=method)
            assert Z.shape[0] == len(X) - 1
