"""Tests for the dataset-aware default hyperparameter engine.

Tests that get_defaults() returns sensible configs for all supported
models, adapts to dataset characteristics (size, imbalance, feature
count), and raises on unknown model names.
"""

import pytest

from xaura.models.defaults import get_defaults
from xaura.profiler.dataprofile import DataProfile

# ─────────────────────────────────────────────────────────────
# Test fixtures — synthetic DataProfile objects
# ─────────────────────────────────────────────────────────────


def _make_profile(
    n_rows: int = 5000,
    n_cols: int = 10,
    n_numeric: int = 8,
    n_categorical: int = 1,
    n_binary: int = 1,
    is_imbalanced: bool = False,
    imbalance_ratio: float = 2.0,
) -> DataProfile:
    """Create a synthetic DataProfile for testing defaults."""
    numeric_cols = [f"num_{i}" for i in range(n_numeric)]
    cat_cols = [f"cat_{i}" for i in range(n_categorical)]
    bin_cols = [f"bin_{i}" for i in range(n_binary)]

    profile = DataProfile(
        shape=(n_rows, n_cols),
        feature_types={
            "numeric": numeric_cols,
            "categorical": cat_cols,
            "binary": bin_cols,
        },
        missing_values={},
    )

    if is_imbalanced:
        majority = int(n_rows * imbalance_ratio / (1 + imbalance_ratio))
        minority = n_rows - majority
        profile.class_balance = {
            "counts": {0: majority, 1: minority},
            "ratio": imbalance_ratio,
            "majority_class": 0,
            "minority_class": 1,
            "n_classes": 2,
        }

    return profile


@pytest.fixture
def small_profile():
    """Small dataset: 500 rows, 5 cols."""
    return _make_profile(n_rows=500, n_cols=5, n_numeric=3, n_categorical=1, n_binary=1)


@pytest.fixture
def medium_profile():
    """Medium dataset: 5000 rows, 10 cols."""
    return _make_profile(n_rows=5000, n_cols=10)


@pytest.fixture
def large_profile():
    """Large dataset: 200k rows, 30 cols."""
    return _make_profile(n_rows=200_000, n_cols=30, n_numeric=25, n_categorical=3, n_binary=2)


@pytest.fixture
def high_dim_profile():
    """High-dimensional: 5000 rows, 80 cols (60 numeric)."""
    return _make_profile(n_rows=5000, n_cols=80, n_numeric=60, n_categorical=15, n_binary=5)


@pytest.fixture
def imbalanced_profile():
    """Imbalanced dataset: 10:1 class ratio."""
    return _make_profile(n_rows=5000, n_cols=10, is_imbalanced=True, imbalance_ratio=10.0)


# ─────────────────────────────────────────────────────────────
# Basic dispatch tests
# ─────────────────────────────────────────────────────────────


class TestDispatch:
    """Tests that get_defaults dispatches correctly for all models."""

    ALL_MODELS = [
        "xgboost_cls",
        "lightgbm_cls",
        "random_forest_reg",
        "xgboost_reg",
        "logistic",
        "random_forest",
        "linear",
        "ridge",
        "lasso",
    ]

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_returns_dict(self, medium_profile, model_name):
        """Every model should return a non-empty dict."""
        config = get_defaults(medium_profile, model_name)
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_unknown_model_raises(self, medium_profile):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_defaults(medium_profile, "nonexistent_model")


# ─────────────────────────────────────────────────────────────
# Size adaptation tests
# ─────────────────────────────────────────────────────────────


class TestSizeAdaptation:
    """Tests that tree-based models adapt to dataset size."""

    def test_small_data_fewer_estimators(self, small_profile):
        config = get_defaults(small_profile, "xgboost_cls")
        assert config["n_estimators"] == 100

    def test_medium_data_moderate_estimators(self, medium_profile):
        config = get_defaults(medium_profile, "xgboost_cls")
        assert config["n_estimators"] == 200

    def test_large_data_more_estimators(self, large_profile):
        config = get_defaults(large_profile, "xgboost_cls")
        assert config["n_estimators"] == 500

    def test_small_data_shallow_depth(self, small_profile):
        config = get_defaults(small_profile, "xgboost_cls")
        assert config["max_depth"] <= 4

    def test_large_data_enables_subsampling(self, large_profile):
        config = get_defaults(large_profile, "xgboost_cls")
        assert config["subsample"] < 1.0

    def test_small_data_no_subsampling(self, small_profile):
        config = get_defaults(small_profile, "xgboost_cls")
        assert config["subsample"] == 1.0


# ─────────────────────────────────────────────────────────────
# Imbalance handling tests
# ─────────────────────────────────────────────────────────────


class TestImbalanceHandling:
    """Tests that classifiers adapt to imbalanced data."""

    def test_xgboost_scale_pos_weight(self, imbalanced_profile):
        config = get_defaults(imbalanced_profile, "xgboost_cls")
        assert "scale_pos_weight" in config
        assert config["scale_pos_weight"] > 1.0

    def test_lightgbm_is_unbalance(self, imbalanced_profile):
        config = get_defaults(imbalanced_profile, "lightgbm_cls")
        assert config.get("is_unbalance") is True

    def test_logistic_balanced_weight(self, imbalanced_profile):
        config = get_defaults(imbalanced_profile, "logistic")
        assert config.get("class_weight") == "balanced"

    def test_rf_classifier_balanced_weight(self, imbalanced_profile):
        config = get_defaults(imbalanced_profile, "random_forest")
        assert config.get("class_weight") == "balanced"

    def test_balanced_data_no_weight(self, medium_profile):
        """Non-imbalanced data should NOT have class weighting."""
        config = get_defaults(medium_profile, "xgboost_cls")
        assert "scale_pos_weight" not in config


# ─────────────────────────────────────────────────────────────
# High-dimensional data tests
# ─────────────────────────────────────────────────────────────


class TestHighDimensional:
    """Tests that models add regularisation for many features."""

    def test_xgboost_stronger_regularisation(self, high_dim_profile):
        config = get_defaults(high_dim_profile, "xgboost_cls")
        assert config["reg_alpha"] > 0

    def test_xgboost_column_subsampling(self, high_dim_profile):
        config = get_defaults(high_dim_profile, "xgboost_cls")
        assert config["colsample_bytree"] < 1.0

    def test_logistic_lower_c(self, high_dim_profile):
        config = get_defaults(high_dim_profile, "logistic")
        assert config["C"] <= 0.1

    def test_ridge_higher_alpha(self, high_dim_profile):
        config = get_defaults(high_dim_profile, "ridge")
        assert config["alpha"] >= 10.0


# ─────────────────────────────────────────────────────────────
# Model-specific key tests
# ─────────────────────────────────────────────────────────────


class TestModelSpecificKeys:
    """Tests that each model returns its expected keys."""

    def test_xgboost_cls_keys(self, medium_profile):
        config = get_defaults(medium_profile, "xgboost_cls")
        expected = {"n_estimators", "max_depth", "learning_rate", "eval_metric", "random_state"}
        assert expected.issubset(set(config.keys()))
        assert config["eval_metric"] == "logloss"

    def test_lightgbm_cls_keys(self, medium_profile):
        config = get_defaults(medium_profile, "lightgbm_cls")
        assert "num_leaves" in config
        assert "verbose" in config
        assert config["verbose"] == -1

    def test_lightgbm_no_max_depth(self, medium_profile):
        """LightGBM uses num_leaves, not max_depth."""
        config = get_defaults(medium_profile, "lightgbm_cls")
        assert "max_depth" not in config

    def test_random_forest_reg_keys(self, medium_profile):
        config = get_defaults(medium_profile, "random_forest_reg")
        assert "bootstrap" in config
        assert config["bootstrap"] is True
        assert "subsample" not in config  # RF doesn't use subsample

    def test_xgboost_reg_keys(self, medium_profile):
        config = get_defaults(medium_profile, "xgboost_reg")
        assert config["eval_metric"] == "rmse"

    def test_linear_minimal(self, medium_profile):
        config = get_defaults(medium_profile, "linear")
        assert "fit_intercept" in config

    def test_lasso_has_max_iter(self, medium_profile):
        config = get_defaults(medium_profile, "lasso")
        assert config["max_iter"] == 2000

    def test_all_tree_models_have_random_state(self, medium_profile):
        """All tree-based models should be reproducible."""
        for name in ["xgboost_cls", "lightgbm_cls", "random_forest_reg", "xgboost_reg"]:
            config = get_defaults(medium_profile, name)
            assert config["random_state"] == 42, f"{name} missing random_state=42"
