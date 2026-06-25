"""Dataset-aware default hyperparameter engine.

Reads a DataProfile and a model name, returns a smart config dict
with hyperparameters tuned to the actual dataset characteristics.

This module is a pure function — no ML library imports, no side effects.
The model wrappers (xgboost_cls, lightgbm_cls, etc.) consume the dict
it returns.

Design decisions:
    - Pure function: get_defaults(profile, model_name) → dict
    - No sklearn/xgboost/lightgbm imports — just returns config dicts
    - Model names match the registry: "xgboost_cls", "lightgbm_cls",
      "random_forest_reg", "xgboost_reg", "logistic", "random_forest",
      "linear", "ridge", "lasso"
    - Shared logic for tree-based models via _base_tree_defaults()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xaura.profiler.dataprofile import DataProfile


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def get_defaults(profile: DataProfile, model_name: str) -> dict[str, Any]:
    """Return dataset-aware default hyperparameters for a given model.

    Inspects the DataProfile (shape, feature types, class balance,
    missing values) and picks sensible defaults accordingly.

    Args:
        profile: A DataProfile instance from xaura.profiler.
        model_name: Registry name of the model, e.g. "xgboost_cls".

    Returns:
        Dict of hyperparameter name → value. Keys are the exact
        kwarg names that the underlying ML library expects.

    Raises:
        ValueError: If model_name is not recognised.

    Example:
        >>> from xaura.profiler.profiler import profile
        >>> from xaura.models.defaults import get_defaults
        >>> dp = profile(df, target="target")
        >>> config = get_defaults(dp, "xgboost_cls")
        >>> config
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1, ...}
    """
    dispatch: dict[str, _DefaultsFn] = {
        # Person B's models (Week 2)
        "xgboost_cls": _xgboost_cls_defaults,
        "lightgbm_cls": _lightgbm_cls_defaults,
        "random_forest_reg": _random_forest_reg_defaults,
        "xgboost_reg": _xgboost_reg_defaults,
        # Person A's models (Week 2) — we provide defaults for them too
        "logistic": _logistic_defaults,
        "random_forest": _random_forest_cls_defaults,
        "linear": _linear_defaults,
        "ridge": _ridge_defaults,
        "lasso": _lasso_defaults,
    }

    fn = dispatch.get(model_name)
    if fn is None:
        raise ValueError(
            f"Unknown model name: {model_name!r}. " f"Supported: {sorted(dispatch.keys())}"
        )

    return fn(profile)


# ─────────────────────────────────────────────────────────────
# Type alias for dispatch functions
# ─────────────────────────────────────────────────────────────

from collections.abc import Callable  # noqa: E402

_DefaultsFn = Callable[["DataProfile"], dict[str, Any]]


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────


def _base_tree_defaults(profile: DataProfile) -> dict[str, Any]:
    """Shared defaults for all tree-based models.

    Adjusts complexity based on dataset size:
    - Small datasets (< 1k rows): shallow trees, fewer estimators
    - Medium datasets (1k–100k): balanced defaults
    - Large datasets (> 100k): more estimators, subsampling enabled

    Args:
        profile: DataProfile instance.

    Returns:
        Base config dict that model-specific functions extend.
    """
    n_rows, n_cols = profile.shape

    # --- Number of estimators scales with data size ---
    if profile.is_small:
        n_estimators = 100
    elif profile.is_large:
        n_estimators = 500
    else:
        n_estimators = 200

    # --- Tree depth: deeper for more features, shallower for small data ---
    if n_cols <= 5:
        max_depth = 4
    elif n_cols <= 20:
        max_depth = 6
    else:
        max_depth = 8

    # Clamp depth for small datasets to avoid overfitting
    if profile.is_small:
        max_depth = min(max_depth, 4)

    # --- Subsampling for large datasets ---
    subsample = 0.8 if profile.is_large else 1.0

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "subsample": subsample,
        "random_state": 42,
    }


def _imbalance_weight(profile: DataProfile) -> float | None:
    """Compute scale_pos_weight for imbalanced binary classification.

    Returns the ratio of majority to minority class count, or None
    if the dataset is not imbalanced or not binary.

    Args:
        profile: DataProfile instance.

    Returns:
        Float weight for positive class, or None.
    """
    if not profile.is_imbalanced:
        return None

    if profile.class_balance and "counts" in profile.class_balance:
        counts = profile.class_balance["counts"]
        if isinstance(counts, dict) and len(counts) == 2:
            values = sorted(counts.values())
            if values[0] > 0:
                return values[1] / values[0]

    return None


def _n_features(profile: DataProfile) -> int:
    """Total number of features (numeric + categorical + binary)."""
    ft = profile.feature_types
    return len(ft.get("numeric", [])) + len(ft.get("categorical", [])) + len(ft.get("binary", []))


# ─────────────────────────────────────────────────────────────
# Person B's models — classifiers
# ─────────────────────────────────────────────────────────────


def _xgboost_cls_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for XGBoost classifier.

    Key adaptations:
    - scale_pos_weight for imbalanced data
    - learning_rate inversely related to n_estimators
    - eval_metric set to 'logloss' for classification
    """
    config = _base_tree_defaults(profile)

    # XGBoost-specific learning rate
    config["learning_rate"] = 0.05 if config["n_estimators"] >= 300 else 0.1

    # Regularisation — stronger for high-dimensional data
    n_feat = _n_features(profile)
    if n_feat > 50:
        config["reg_alpha"] = 0.1  # L1
        config["reg_lambda"] = 1.0  # L2
    else:
        config["reg_alpha"] = 0.0
        config["reg_lambda"] = 1.0

    # Minimum child weight — higher for large datasets to prevent overfitting
    config["min_child_weight"] = 5 if profile.is_large else 1

    # Column subsampling for high-dimensional data
    if n_feat > 20:
        config["colsample_bytree"] = 0.8
    else:
        config["colsample_bytree"] = 1.0

    # Imbalance handling
    weight = _imbalance_weight(profile)
    if weight is not None:
        config["scale_pos_weight"] = round(weight, 2)

    # Evaluation metric
    config["eval_metric"] = "logloss"

    return config


def _lightgbm_cls_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for LightGBM classifier.

    Key adaptations:
    - num_leaves instead of max_depth (LightGBM is leaf-wise)
    - is_unbalance flag for imbalanced data
    - min_child_samples adapts to dataset size
    """
    tree = _base_tree_defaults(profile)

    # LightGBM uses num_leaves instead of max_depth
    # Rule of thumb: num_leaves ≤ 2^max_depth
    max_depth = tree.pop("max_depth")
    num_leaves = min(2**max_depth - 1, 63)  # cap at 63 to prevent overfitting

    config: dict[str, Any] = {
        "n_estimators": tree["n_estimators"],
        "num_leaves": num_leaves,
        "learning_rate": 0.05 if tree["n_estimators"] >= 300 else 0.1,
        "subsample": tree["subsample"],
        "random_state": tree["random_state"],
    }

    # Min samples per leaf — higher for large datasets
    if profile.is_large:
        config["min_child_samples"] = 50
    elif profile.is_small:
        config["min_child_samples"] = 10
    else:
        config["min_child_samples"] = 20

    # Column subsampling
    n_feat = _n_features(profile)
    config["colsample_bytree"] = 0.8 if n_feat > 20 else 1.0

    # Regularisation
    if n_feat > 50:
        config["reg_alpha"] = 0.1
        config["reg_lambda"] = 0.1
    else:
        config["reg_alpha"] = 0.0
        config["reg_lambda"] = 0.0

    # Imbalance handling — LightGBM uses is_unbalance flag
    if profile.is_imbalanced:
        config["is_unbalance"] = True

    # Suppress LightGBM verbosity
    config["verbose"] = -1

    return config


# ─────────────────────────────────────────────────────────────
# Person B's models — regressors
# ─────────────────────────────────────────────────────────────


def _random_forest_reg_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for Random Forest regressor.

    Key adaptations:
    - No subsampling param (RF bootstraps by default)
    - max_features tuned for regression (fewer = less correlation)
    - min_samples_split adapts to data size
    """
    tree = _base_tree_defaults(profile)

    # RF doesn't use subsample — it bootstraps
    tree.pop("subsample", None)

    config: dict[str, Any] = {
        "n_estimators": tree["n_estimators"],
        "max_depth": tree["max_depth"],
        "random_state": tree["random_state"],
    }

    # Max features: sqrt for many features, all for few
    n_feat = _n_features(profile)
    if n_feat > 10:
        config["max_features"] = "sqrt"
    else:
        config["max_features"] = 1.0  # use all features

    # Min samples to split — prevent overfitting on small data
    if profile.is_small:
        config["min_samples_split"] = 5
        config["min_samples_leaf"] = 2
    elif profile.is_large:
        config["min_samples_split"] = 10
        config["min_samples_leaf"] = 4
    else:
        config["min_samples_split"] = 5
        config["min_samples_leaf"] = 2

    # Bootstrap is True by default in sklearn, but be explicit
    config["bootstrap"] = True

    return config


def _xgboost_reg_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for XGBoost regressor.

    Similar to XGBoost classifier but with regression-specific
    eval_metric and no class weighting.
    """
    config = _base_tree_defaults(profile)

    # Learning rate
    config["learning_rate"] = 0.05 if config["n_estimators"] >= 300 else 0.1

    # Regularisation
    n_feat = _n_features(profile)
    if n_feat > 50:
        config["reg_alpha"] = 0.1
        config["reg_lambda"] = 1.0
    else:
        config["reg_alpha"] = 0.0
        config["reg_lambda"] = 1.0

    # Min child weight
    config["min_child_weight"] = 5 if profile.is_large else 1

    # Column subsampling
    config["colsample_bytree"] = 0.8 if n_feat > 20 else 1.0

    # Regression-specific eval metric
    config["eval_metric"] = "rmse"

    return config


# ─────────────────────────────────────────────────────────────
# Person A's models — we provide defaults for completeness
# ─────────────────────────────────────────────────────────────


def _logistic_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for Logistic Regression.

    Key adaptations:
    - Regularisation strength (C) based on feature count
    - class_weight='balanced' for imbalanced data
    - solver chosen by dataset size
    """
    n_feat = _n_features(profile)

    # Regularisation: more features → stronger regularisation (lower C)
    if n_feat > 50:
        c_value = 0.1
    elif n_feat > 20:
        c_value = 1.0
    else:
        c_value = 10.0

    # Solver: 'lbfgs' is default, 'saga' scales better for large data
    if profile.is_large:
        solver = "saga"
        max_iter = 500
    else:
        solver = "lbfgs"
        max_iter = 200

    config: dict[str, Any] = {
        "C": c_value,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": 42,
    }

    # Imbalance handling
    if profile.is_imbalanced:
        config["class_weight"] = "balanced"

    return config


def _random_forest_cls_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for Random Forest classifier (Person A's).

    Same base as RF regressor but with class_weight for imbalanced data
    and classification-appropriate max_features default.
    """
    tree = _base_tree_defaults(profile)
    tree.pop("subsample", None)

    config: dict[str, Any] = {
        "n_estimators": tree["n_estimators"],
        "max_depth": tree["max_depth"],
        "random_state": tree["random_state"],
    }

    n_feat = _n_features(profile)
    config["max_features"] = "sqrt" if n_feat > 10 else "log2"

    if profile.is_small:
        config["min_samples_split"] = 5
        config["min_samples_leaf"] = 2
    elif profile.is_large:
        config["min_samples_split"] = 10
        config["min_samples_leaf"] = 4
    else:
        config["min_samples_split"] = 5
        config["min_samples_leaf"] = 2

    config["bootstrap"] = True

    # Classification-specific: handle imbalance
    if profile.is_imbalanced:
        config["class_weight"] = "balanced"

    return config


def _linear_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for Linear Regression.

    Linear regression has almost no hyperparameters.
    We just set fit_intercept based on data characteristics.
    """
    return {
        "fit_intercept": True,
    }


def _ridge_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for Ridge Regression.

    Key adaptation: alpha (regularisation strength) based on
    feature count and multicollinearity risk.
    """
    n_feat = _n_features(profile)

    # More features or potential multicollinearity → stronger regularisation
    if n_feat > 50:
        alpha = 10.0
    elif n_feat > 20:
        alpha = 1.0
    else:
        alpha = 0.1

    return {
        "alpha": alpha,
        "fit_intercept": True,
        "random_state": 42,
    }


def _lasso_defaults(profile: DataProfile) -> dict[str, Any]:
    """Defaults for Lasso Regression.

    Key adaptation: alpha tuned for feature selection.
    Lasso drives some coefficients to zero (L1), so we use it
    when there are many features.
    """
    n_feat = _n_features(profile)

    # Lasso is naturally feature-selective; stronger alpha for more features
    if n_feat > 50:
        alpha = 1.0
    elif n_feat > 20:
        alpha = 0.1
    else:
        alpha = 0.01

    return {
        "alpha": alpha,
        "fit_intercept": True,
        "max_iter": 2000,
        "random_state": 42,
    }
