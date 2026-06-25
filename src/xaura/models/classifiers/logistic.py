"""Logistic Regression model wrapper.

A linear classifier that works well on small-to-medium datasets
with relatively few features. Supports dataset-aware defaults:
    - Imbalanced data → class_weight="balanced"
    - Many features (>50) → L1 penalty with saga solver
    - Small datasets → stronger regularisation (lower C)
"""

from __future__ import annotations

from typing import Any

from xaura.models.base import BaseModel
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier with dataset-aware defaults."""

    name = "logistic_regression"
    display_name = "Logistic Regression"
    task_type = "classification"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for Logistic Regression.

        Adjustments based on profile:
            - Imbalanced → class_weight="balanced" (penalises majority class)
            - Many features (>50) → saga solver + L1 penalty (sparse solution)
            - Small dataset (<1000 rows) → lower C (more regularisation)
        """
        config: dict[str, Any] = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }

        # Imbalanced data → auto-balance class weights
        if profile.is_imbalanced:
            config["class_weight"] = "balanced"

        # Many features → use L1 for feature selection
        if profile.n_cols > 50:
            config["solver"] = "saga"
            config["penalty"] = "l1"

        # Small dataset → stronger regularisation
        if profile.is_small:
            config["C"] = 0.5

        return config

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn LogisticRegression with the given config."""
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**config)
