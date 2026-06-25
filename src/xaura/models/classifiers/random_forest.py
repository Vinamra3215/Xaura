"""Random Forest Classifier model wrapper.

An ensemble of decision trees that votes on predictions. Works well
on most tabular datasets with minimal tuning. Supports dataset-aware
defaults:
    - Large datasets → cap max_depth at 20 (prevents overfitting)
    - Imbalanced data → class_weight="balanced_subsample"
    - Small datasets → fewer trees (50 instead of 100)
"""

from __future__ import annotations

from typing import Any

from xaura.models.base import BaseModel
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class RandomForestClassifierModel(BaseModel):
    """Random Forest Classifier with dataset-aware defaults."""

    name = "rf_classifier"
    display_name = "Random Forest Classifier"
    task_type = "classification"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for Random Forest.

        Adjustments based on profile:
            - Large (>100k rows) → max_depth=20 (prevent overfitting)
            - Imbalanced → class_weight="balanced_subsample"
            - Small (<1000 rows) → fewer trees (50) to avoid overfitting
        """
        config: dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1,
        }

        # Large dataset → limit tree depth
        if profile.is_large:
            config["max_depth"] = 20

        # Imbalanced → bootstrap balanced subsamples
        if profile.is_imbalanced:
            config["class_weight"] = "balanced_subsample"

        # Small dataset → fewer trees
        if profile.is_small:
            config["n_estimators"] = 50

        return config

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn RandomForestClassifier with the given config."""
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**config)
