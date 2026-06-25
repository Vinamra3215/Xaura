"""XGBoost Classifier — gradient-boosted tree classifier wrapper.

Wraps xgboost.XGBClassifier with XAURA's BaseModel interface.
Uses dataset-aware defaults from defaults.py via get_default_config().

XGBoost builds trees one at a time, where each new tree corrects
the errors of the previous ones. It's one of the best models for
tabular data — fast, accurate, handles imbalanced data well.

Usage:
    from xaura import run_model, profile

    dp = profile(df)
    result = run_model("xgboost_cls", df, dp)
    print(result.metrics)
"""

from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier as _XGBClassifier

from xaura.models.base import BaseModel
from xaura.models.defaults import get_defaults
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class XGBoostClassifier(BaseModel):
    """XAURA wrapper for XGBoost classification.

    Inherits from BaseModel — only implements build() and
    get_default_config(). Everything else (split, fit, predict,
    evaluate, Result) is handled by BaseModel.run().

    Features:
        - Dataset-aware defaults via defaults.py engine
        - Handles imbalanced data (scale_pos_weight)
        - L1/L2 regularisation tuned to feature count
        - Column subsampling for high-dimensional data

    Class attributes:
        name: "xgboost_cls" — matches defaults engine and registry.
        display_name: "XGBoost Classifier" — for UI display.
        task_type: "classification".
    """

    name = "xgboost_cls"
    display_name = "XGBoost Classifier"
    task_type = "classification"

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build an XGBClassifier with the given config.

        Args:
            config: Hyperparameters dict (merged defaults + user overrides).
            profile: DataProfile of the dataset (unused here, but part
                     of the BaseModel interface for models that need it).

        Returns:
            An xgboost.XGBClassifier instance (unfitted).
        """
        return _XGBClassifier(**config)

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for XGBoost classification.

        Delegates to the central defaults engine which inspects the
        DataProfile (shape, imbalance, feature count) to pick smart
        hyperparameters.

        Args:
            profile: DataProfile of the dataset.

        Returns:
            Dict of XGBoost hyperparameters.
        """
        return get_defaults(profile, self.name)
