"""LightGBM Classifier — light gradient-boosted tree classifier wrapper.

Wraps lightgbm.LGBMClassifier with XAURA's BaseModel interface.
Uses dataset-aware defaults from defaults.py via get_default_config().

LightGBM is Microsoft's gradient boosting framework. Compared to XGBoost:
- Faster training (leaf-wise growth vs level-wise)
- Lower memory usage (histogram-based splitting)
- Native categorical feature support
- Uses num_leaves instead of max_depth for complexity control

Usage:
    from xaura import run_model, profile

    dp = profile(df)
    result = run_model("lightgbm_cls", df, dp)
    print(result.metrics)
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier as _LGBMClassifier

from xaura.models.base import BaseModel
from xaura.models.defaults import get_defaults
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class LightGBMClassifier(BaseModel):
    """XAURA wrapper for LightGBM classification.

    Inherits from BaseModel — only implements build() and
    get_default_config(). Everything else (split, fit, predict,
    evaluate, Result) is handled by BaseModel.run().

    Key differences from XGBoost:
        - Uses num_leaves (not max_depth) for tree complexity
        - is_unbalance flag instead of scale_pos_weight
        - verbose=-1 to suppress LightGBM training output
        - Leaf-wise growth gives different tree shapes

    Class attributes:
        name: "lightgbm_cls" — matches defaults engine and registry.
        display_name: "LightGBM Classifier" — for UI display.
        task_type: "classification".
    """

    name = "lightgbm_cls"
    display_name = "LightGBM Classifier"
    task_type = "classification"

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build an LGBMClassifier with the given config.

        Args:
            config: Hyperparameters dict (merged defaults + user overrides).
            profile: DataProfile of the dataset (unused here, but part
                     of the BaseModel interface for models that need it).

        Returns:
            A lightgbm.LGBMClassifier instance (unfitted).
        """
        return _LGBMClassifier(**config)

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for LightGBM classification.

        Delegates to the central defaults engine which inspects the
        DataProfile (shape, imbalance, feature count) to pick smart
        hyperparameters. Includes verbose=-1 by default.

        Args:
            profile: DataProfile of the dataset.

        Returns:
            Dict of LightGBM hyperparameters.
        """
        return get_defaults(profile, self.name)
