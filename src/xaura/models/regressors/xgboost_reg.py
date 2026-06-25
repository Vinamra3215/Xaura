"""XGBoost Regressor — gradient-boosted tree regressor wrapper.

Wraps xgboost.XGBRegressor with XAURA's BaseModel interface.
Uses dataset-aware defaults from defaults.py via get_default_config().

XGBoost regression works identically to XGBoost classification under
the hood — it builds trees sequentially to correct errors — but the
loss function is squared error instead of log-loss, and the output
is a continuous value instead of a class label.

For regression, XGBoost defaults use:
- eval_metric="rmse" (root mean squared error)
- No scale_pos_weight (no class imbalance in regression)
- Same regularisation/subsampling logic as classification

Usage:
    from xaura import run_model, profile

    dp = profile(df)
    result = run_model("xgboost_reg", df, dp)
    print(result.metrics)  # {"mse": ..., "rmse": ..., "mae": ..., "r2": ...}
"""

from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor as _XGBRegressor

from xaura.models.base import BaseModel
from xaura.models.defaults import get_defaults
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class XGBoostRegressor(BaseModel):
    """XAURA wrapper for XGBoost regression.

    Inherits from BaseModel — only implements build() and
    get_default_config(). BaseModel.run() handles splitting,
    encoding, fitting, predicting, and computing regression
    metrics (mse, rmse, mae, r2).

    Class attributes:
        name: "xgboost_reg" — matches defaults engine and registry.
        display_name: "XGBoost Regressor" — for UI display.
        task_type: "regression" — tells BaseModel to use regression metrics.
    """

    name = "xgboost_reg"
    display_name = "XGBoost Regressor"
    task_type = "regression"

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build an XGBRegressor with the given config.

        Args:
            config: Hyperparameters dict (merged defaults + user overrides).
            profile: DataProfile of the dataset.

        Returns:
            An xgboost.XGBRegressor instance (unfitted).
        """
        return _XGBRegressor(**config)

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for XGBoost regression.

        Delegates to the central defaults engine which inspects the
        DataProfile (shape, feature count) to pick smart hyperparameters.

        Args:
            profile: DataProfile of the dataset.

        Returns:
            Dict of XGBRegressor hyperparameters.
        """
        return get_defaults(profile, self.name)
