"""Random Forest Regressor — ensemble tree regressor wrapper.

Wraps sklearn.ensemble.RandomForestRegressor with XAURA's BaseModel
interface. Uses dataset-aware defaults from defaults.py via
get_default_config().

Random Forest fits many decision trees on random subsets of the data
and averages their predictions. This reduces variance (overfitting)
compared to a single tree, while keeping bias low.

For regression, RF defaults use:
- max_features="sqrt" for many features (decorrelates trees)
- bootstrap=True (each tree sees a different sample)
- No class weighting (regression has no classes)

Usage:
    from xaura import run_model, profile

    dp = profile(df)
    result = run_model("random_forest_reg", df, dp)
    print(result.metrics)  # {"mse": ..., "rmse": ..., "mae": ..., "r2": ...}
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor as _RFRegressor

from xaura.models.base import BaseModel
from xaura.models.defaults import get_defaults
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class RandomForestRegressor(BaseModel):
    """XAURA wrapper for Random Forest regression.

    Inherits from BaseModel — only implements build() and
    get_default_config(). BaseModel.run() handles splitting,
    encoding, fitting, predicting, and computing regression
    metrics (mse, rmse, mae, r2).

    Class attributes:
        name: "random_forest_reg" — matches defaults engine and registry.
        display_name: "Random Forest Regressor" — for UI display.
        task_type: "regression" — tells BaseModel to use regression metrics.
    """

    name = "random_forest_reg"
    display_name = "Random Forest Regressor"
    task_type = "regression"

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a RandomForestRegressor with the given config.

        Args:
            config: Hyperparameters dict (merged defaults + user overrides).
            profile: DataProfile of the dataset.

        Returns:
            A sklearn RandomForestRegressor instance (unfitted).
        """
        return _RFRegressor(**config)

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for RF regression.

        Delegates to the central defaults engine which inspects the
        DataProfile (shape, feature count) to pick smart hyperparameters.

        Args:
            profile: DataProfile of the dataset.

        Returns:
            Dict of RandomForestRegressor hyperparameters.
        """
        return get_defaults(profile, self.name)
