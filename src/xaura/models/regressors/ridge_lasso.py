"""Ridge and Lasso regression model wrappers.

Two regularised linear regression models in one file — they share
the same structure but differ in their penalty type:

    - **Ridge (L2):** Shrinks coefficients toward zero but keeps all
      features. Good when many features are mildly correlated.
    - **Lasso (L1):** Can shrink coefficients to exactly zero, effectively
      performing feature selection. Good when you suspect only a few
      features matter.

Dataset-aware defaults:
    - Ridge: stronger alpha when many features (> 50)
    - Lasso: weaker alpha when few features (< 10), higher max_iter
"""

from __future__ import annotations

from typing import Any

from xaura.models.base import BaseModel
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class RidgeModel(BaseModel):
    """Ridge Regression (L2 penalty) with dataset-aware defaults."""

    name = "ridge"
    display_name = "Ridge Regression"
    task_type = "regression"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for Ridge Regression.

        Adjustments based on profile:
            - Many features (> 50) → stronger regularisation (alpha=10.0)
              to prevent overfitting on high-dimensional data.
            - Large dataset → standard alpha is fine.

        Args:
            profile: Dataset profile for adaptive tuning.

        Returns:
            Config dict for sklearn Ridge.
        """
        config: dict[str, Any] = {
            "alpha": 1.0,
            "fit_intercept": True,
            "random_state": 42,
        }

        # Many features → stronger regularisation
        if profile.n_cols > 50:
            config["alpha"] = 10.0

        return config

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn Ridge with the given config."""
        from sklearn.linear_model import Ridge

        return Ridge(**config)


@register_model
class LassoModel(BaseModel):
    """Lasso Regression (L1 penalty) with dataset-aware defaults."""

    name = "lasso"
    display_name = "Lasso Regression"
    task_type = "regression"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return dataset-aware defaults for Lasso Regression.

        Adjustments based on profile:
            - Few features (< 10) → weaker regularisation (alpha=0.1)
              so Lasso doesn't zero out too many coefficients.
            - Many features (> 50) → keep alpha=1.0 to encourage sparsity.
            - Always uses a high max_iter since Lasso's coordinate descent
              can be slow to converge.

        Args:
            profile: Dataset profile for adaptive tuning.

        Returns:
            Config dict for sklearn Lasso.
        """
        config: dict[str, Any] = {
            "alpha": 1.0,
            "fit_intercept": True,
            "max_iter": 5000,
            "random_state": 42,
        }

        # Few features → weaker regularisation
        if profile.n_cols < 10:
            config["alpha"] = 0.1

        return config

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn Lasso with the given config."""
        from sklearn.linear_model import Lasso

        return Lasso(**config)
