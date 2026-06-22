"""Linear Regression model wrapper.

The simplest regression model — fits a straight line (or hyperplane)
through the data by minimising the sum of squared residuals. No
regularisation, so it works best when:
    - Features are not highly correlated (no multicollinearity)
    - The relationship between features and target is roughly linear
    - You have more samples than features

Dataset-aware defaults are minimal for Linear Regression since it
has very few hyperparameters.
"""

from __future__ import annotations

from typing import Any

from xaura.models.base import BaseModel
from xaura.models.registry import register_model
from xaura.profiler.dataprofile import DataProfile


@register_model
class LinearRegressionModel(BaseModel):
    """Linear Regression with dataset-aware defaults."""

    name = "linear_regression"
    display_name = "Linear Regression"
    task_type = "regression"

    def get_default_config(self, profile: DataProfile) -> dict[str, Any]:
        """Return default config for Linear Regression.

        Linear Regression has very few tuneable parameters.
        We always fit an intercept unless the user overrides.

        Args:
            profile: Dataset profile (used for consistency, minimal
                     adjustments needed for OLS).

        Returns:
            Config dict for sklearn LinearRegression.
        """
        return {
            "fit_intercept": True,
        }

    def build(self, config: dict[str, Any], profile: DataProfile) -> Any:
        """Build a sklearn LinearRegression with the given config."""
        from sklearn.linear_model import LinearRegression

        return LinearRegression(**config)
