from typing import Any

from xaura.models.base import BaseModel
from xaura.models.registry import register_model

try:
    from lightgbm import LGBMRegressor

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


@register_model
class LightGBMRegressorModel(BaseModel):
    """Wrapper for LightGBM Regressor."""

    name = "lightgbm_reg"
    display_name = "LightGBM Regressor"
    task_type = "regression"

    def __init__(self, **kwargs: Any) -> None:
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed.")
        # Default parameters for robust regression
        self.config = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "random_state": kwargs.get("random_state", 42),
        }
        self.model = LGBMRegressor(**self.config)

    def get_default_config(self, profile) -> dict[str, Any]:
        """Return default config."""
        return {}

    def build(self, config: dict[str, Any], profile) -> Any:
        """Build the model."""
        if not _LGBM_AVAILABLE:
            raise ImportError("lightgbm is not installed.")
        # Ensure learning_rate is not passed multiple times
        conf = dict(self.config)
        conf.update(config)
        return LGBMRegressor(**conf)

    def fit(self, X: Any, y: Any) -> None:
        """Fit the LightGBM Regressor to the data."""
        self.model.fit(X, y)

    def predict(self, X: Any) -> Any:
        """Predict using the fitted model."""
        return self.model.predict(X)

    def get_feature_importances(self) -> dict[str, float]:
        """Return feature importances if available."""
        if hasattr(self.model, "feature_importances_") and hasattr(self.model, "feature_name_"):
            importances = self.model.feature_importances_
            # feature_name_ might not exist if X was a numpy array without names
            # but usually LightGBM creates them as Column_0, etc.
            names = getattr(self.model, "feature_name_", [f"f{i}" for i in range(len(importances))])
            return {str(name): float(imp) for name, imp in zip(names, importances, strict=False)}
        return {}
