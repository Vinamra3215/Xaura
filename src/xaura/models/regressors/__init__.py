"""XAURA Regressors — regression model wrappers.

Importing this module triggers @register_model decorators,
adding all regressors to the global model registry.
"""

from xaura.models.regressors.linear import LinearRegressionModel
from xaura.models.regressors.ridge_lasso import LassoModel, RidgeModel

__all__ = ["LinearRegressionModel", "RidgeModel", "LassoModel"]
