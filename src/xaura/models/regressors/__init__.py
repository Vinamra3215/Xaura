"""XAURA Regressors — regression model wrappers.

Importing this module triggers @register_model decorators,
adding all regressors to the global model registry.
"""

from xaura.models.regressors.lightgbm_reg import LightGBMRegressorModel
from xaura.models.regressors.linear import LinearRegressionModel
from xaura.models.regressors.random_forest_reg import RandomForestRegressor
from xaura.models.regressors.ridge_lasso import LassoModel, RidgeModel
from xaura.models.regressors.xgboost_reg import XGBoostRegressor

__all__ = [
    "LinearRegressionModel",
    "RidgeModel",
    "LassoModel",
    "RandomForestRegressor",
    "XGBoostRegressor",
    "LightGBMRegressorModel",
]
