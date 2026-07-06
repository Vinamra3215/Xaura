"""XAURA Models — model wrappers and registry."""

# Import to trigger @register_model decorators
import xaura.models.classifiers  # noqa: F401
import xaura.models.clusterers  # noqa: F401
import xaura.models.regressors  # noqa: F401
from xaura.models.base import BaseModel, Result
from xaura.models.registry import list_models, register_model, run_model

__all__ = ["BaseModel", "Result", "run_model", "list_models", "register_model"]
