"""XAURA — eXtendable Automated Unified Research & Analytics."""

__version__ = "0.1.0"

from xaura.profiler import profile, DataProfile
from xaura.models import run_model

__all__ = ["profile", "DataProfile", "run_model", "__version__"]
