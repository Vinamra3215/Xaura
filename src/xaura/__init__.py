"""XAURA — eXtendable Automated Unified Research & Analytics."""

__version__ = "0.1.0"

from xaura.profiler import DataProfile, profile
from xaura.models import run_model

__all__ = ["DataProfile", "profile", "run_model", "__version__"]
