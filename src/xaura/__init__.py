"""XAURA — eXtendable Automated Unified Research & Analytics."""

__version__ = "0.1.0"

from xaura.launcher import show_ui
from xaura.models import run_model
from xaura.profiler import DataProfile, profile

__all__ = ["DataProfile", "profile", "run_model", "show_ui", "__version__"]
