"""Programmatic launcher for the XAURA web UI."""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xaura.models.base import Result

logger = logging.getLogger(__name__)


def _open_browser(url: str, delay: float = 1.5):
    """Wait for the server to start, then open the browser."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception as exc:
        logger.warning(f"Could not open browser automatically: {exc}")


def show_ui(result: Result | None = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the XAURA FastAPI web server and open it in the default browser.

    If a Result object is provided and has a run_id, the browser will navigate
    directly to the specific experiment's result page. Otherwise, it defaults
    to the experiments log.

    This function blocks the main thread to keep the server running.

    Args:
        result: Optional Result object from xaura.run_model().
        host: Host to bind the server to.
        port: Port to run the server on.
    """
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "uvicorn is required to run the UI. Install with: pip install uvicorn[standard]"
        ) from None

    base_url = f"http://{host}:{port}"
    target_url = f"{base_url}/experiments"

    if result is not None and getattr(result, "run_id", None):
        target_url = f"{base_url}/results/{result.run_id}"

        try:
            # Generate charts for the live results page
            import contextlib

            from xaura.server.app import app
            from xaura.visualisation.plotly_charts import (
                confusion_matrix_chart,
                feature_importance_chart,
                precision_recall_chart,
                roc_curve_chart,
            )
            from xaura.visualisation.plotly_clustering import cluster_scatter_pca, silhouette_plot
            from xaura.visualisation.plotly_regression import (
                predicted_vs_actual,
                qq_plot,
                residual_distribution,
                residuals_vs_fitted,
            )

            charts = {}
            if result.task_type == "classification":
                for key, func in [
                    ("confusion_matrix", confusion_matrix_chart),
                    ("roc_curve", roc_curve_chart),
                    ("precision_recall", precision_recall_chart),
                    ("feature_importance", feature_importance_chart),
                ]:
                    with contextlib.suppress(Exception):
                        charts[key] = func(result).to_json()
            elif result.task_type == "regression":
                for key, func in [
                    ("residuals_vs_fitted", residuals_vs_fitted),
                    ("qq_plot", qq_plot),
                    ("predicted_vs_actual", predicted_vs_actual),
                    ("residual_distribution", residual_distribution),
                ]:
                    with contextlib.suppress(Exception):
                        charts[key] = func(result).to_json()
            elif result.task_type == "clustering":
                for key, func in [
                    ("cluster_scatter", cluster_scatter_pca),
                    ("silhouette", silhouette_plot),
                ]:
                    with contextlib.suppress(Exception):
                        charts[key] = func(result).to_json()

            # Clean up missing charts
            charts = {k: v for k, v in charts.items() if v is not None}

            # Populate the session so the results page can load it
            app.state.sessions[result.run_id] = {
                "result": result,
                "charts": charts,
            }
        except Exception as exc:
            logger.warning(f"Could not generate charts for programmatic UI: {exc}")

    print(f"\n🚀 Launching XAURA UI at: {target_url}\n")
    print("Press Ctrl+C to stop the server and return to your script.\n")

    # Start the browser-opener in a daemon thread so it doesn't block server startup
    threading.Thread(target=_open_browser, args=(target_url,), daemon=True).start()

    # Start the server (this blocks the main thread until the user stops it)
    uvicorn.run("xaura.server.app:app", host=host, port=port, reload=False, log_level="warning")
