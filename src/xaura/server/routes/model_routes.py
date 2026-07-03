"""Model routes - run models and view results.

Endpoints:
    POST /api/run              Run a model on the uploaded dataset.
    GET  /results/{id}         Render the results page with charts.
    GET  /api/export/{id}      Download the ZIP export bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Register all models so they appear in the registry
import xaura.models.classifiers  # noqa: F401
import xaura.models.clusterers  # noqa: F401
import xaura.models.regressors  # noqa: F401
from xaura import run_model
from xaura.export import export_run
from xaura.visualisation.plotly_charts import (
    confusion_matrix_chart,
    feature_importance_chart,
    precision_recall_chart,
    roc_curve_chart,
)

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# POST /api/run — Train a model on the session dataset
# ---------------------------------------------------------------------------


@router.post("/api/run")
async def run_model_endpoint(request: Request):
    """Run a model on the uploaded dataset and store the result.

    Expects JSON body:
        { "session_id": "...", "model_name": "...", "target_col": "..." }

    Returns:
        JSON with metrics and chart JSONs.
    """
    body = await request.json()
    session_id = body.get("session_id", "")
    model_name = body.get("model_name", "")
    target_col = body.get("target_col", "")

    sessions = request.app.state.sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    df = session["df"]
    profile = session["profile"]

    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required.")

    # Override target column if user specified one
    if target_col and target_col != profile.target_column:
        profile.target_column = target_col

    # Run the model
    try:
        result = run_model(model_name, df, profile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model run failed: {exc}") from None

    # Store result in session
    session["result"] = result

    # Build chart JSONs for classification results
    charts = {}
    if result.task_type == "classification":
        try:
            charts["confusion_matrix"] = confusion_matrix_chart(result).to_json()
        except Exception:
            charts["confusion_matrix"] = None
        try:
            charts["roc_curve"] = roc_curve_chart(result).to_json()
        except Exception:
            charts["roc_curve"] = None
        try:
            charts["precision_recall"] = precision_recall_chart(result).to_json()
        except Exception:
            charts["precision_recall"] = None
        try:
            charts["feature_importance"] = feature_importance_chart(result).to_json()
        except Exception:
            charts["feature_importance"] = None

    # Store charts for the results page
    session["charts"] = charts

    # Build serialisable metrics
    metrics = {}
    for k, v in result.metrics.items():
        try:
            metrics[k] = round(float(v), 4)
        except (TypeError, ValueError):
            metrics[k] = str(v)

    return {
        "session_id": session_id,
        "model_name": result.model_name,
        "task_type": result.task_type,
        "train_time": round(result.train_time_seconds, 2),
        "metrics": metrics,
        "charts": {k: (v is not None) for k, v in charts.items()},
    }


# ---------------------------------------------------------------------------
# GET /results/{session_id} — Render results page
# ---------------------------------------------------------------------------


@router.get("/results/{session_id}", response_class=HTMLResponse)
async def results_page(request: Request, session_id: str):
    """Render the results page with metrics and interactive charts."""
    sessions = request.app.state.sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    result = session.get("result")
    if result is None:
        raise HTTPException(status_code=400, detail="No model has been run yet.")

    charts = session.get("charts", {})

    # Build serialisable metrics
    metrics = {}
    for k, v in result.metrics.items():
        try:
            metrics[k] = round(float(v), 4)
        except (TypeError, ValueError):
            metrics[k] = str(v)

    # Config as JSON string for display
    config_str = json.dumps(result.config, indent=2, default=str)

    return _templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "session_id": session_id,
            "filename": session.get("filename", "dataset"),
            "model_name": result.model_name,
            "task_type": result.task_type,
            "train_time": round(result.train_time_seconds, 2),
            "metrics": metrics,
            "config_str": config_str,
            "charts": charts,
        },
    )


# ---------------------------------------------------------------------------
# GET /api/export/{session_id} — Download ZIP bundle
# ---------------------------------------------------------------------------


@router.get("/api/export/{session_id}")
async def export_bundle(request: Request, session_id: str):
    """Create a ZIP export bundle and return it as a file download."""
    sessions = request.app.state.sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    result = session.get("result")
    profile = session.get("profile")

    if result is None:
        raise HTTPException(status_code=400, detail="No model has been run yet.")

    try:
        zip_path = export_run(result, profile)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from None

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=Path(zip_path).name,
    )
