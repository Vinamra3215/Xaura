"""Profile routes - upload CSV and view dataset profile.

Endpoints:
    POST /api/profile     Upload a CSV, run profile(), return session ID + profile JSON.
    GET  /profile/{id}    Render the profile page for a given session.
"""

from __future__ import annotations

import uuid
from io import StringIO
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from xaura import profile as profile_data

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# POST /api/profile — Upload CSV and profile it
# ---------------------------------------------------------------------------


@router.post("/api/profile")
async def upload_and_profile(request: Request, file: Annotated[UploadFile, File()]):
    """Accept a CSV upload, profile it, store in session, return JSON.

    Returns:
        JSON with session_id and profile summary.
    """
    # Validate file type
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Read CSV
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from None

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    # Profile the dataset
    p = profile_data(df)

    # Generate session ID and store in app state
    session_id = uuid.uuid4().hex[:12]
    request.app.state.sessions[session_id] = {
        "df": df,
        "profile": p,
        "filename": file.filename,
        "result": None,
    }

    # Build response with profile summary
    feature_summary = {}
    for feat_type, columns in p.feature_types.items():
        if columns:
            feature_summary[feat_type] = len(columns)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "profile": {
            "n_rows": p.n_rows,
            "n_cols": p.n_cols,
            "shape": list(p.shape),
            "feature_types": feature_summary,
            "target_column": p.target_column,
            "task_type": p.task_type,
            "is_imbalanced": bool(p.is_imbalanced),
            "has_missing": bool(p.has_missing),
            "missing_fraction": float(p.missing_fraction),
            "warnings": p.warnings,
            "dataset_hash": p.dataset_hash,
        },
    }


# ---------------------------------------------------------------------------
# POST /api/profile/update_target — Dynamically update task type
# ---------------------------------------------------------------------------


@router.post("/api/profile/update_target")
async def update_target(request: Request):
    """Update the target column in the session and return the newly inferred task type."""
    body = await request.json()
    session_id = body.get("session_id")
    target_col = body.get("target_col")

    sessions = request.app.state.sessions
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    df = session["df"]
    profile = session["profile"]

    # Update target
    profile.target_column = target_col

    # Re-infer task type
    from xaura.profiler.profiler import _infer_task_type

    new_task_type = _infer_task_type(df, target_col)
    profile.task_type = new_task_type

    return {"task_type": new_task_type}


# ---------------------------------------------------------------------------
# GET /profile/{session_id} — Render profile page
# ---------------------------------------------------------------------------


@router.get("/profile/{session_id}", response_class=HTMLResponse)
async def profile_page(request: Request, session_id: str):
    """Render the profile HTML page for a given session."""
    sessions = request.app.state.sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    p = session["profile"]
    df = session["df"]

    # Build column details for the table
    column_details = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = int(df[col].nunique())
        column_details.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 1),
                "unique": unique_count,
            }
        )

    # Feature type summary
    feature_summary = {}
    for feat_type, columns in p.feature_types.items():
        if columns:
            feature_summary[feat_type] = len(columns)

    return _templates.TemplateResponse(
        request,
        "profile.html",
        {
            "session_id": session_id,
            "filename": session["filename"],
            "n_rows": p.n_rows,
            "n_cols": p.n_cols,
            "feature_types": feature_summary,
            "target_column": p.target_column,
            "task_type": p.task_type,
            "is_imbalanced": bool(p.is_imbalanced),
            "has_missing": bool(p.has_missing),
            "missing_pct": round(float(p.missing_fraction) * 100, 1),
            "warnings": p.warnings,
            "columns": column_details,
            "dataset_hash": p.dataset_hash[:16],
        },
    )
