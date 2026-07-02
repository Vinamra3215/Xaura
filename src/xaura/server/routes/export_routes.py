"""Export routes — download ZIP bundles, plots, and CSV logs.

Endpoints:
    GET  /api/export/{run_id}/zip     Download a run as a ZIP bundle
    GET  /api/export/{run_id}/plots   Download all plots as a ZIP
    GET  /api/export/log/csv          Download the full experiment log as CSV
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from xaura.export.csv_export import export_log_csv
from xaura.store.sqlite_store import get_run

router = APIRouter(prefix="/api/export", tags=["export"])


# ---------------------------------------------------------------------------
# GET /api/export/log/csv — download full experiment log
# ---------------------------------------------------------------------------
# NOTE: This must come BEFORE /{run_id} routes to avoid FastAPI treating
# "log" as a run_id.


@router.get("/log/csv")
async def export_csv_log():
    """Download the full experiment log as a CSV file.

    Returns:
        CSV file download.

    Raises:
        404: If no experiment runs exist yet.
    """
    try:
        # Use a temp file that the server cleans up after download
        tmp = Path(tempfile.mkdtemp()) / "xaura_experiment_log.csv"
        export_log_csv(tmp)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FileResponse(
        path=str(tmp),
        filename="xaura_experiment_log.csv",
        media_type="text/csv",
    )


# ---------------------------------------------------------------------------
# GET /api/export/{run_id}/zip — download a run as a ZIP bundle
# ---------------------------------------------------------------------------


@router.get("/{run_id}/zip")
async def export_zip_bundle(run_id: str, request: Request):
    """Download a model run as a ZIP bundle.

    The ZIP contains config.json, metrics.json, and run_info.json.
    If a Result object is available in the session, plots are included too.

    Returns:
        ZIP file download.

    Raises:
        404: If no run exists with that ID.
    """
    import json
    import zipfile

    run = get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )

    # Build a ZIP in a temp directory
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / f"xaura_{run.get('model_name', 'run')}_{run_id[:8]}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Config
        zf.writestr(
            "config.json",
            json.dumps(run.get("config", {}), indent=2, default=str),
        )
        # Metrics
        zf.writestr(
            "metrics.json",
            json.dumps(run.get("metrics", {}), indent=2, default=str),
        )
        # Full run info
        zf.writestr(
            "run_info.json",
            json.dumps(run, indent=2, default=str),
        )

    return FileResponse(
        path=str(zip_path),
        filename=zip_path.name,
        media_type="application/zip",
    )


# ---------------------------------------------------------------------------
# GET /api/export/{run_id}/plots — download plots as a ZIP
# ---------------------------------------------------------------------------


@router.get("/{run_id}/plots")
async def export_plots_zip(run_id: str, request: Request):
    """Download all plots for a run as a ZIP file.

    Generates plots from the session's Result object if available,
    otherwise returns the stored run metadata.

    Returns:
        ZIP file with PNG plots.

    Raises:
        404: If the run doesn't exist.
        400: If no Result object is available in the session.
    """
    import zipfile

    import matplotlib.pyplot as plt

    run = get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )

    # Check if a Result is available in the session store
    sessions = request.app.state.sessions
    result = None
    for session in sessions.values():
        if hasattr(session, "get") and session.get("result") is not None:
            r = session["result"]
            if r.model_name == run.get("model_name"):
                result = r
                break

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="No active Result object found for this run. "
            "Re-run the model first, then export plots.",
        )

    # Generate plots
    from xaura.export.plot_export import export_plots

    tmp_dir = Path(tempfile.mkdtemp())
    plots_dir = tmp_dir / "plots"
    saved_paths = export_plots(result, plots_dir, fmt="png")
    plt.close("all")

    # Bundle into ZIP
    zip_path = tmp_dir / f"xaura_plots_{run_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, path in saved_paths.items():
            zf.write(path, f"plots/{name}.png")

    return FileResponse(
        path=str(zip_path),
        filename=zip_path.name,
        media_type="application/zip",
    )
