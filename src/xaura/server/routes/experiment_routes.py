"""Experiment routes — browse, compare, and manage experiment runs.

Endpoints:
    GET  /api/experiments          List all runs (with optional filters)
    GET  /api/experiments/{id}     Get a single run
    DELETE /api/experiments/{id}   Delete a run
    GET  /api/experiments/compare  Compare multiple runs side-by-side
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from xaura.store.sqlite_store import (
    delete_run,
    get_metrics_comparison,
    get_run,
    list_runs,
)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# ---------------------------------------------------------------------------
# GET /api/experiments — list all runs
# ---------------------------------------------------------------------------


@router.get("")
async def list_experiments(
    model_name: str | None = Query(None, description="Filter by model name"),
    task_type: str | None = Query(None, description="Filter by task type"),
    dataset_name: str | None = Query(None, description="Filter by dataset name"),
):
    """Return all experiment runs, optionally filtered.

    Query parameters are optional. If none are provided, all runs
    are returned sorted by created_at (newest first).

    Returns:
        JSON array of experiment run dicts.
    """
    filters: dict[str, str] = {}
    if model_name:
        filters["model_name"] = model_name
    if task_type:
        filters["task_type"] = task_type
    if dataset_name:
        filters["dataset_name"] = dataset_name

    runs = list_runs(filters=filters if filters else None)
    return {"runs": runs, "count": len(runs)}


# ---------------------------------------------------------------------------
# GET /api/experiments/compare — compare multiple runs
# ---------------------------------------------------------------------------
# NOTE: This must come BEFORE the /{run_id} route, otherwise FastAPI
# would interpret "compare" as a run_id.


@router.get("/compare")
async def compare_experiments(
    ids: str = Query(
        ...,
        description="Comma-separated run IDs to compare",
        examples=["id1,id2,id3"],
    ),
):
    """Compare multiple experiment runs side-by-side.

    Pass run IDs as a comma-separated string in the `ids` query param.

    Returns:
        JSON array with comparison-relevant fields for each run.

    Raises:
        400: If no IDs are provided.
        404: If none of the provided IDs match any runs.
    """
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]

    if not run_ids:
        raise HTTPException(status_code=400, detail="No run IDs provided")

    results = get_metrics_comparison(run_ids)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No runs found for the provided IDs",
        )

    return {"runs": results, "count": len(results)}


# ---------------------------------------------------------------------------
# GET /api/experiments/{run_id} — get a single run
# ---------------------------------------------------------------------------


@router.get("/{run_id}")
async def get_experiment(run_id: str):
    """Fetch a single experiment run by its UUID.

    Returns:
        Full run dict including config, metrics, and profile summary.

    Raises:
        404: If no run exists with that ID.
    """
    run = get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )
    return run


# ---------------------------------------------------------------------------
# DELETE /api/experiments/{run_id} — delete a run
# ---------------------------------------------------------------------------


@router.delete("/{run_id}")
async def delete_experiment(run_id: str):
    """Delete an experiment run by its UUID.

    Returns:
        Confirmation message.

    Raises:
        404: If no run exists with that ID.
    """
    deleted = delete_run(run_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found",
        )
    return {"message": f"Run '{run_id}' deleted", "id": run_id}
