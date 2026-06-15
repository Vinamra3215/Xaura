"""SQLite Store — experiment tracking persistence layer.

This module handles saving and retrieving ML experiment runs in a local
SQLite database. Every time a model is trained, the run details (model name,
hyperparameters, metrics, etc.) are saved here so you can compare results
across experiments.

SQLite is used because:
- It's built into Python (no extra installs)
- It's a single file on disk (no server to run)
- It's perfect for local, single-user experiment tracking

The database file lives at ~/.xaura/store.db by default.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────
# Default database location
# ─────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = Path.home() / ".xaura" / "store.db"


# ─────────────────────────────────────────────────────────────
# SQL Schema
# ─────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id               TEXT PRIMARY KEY,
    model_name       TEXT NOT NULL,
    dataset_name     TEXT DEFAULT '',
    task_type        TEXT DEFAULT '',
    config           TEXT DEFAULT '{}',
    metrics          TEXT DEFAULT '{}',
    profile_summary  TEXT DEFAULT '{}',
    model_path       TEXT DEFAULT '',
    created_at       TEXT NOT NULL,
    duration_seconds REAL DEFAULT 0.0,
    tags             TEXT DEFAULT '[]'
);
"""

# ─────────────────────────────────────────────────────────────
# 1. init_db — Create the database and table
# ─────────────────────────────────────────────────────────────


def init_db(db_path: str | Path | None = None) -> Path:
    """Initialise the SQLite database and create the experiments table.

    If the database file or its parent directories don't exist, they are
    created automatically. If the table already exists, this is a no-op
    (safe to call multiple times).

    How it works:
    1. Resolves the database path (uses ~/.xaura/store.db if none given)
    2. Creates the parent directory if needed (mkdir -p equivalent)
    3. Connects to SQLite (creates the .db file if it doesn't exist)
    4. Runs CREATE TABLE IF NOT EXISTS (creates table only on first run)

    Args:
        db_path: Path to the database file. Defaults to ~/.xaura/store.db.

    Returns:
        The resolved Path to the database file.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()

    return path


# ─────────────────────────────────────────────────────────────
# 2. create_run — Save a new experiment
# ─────────────────────────────────────────────────────────────


def create_run(run_data: dict[str, Any], db_path: str | Path | None = None) -> str:
    """Save a new experiment run to the database.

    Generates a unique UUID for the run, timestamps it, serialises any
    dict/list fields as JSON strings, and inserts a row.

    How it works:
    1. Generates a UUID4 (random unique ID like "a3f8c2d1-...")
    2. Gets the current UTC time as an ISO string
    3. Converts dict fields (config, metrics, profile_summary, tags)
       to JSON strings — because SQLite only stores text, not dicts
    4. Inserts the row into the experiments table

    Args:
        run_data: Dictionary with experiment details. Expected keys:
            - model_name (str, required): e.g. "random_forest"
            - dataset_name (str, optional): e.g. "iris.csv"
            - task_type (str, optional): "classification"/"regression"/"clustering"
            - config (dict, optional): hyperparameters used
            - metrics (dict, optional): results like {"accuracy": 0.89}
            - profile_summary (dict, optional): snapshot of DataProfile
            - model_path (str, optional): path to saved model file
            - duration_seconds (float, optional): training time
            - tags (list, optional): labels like ["baseline", "v1"]
        db_path: Path to the database file.

    Returns:
        The generated UUID string for this run.

    Raises:
        KeyError: If 'model_name' is not in run_data.
    """
    if "model_name" not in run_data:
        raise KeyError("run_data must contain 'model_name'")

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        conn.execute(
            """
            INSERT INTO experiments
                (id, model_name, dataset_name, task_type, config, metrics,
                 profile_summary, model_path, created_at, duration_seconds, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_data["model_name"],
                run_data.get("dataset_name", ""),
                run_data.get("task_type", ""),
                json.dumps(run_data.get("config", {})),
                json.dumps(run_data.get("metrics", {})),
                json.dumps(run_data.get("profile_summary", {})),
                run_data.get("model_path", ""),
                now,
                run_data.get("duration_seconds", 0.0),
                json.dumps(run_data.get("tags", [])),
            ),
        )
        conn.commit()

    return run_id


# ─────────────────────────────────────────────────────────────
# 3. get_run — Fetch a single experiment by ID
# ─────────────────────────────────────────────────────────────


def _row_to_dict(row: tuple, columns: list[str]) -> dict[str, Any]:
    """Convert a raw SQLite row tuple into a clean Python dict.

    SQLite stores everything as text, so we need to:
    - Pair each value with its column name (SQLite returns plain tuples)
    - Deserialise JSON strings back into dicts/lists

    Args:
        row: Tuple of values from SQLite.
        columns: List of column names in the same order.

    Returns:
        Dict with proper Python types (dicts for config/metrics, etc.)
    """
    d = dict(zip(columns, row, strict=False))

    # Deserialise JSON fields back into Python objects
    for json_field in ("config", "metrics", "profile_summary", "tags"):
        if json_field in d and isinstance(d[json_field], str):
            with contextlib.suppress(json.JSONDecodeError):
                d[json_field] = json.loads(d[json_field])

    return d


def get_run(run_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """Fetch a single experiment run by its UUID.

    How it works:
    1. Queries the experiments table with WHERE id = ?
    2. If found, converts the row to a dict with deserialised JSON fields
    3. If not found, returns None (not an error — the ID might be wrong)

    The '?' placeholder prevents SQL injection — never use f-strings
    with SQL queries!

    Args:
        run_id: The UUID string of the run to fetch.
        db_path: Path to the database file.

    Returns:
        Dict with all run details, or None if not found.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM experiments WHERE id = ?", (run_id,))
        row = cursor.fetchone()

    if row is None:
        return None

    columns = row.keys()
    return _row_to_dict(tuple(row), list(columns))


# ─────────────────────────────────────────────────────────────
# 4. list_runs — List all experiments with optional filters
# ─────────────────────────────────────────────────────────────


def list_runs(
    filters: dict[str, str] | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List all experiment runs, optionally filtered.

    How it works:
    1. Starts with "SELECT * FROM experiments"
    2. If filters are provided (e.g. {"model_name": "xgboost"}), adds
       WHERE clauses dynamically
    3. Orders by created_at DESC (newest first)
    4. Converts each row to a dict

    Only allows filtering on safe columns (model_name, task_type,
    dataset_name) to prevent SQL injection via column names.

    Args:
        filters: Optional dict to filter results. Supported keys:
            - model_name: exact match
            - task_type: exact match
            - dataset_name: exact match
        db_path: Path to the database file.

    Returns:
        List of dicts, one per run. Empty list if no matches.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH

    # Only allow filtering on these columns (security)
    allowed_filters = {"model_name", "task_type", "dataset_name"}

    query = "SELECT * FROM experiments"
    params: list[str] = []

    if filters:
        conditions = []
        for key, value in filters.items():
            if key in allowed_filters:
                conditions.append(f"{key} = ?")
                params.append(value)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY created_at DESC"

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    if not rows:
        return []

    columns = list(rows[0].keys())
    return [_row_to_dict(tuple(row), columns) for row in rows]


# ─────────────────────────────────────────────────────────────
# 5. delete_run — Remove an experiment by ID
# ─────────────────────────────────────────────────────────────


def delete_run(run_id: str, db_path: str | Path | None = None) -> bool:
    """Delete an experiment run by its UUID.

    How it works:
    1. Runs DELETE FROM experiments WHERE id = ?
    2. Checks cursor.rowcount — if 1, the row was deleted; if 0, it
       didn't exist
    3. Returns True/False accordingly

    This does NOT delete the saved model file (if any) — only the
    database record. Model file cleanup is the caller's responsibility.

    Args:
        run_id: The UUID string of the run to delete.
        db_path: Path to the database file.

    Returns:
        True if a run was deleted, False if no run with that ID existed.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        cursor = conn.execute("DELETE FROM experiments WHERE id = ?", (run_id,))
        conn.commit()
        deleted = cursor.rowcount > 0

    return deleted


# ─────────────────────────────────────────────────────────────
# 6. get_metrics_comparison — Compare multiple runs
# ─────────────────────────────────────────────────────────────


def get_metrics_comparison(
    run_ids: list[str],
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch metrics for multiple runs for side-by-side comparison.

    How it works:
    1. Builds a query with WHERE id IN (?, ?, ?) using the right number
       of placeholders for the number of IDs
    2. Returns only the fields relevant for comparison (id, model_name,
       task_type, metrics, config, created_at, duration)
    3. Deserialises the JSON fields so you get real dicts

    Example output:
        [
            {"id": "abc...", "model_name": "rf", "metrics": {"accuracy": 0.87}},
            {"id": "def...", "model_name": "xgb", "metrics": {"accuracy": 0.91}},
        ]

    This is what powers the "compare experiments" feature in the web UI.

    Args:
        run_ids: List of UUID strings to compare.
        db_path: Path to the database file.

    Returns:
        List of dicts with comparison-relevant fields. Empty if no matches.
    """
    if not run_ids:
        return []

    path = Path(db_path) if db_path else DEFAULT_DB_PATH

    # Build the right number of ? placeholders
    placeholders = ", ".join("?" for _ in run_ids)
    query = f"""
        SELECT id, model_name, task_type, metrics, config,
               created_at, duration_seconds
        FROM experiments
        WHERE id IN ({placeholders})
        ORDER BY created_at DESC
    """

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, run_ids)
        rows = cursor.fetchall()

    if not rows:
        return []

    columns = list(rows[0].keys())
    return [_row_to_dict(tuple(row), columns) for row in rows]
