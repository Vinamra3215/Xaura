"""CLI entry point for XAURA.

Commands are split across Person A and Person B:
    Person A: xaura profile, xaura run
    Person B: xaura serve, xaura export
"""

import sys
from pathlib import Path

import click
import pandas as pd

from xaura import profile as profile_data
from xaura import run_model

# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="xaura")
def main():
    """XAURA - eXtendable Automated Unified Research & Analytics.

    An automated machine learning framework for profiling, modelling,
    visualising and exporting reproducible ML experiments.
    """


# -------------------------------------------------------------------
# Person B commands
# -------------------------------------------------------------------


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the server to.")
@click.option("--port", default=8000, type=int, help="Port to run the server on.")
@click.option("--reload", "use_reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host: str, port: int, use_reload: bool):
    """Start the XAURA FastAPI web server.

    Launches the dashboard at http://<host>:<port> where you can
    upload data, run models, view results, and manage experiments.

    Examples:

        xaura serve

        xaura serve --port 9000

        xaura serve --host 0.0.0.0 --reload
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is not installed. Run: pip install uvicorn[standard]", err=True)
        raise SystemExit(1) from None

    click.echo(f"Starting XAURA server at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "xaura.server.app:app",
        host=host,
        port=port,
        reload=use_reload,
    )


@main.command()
@click.argument("run_id")
@click.option(
    "--output-dir",
    "-o",
    default="./xaura_export",
    help="Directory to save the exported files.",
)
@click.option(
    "--fmt",
    type=click.Choice(["png", "pdf"], case_sensitive=False),
    default="png",
    help="Image format for exported plots.",
)
@click.option("--no-plots", is_flag=True, help="Skip exporting plots (export metadata only).")
@click.option(
    "--csv", "export_csv", is_flag=True, help="Also export the full experiment log as CSV."
)
def export(run_id: str, output_dir: str, fmt: str, no_plots: bool, export_csv: bool):
    """Export a model run bundle by its run ID.

    Fetches the run from the SQLite store and exports:
      - config.json — hyperparameters used
      - metrics.json — evaluation results
      - plots/ — all visualisation charts (unless --no-plots)
      - experiment_log.csv — full log (if --csv flag is set)

    Examples:

        xaura export abc123-def456

        xaura export abc123 -o ./my_export --fmt pdf

        xaura export abc123 --csv --no-plots
    """
    import json
    from pathlib import Path

    from xaura.store import get_run

    # 1. Fetch the run
    run = get_run(run_id)
    if run is None:
        click.echo(f"Error: No run found with ID '{run_id}'", err=True)
        click.echo("Use the experiment log to find valid run IDs.", err=True)
        raise SystemExit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.echo(f"Exporting run: {run_id}")
    click.echo(f"  Model: {run.get('model_name', 'unknown')}")
    click.echo(f"  Task:  {run.get('task_type', 'unknown')}")
    click.echo(f"  Output: {out.resolve()}\n")

    # 2. Export config.json
    config_path = out / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run.get("config", {}), f, indent=2, default=str)
    click.echo(f"  ✓ {config_path.name}")

    # 3. Export metrics.json
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(run.get("metrics", {}), f, indent=2, default=str)
    click.echo(f"  ✓ {metrics_path.name}")

    # 4. Export run_info.json (full metadata)
    info_path = out / "run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    click.echo(f"  ✓ {info_path.name}")

    # 5. Export plots (if not skipped)
    if not no_plots:
        click.echo("\n  Plots require a Result object from run_model().")
        click.echo("  Use xaura.export.export_plots(result, output_dir) in Python.")
        click.echo("  (CLI plot export from stored runs is planned for a future release.)")

    # 6. Export full CSV log (if requested)
    if export_csv:
        try:
            from xaura.export.csv_export import export_log_csv

            csv_path = out / "experiment_log.csv"
            export_log_csv(csv_path)
            click.echo(f"  ✓ {csv_path.name}")
        except Exception as exc:
            click.echo(f"  ✗ CSV export failed: {exc}", err=True)

    click.echo(f"\nDone! Files saved to: {out.resolve()}")


# ---------------------------------------------------------------------------
# xaura profile <csv>
# ---------------------------------------------------------------------------


@main.command("profile")
@click.argument("csv_path", type=click.Path(exists=True))
def profile_cmd(csv_path):
    """Profile a CSV dataset and display summary statistics.

    Reads the given CSV file, computes a DataProfile, and prints a
    formatted summary to stdout including shape, feature types,
    missing values, and warnings.
    """
    csv_path = Path(csv_path)
    click.echo(f"\nLoading {csv_path.name}...")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        click.secho(f"Error reading CSV: {exc}", fg="red")
        sys.exit(1)

    click.echo(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    p = profile_data(df)

    # Header
    click.secho("=" * 50, fg="cyan")
    click.secho("  XAURA Dataset Profile", fg="cyan", bold=True)
    click.secho("=" * 50, fg="cyan")

    # Shape
    click.echo(f"\n  Shape:              {p.n_rows:,} rows x {p.n_cols} columns")

    # Feature types
    for feat_type, columns in p.feature_types.items():
        if columns:
            click.echo(f"  {feat_type:20s} {len(columns)}")

    # Target
    if p.target_column:
        click.echo(f"\n  Target column:      {p.target_column}")
        click.echo(f"  Task type:          {p.task_type}")
        if p.is_imbalanced:
            click.secho("  Imbalanced:         Yes", fg="yellow")
        else:
            click.echo("  Imbalanced:         No")

    # Missing values
    if p.has_missing:
        pct = p.missing_fraction * 100
        click.secho(f"\n  Missing values:     {pct:.1f}%", fg="yellow")
    else:
        click.secho("\n  Missing values:     None", fg="green")

    # Warnings
    if p.warnings:
        click.echo()
        click.secho("  Warnings:", fg="yellow")
        for warning in p.warnings:
            click.echo(f"    - {warning}")

    # Hash
    click.echo(f"\n  Dataset hash:       {p.dataset_hash[:16]}...")
    click.echo()


# ---------------------------------------------------------------------------
# xaura run <model> <csv>
# ---------------------------------------------------------------------------


@main.command("run")
@click.argument("model_name")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Target column name.")
@click.option(
    "--export",
    "-e",
    is_flag=True,
    default=False,
    help="Export results as a ZIP bundle.",
)
@click.option(
    "--output-dir",
    "-o",
    default="./exports",
    help="Directory for exported ZIP (default: ./exports).",
)
def run_cmd(model_name, csv_path, target, export, output_dir):
    """Train and evaluate a model on a CSV dataset.

    MODEL_NAME is the registered model name (e.g., rf_classifier).
    CSV_PATH is the path to the dataset CSV file.
    """
    # Register all models
    import xaura.models.classifiers  # noqa: F401
    import xaura.models.clusterers  # noqa: F401
    import xaura.models.regressors  # noqa: F401

    csv_path = Path(csv_path)
    click.echo(f"\nLoading {csv_path.name}...")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        click.secho(f"Error reading CSV: {exc}", fg="red")
        sys.exit(1)

    click.echo(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    if target not in df.columns:
        click.secho(f"Error: column '{target}' not found in CSV.", fg="red")
        click.echo(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Profile
    click.echo("\nProfiling dataset...")
    p = profile_data(df)

    # Run model
    click.echo(f"Running {model_name}...")
    try:
        result = run_model(model_name, df, p, target_col=target, auto_log=False)
    except Exception as exc:
        click.secho(f"Error running model: {exc}", fg="red")
        sys.exit(1)

    click.echo(f"  Train time: {result.train_time_seconds:.2f}s")

    # Print metrics
    click.echo()
    click.secho("=" * 50, fg="cyan")
    click.secho("  Metrics", fg="cyan", bold=True)
    click.secho("=" * 50, fg="cyan")
    click.echo()

    for name, value in result.metrics.items():
        bar_len = int(value * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        click.echo(f"  {name:20s} {value:.4f}  [{bar}]")

    click.echo()

    # Export if requested
    if export:
        from xaura.export import export_run

        click.echo("Exporting ZIP bundle...")
        zip_path = export_run(result, p, output_dir=output_dir)
        click.secho(f"  Saved to: {zip_path}", fg="green")
        click.echo()


if __name__ == "__main__":
    main()
