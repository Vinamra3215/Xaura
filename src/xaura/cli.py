"""CLI entry point for XAURA.

Commands are split across Person A and Person B:
    Person A: xaura profile, xaura run
    Person B: xaura serve, xaura export
"""

import click


@click.group()
def main():
    """XAURA — eXtendable Automated Unified Research & Analytics."""
    pass


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


# -------------------------------------------------------------------
# Person A commands (placeholders — Person A will implement these)
# -------------------------------------------------------------------

# @main.command()
# def profile(): ...

# @main.command()
# def run(): ...
