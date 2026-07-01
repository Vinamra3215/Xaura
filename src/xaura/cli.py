"""CLI entry point for XAURA.

Commands:
    xaura profile <csv>                    Profile a dataset
    xaura run <model> <csv> --target col   Train and evaluate a model
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
