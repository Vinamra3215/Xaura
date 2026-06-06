"""CLI entry point for XAURA."""

import click


@click.group()
def main():
    """XAURA — eXtendable Automated Unified Research & Analytics."""
    pass


# Commands will be added in Week 3:
# xaura profile <csv>
# xaura run <model> <csv>
# xaura serve
# xaura export <run_id>
