"""CLI smoke tests — verify that CLI commands load and display help.

These tests use Click's built-in CliRunner to invoke commands without
actually starting servers or writing to real databases.
"""

import pytest
from click.testing import CliRunner

from xaura.cli import main


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------


class TestMainGroup:
    """Tests for the top-level 'xaura' command."""

    def test_help_exits_zero(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_commands(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "serve" in result.output
        assert "export" in result.output

    def test_no_args_shows_usage(self, runner):
        result = runner.invoke(main, [])
        # Click returns exit code 2 for missing required command
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# xaura serve
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Smoke tests for 'xaura serve'."""

    def test_serve_help(self, runner):
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--reload" in result.output


# ---------------------------------------------------------------------------
# xaura export
# ---------------------------------------------------------------------------


class TestExportCommand:
    """Smoke tests for 'xaura export'."""

    def test_export_help(self, runner):
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "RUN_ID" in result.output
        assert "--output-dir" in result.output
        assert "--fmt" in result.output
        assert "--no-plots" in result.output
        assert "--csv" in result.output

    def test_export_missing_id_fails(self, runner):
        """Should fail without a run_id argument."""
        result = runner.invoke(main, ["export"])
        assert result.exit_code != 0

    def test_export_invalid_id(self, runner, tmp_path):
        """Should print error for non-existent run ID."""
        result = runner.invoke(
            main,
            ["export", "nonexistent-id-12345", "-o", str(tmp_path / "out")],
        )
        assert result.exit_code != 0
        assert "No run found" in result.output or result.exit_code == 1
