"""Shared style constants for all XAURA visualisation modules.

Both Plotly and Matplotlib charts import from here to ensure
a consistent dark-theme look across the entire library.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Colour palette — dark theme (matches Person A's classification charts)
# ---------------------------------------------------------------------------
BG_COLOR = "#0d1117"
PAPER_COLOR = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"

ACCENT_COLORS = [
    "#b4a7d6",  # lavender
    "#e67e22",  # orange
    "#8b0000",  # blood red
    "#2454b8",  # royal blue
    "#2ecc71",  # emerald green
    "#00e5ff",  # cyan
    "#ff69b4",  # pink
    "#8b6914",  # hazelnut brown
]

# Semantic aliases for common plot elements
PRIMARY = ACCENT_COLORS[0]  # lavender
SECONDARY = ACCENT_COLORS[1]  # orange
REFERENCE_LINE = "#e67e22"  # orange — stands out on dark bg

# ---------------------------------------------------------------------------
# Plotly base layout
# ---------------------------------------------------------------------------


def plotly_base_layout(**overrides: Any) -> dict[str, Any]:
    """Return a base Plotly layout dict with consistent dark theme styling."""
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": PAPER_COLOR,
        "plot_bgcolor": BG_COLOR,
        "font": {"family": FONT_FAMILY, "color": TEXT_COLOR, "size": 13},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        "colorway": ACCENT_COLORS,
    }
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

# Dark theme params for matplotlib
MPL_RC_PARAMS = {
    "figure.facecolor": PAPER_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "savefig.facecolor": PAPER_COLOR,
    "savefig.edgecolor": PAPER_COLOR,
}

MPL_FIG_SIZE = (8, 6)
MPL_DPI = 150
