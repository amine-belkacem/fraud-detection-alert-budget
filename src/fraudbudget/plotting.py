"""Shared colours and axis style for every figure of the project.

Series take colours in this order: teal, violet, magenta, yellow, green. Blue
and orange come last, only if a figure ever needs more than five series; I
prefer to keep them out of the charts. Grey is reserved for baselines (dummy
model, random ranking, "flag nothing"). Teal, magenta and yellow have a low
contrast on white, so every figure keeps visible labels or a legend.
"""

from __future__ import annotations

TEAL = "#1baf7a"
VIOLET = "#6250d6"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
GREEN = "#008300"
BLUE = "#2a78d6"    # last resort
ORANGE = "#eb6834"  # last resort
GREY = "#8a8986"    # baselines only

SERIES = (TEAL, VIOLET, MAGENTA, YELLOW, GREEN, BLUE, ORANGE)

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def style_axis(ax, grid_axis: str = "both") -> None:
    """Light grid behind the data, no top/right spines, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
