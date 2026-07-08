"""Publication figure styling constants."""

from __future__ import annotations

import matplotlib as mpl

DPI = 300
FONT_SIZE = 10
TITLE_FONT_SIZE = 12
PANEL_TITLE_SIZE = 11

# Matplotlib RGB colors (0–1)
COLOR_GT = (0.15, 0.72, 0.25)
COLOR_YOLO = (0.86, 0.18, 0.18)
COLOR_CENTER = (0.15, 0.45, 0.90)
COLOR_ENDPOINT = (0.95, 0.78, 0.10)
COLOR_HIGHLIGHT = (0.55, 0.20, 0.75)
COLOR_TEXT = (0.10, 0.10, 0.10)
COLOR_PANEL_BG = (1.0, 1.0, 1.0)

BBOX_LINEWIDTH = 3
POINT_SIZE = 48
ENDPOINT_SIZE = 36

MODEL_DISPLAY_NAME = "Qwen2.5-VL"


def apply_publication_style() -> None:
    """Configure matplotlib for clean publication figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": FONT_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "figure.titlesize": TITLE_FONT_SIZE,
            "axes.linewidth": 0.8,
            "xtick.bottom": False,
            "ytick.left": False,
            "axes.grid": False,
        }
    )


def wrap_text(text: str, width: int = 52) -> str:
    """Wrap long reasoning text for figure panels."""
    if not text:
        return "n/a"
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)
