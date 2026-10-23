"""Shared figure style (Physical Review Research / APS format) for the Fujitsu_OpenQarp figures.

Edit here to restyle every figure that uses it.
  widths   one column 3.375 in, two columns 7.0 in
  text     Times-like serif (Liberation Serif), STIX math, 8 pt (ticks / legends 7 pt)
  axes     box on all four sides, inward ticks with minor ticks, no grid, no titles
  panels   (a), (b), ... via panel_letters
  output   pdf (TrueType embedded), png 600 dpi, svg (text kept as text) via save()
Colours (validated with the dataviz palette validator):
  SA alpha sweep = ink black; random proposals + FM selection = muted gray, dashed;
  XY-FMQAOA depth p = one blue ramp light -> dark (ordinal), told apart also by line style;
  layer count L = categorical slots 1-3 (blue, orange, aqua) with distinct markers.
"""
from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W, DCOL_W = 3.375, 7.0
INK, MUTED, BAND = "#000000", "#8a8984", "#d9d8d4"
SA = dict(color=INK, marker="o", ls="-", mfc=INK, mec=INK)
RANDOMFM = dict(color=MUTED, ls="--", lw=1.0)
XY_DEPTH = {1: dict(color="#86b6ef", ls=":", lw=1.2), 2: dict(color="#2a78d6", ls="-.", lw=1.2),
            3: dict(color="#0d366b", ls="--", lw=1.2)}
XQAOA = dict(color=INK, marker="x", ls="none", ms=5)
# depth figure: XY objective (series over p) and SA references (horizontal lines, alpha told apart by line style)
XY_OBJECTIVE = {"cvar10": dict(color="#0d366b", ls="-", marker="o", mfc="#0d366b", mec="#0d366b", lw=1.2),
                "mean": dict(color="#2a78d6", ls="--", marker="s", mfc="white", mec="#2a78d6", lw=1.2)}
# closed-loop comparison (bbo): XY = slot 1 blue, tuned SA = ink, mis-tuned SA = slots 2 (orange) and 3 (aqua), random = muted
BBO = {"XY-FMQAOA": dict(color="#2a78d6", ls="-", lw=1.6),
       "alpha1": dict(color=INK, ls="--", lw=1.0),
       "alpha100": dict(color="#eb6834", ls="-", lw=1.2),
       "alpha0.3": dict(color="#1baf7a", ls="-.", lw=1.2),
       "RandomFM": dict(color=MUTED, ls="--", lw=1.0),
       "Random": dict(color=MUTED, ls=":", lw=1.2)}
SA_REF = {1: dict(color=INK, ls="-", lw=1.1), 3: dict(color="#1baf7a", ls="--", lw=1.1),
          10: dict(color="#eb6834", ls="-.", lw=1.2), 100: dict(color="#b51963", ls=":", lw=1.3)}
LAYER = {6: dict(color="#2a78d6", marker="o"), 7: dict(color="#eb6834", marker="s"), 8: dict(color="#1baf7a", marker="^")}
DPI = 600
FORMATS = ("pdf", "png", "svg")
RC = {
    "font.family": "serif", "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "axes.grid": False, "legend.frameon": False,
    "legend.handlelength": 2.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.titlesize": 8,
}
plt.rcParams.update(RC)


def plain_log_axis(axis, ticks=None, subs=(1.0, 2.0, 5.0)):
    """Log axis labelled 0.3, 1, 3, ... instead of 3 x 10^-1 (no minor labels)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(FixedLocator(ticks) if ticks is not None else LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def panel_letters(axes, start: int = 0, x: float = -0.02, y: float = 1.02):
    for k, ax in enumerate(axes):
        ax.text(x, y, f"({ascii_lowercase[start + k]})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)


def save(fig, out_dir: Path, stem: str, formats=FORMATS):
    """<out_dir>/<fmt>/<stem>.<fmt> for every format."""
    for fmt in formats:
        (Path(out_dir) / fmt).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / fmt / f"{stem}.{fmt}", dpi=DPI)
    plt.close(fig)
