"""Physical Review Research / APS matplotlib style for publication figures."""
from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402

COL_W, DCOL_W = 3.375, 7.0
INK, MUTED, BAND = "#000000", "#8a8984", "#d9d8d4"
SA = dict(color=INK, marker="o", ls="-", mfc=INK, mec=INK)
RANDOMFM = dict(color=MUTED, ls="--", lw=1.0)
BBO = {
    "XY-FMQAOA": dict(color="#2a78d6", ls="-", lw=1.6),
    "alpha1": dict(color=INK, ls="--", lw=1.0),
    "alpha100": dict(color="#eb6834", ls="-", lw=1.2),
    "alpha0.3": dict(color="#1baf7a", ls="-.", lw=1.2),
    "RandomFM": dict(color=MUTED, ls="--", lw=1.0),
    "Random": dict(color=MUTED, ls=":", lw=1.2),
}
DPI = 600
FORMATS = ("pdf", "png", "svg")
RC = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.handlelength": 2.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.titlesize": 8,
}
plt.rcParams.update(RC)


def apply_style() -> None:
    plt.rcParams.update(RC)


def plain_log_axis(axis, ticks=None, subs=(1.0, 2.0, 5.0)) -> None:
    axis.set_major_locator(
        FixedLocator(ticks) if ticks is not None else LogLocator(base=10, subs=subs)
    )
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def panel_letters(axes, start: int = 0, x: float = -0.02, y: float = 1.02) -> None:
    for k, ax in enumerate(axes):
        ax.text(
            x,
            y,
            f"({ascii_lowercase[start + k]})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=INK,
        )


def save(fig, out_dir: Path, stem: str, formats=FORMATS) -> None:
    for fmt in formats:
        (Path(out_dir) / fmt).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / fmt / f"{stem}.{fmt}", dpi=DPI)
    plt.close(fig)

