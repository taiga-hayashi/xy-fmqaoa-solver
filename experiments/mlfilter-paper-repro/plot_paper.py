"""Build plotting CSVs and render all 8 APS-style paper figures (excluding Boltzmann).

Usage:
    python plot_paper.py --run-dir <runs_dir> --fig-dir <figures_dir>
    python plot_paper.py --fig-dir <figures_dir> --replot
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from repro import aps_style as S
from repro import default_output_dirs
from repro.aggregate import MAIN_CONDITIONS, build_all_csvs
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {
    "XY-FMQAOA": r"XY-FMQAOA ($\lambda=0$)",
    "alpha1": r"SA, $\alpha=1$",
    "alpha100": r"SA, $\alpha=100$",
    "alpha0.3": r"SA, $\alpha=0.3$",
    "RandomFM": "Random proposals + FM selection",
    "Random": "Random search",
}

SCHEDULE_STYLE = {
    "auto": dict(color=S.INK, ls="-", marker="o"),
    "fmscale": dict(color="#eb6834", ls="--", marker="s", mfc="white"),
    "wide10x": dict(color="#1baf7a", ls="-.", marker="^", mfc="white"),
    "fmscale10x": dict(color="#2a78d6", ls=":", marker="D", mfc="white"),
}
SCHEDULE_LABEL = {
    "auto": "automatic (1000 sweeps)",
    "fmscale": "FM-scaled range",
    "wide10x": "automatic, 10x sweeps",
    "fmscale10x": "FM-scaled, 10x sweeps",
}

SHAPE = {
    "XY-FMQAOA": dict(color="#2a78d6", ls="-", lw=1.6),
    "CatSA": dict(color="#eb6834", ls="--", lw=1.2),
    "alpha1": dict(color=S.INK, ls="--", lw=1.0),
    "alpha100": dict(color=S.MUTED, ls="-", lw=1.0),
    "alpha0.3": dict(color=S.MUTED, ls=":", lw=1.2),
    "RandomFM": dict(color=S.MUTED, ls="--", lw=1.0),
}
SHAPE_LABEL = {
    "XY-FMQAOA": r"XY-FMQAOA ($\lambda=0$)",
    "CatSA": "Categorical SA",
    "alpha1": r"SA, $\alpha=1$",
    "alpha100": r"SA, $\alpha=100$",
    "alpha0.3": r"SA, $\alpha=0.3$",
    "RandomFM": "Random proposals + FM selection",
}


def _smart_log_yaxis(ax, preferred_ticks: list[float]) -> None:
    lo, hi = ax.get_ylim()
    in_view = [t for t in preferred_ticks if lo <= t <= hi]
    S.plain_log_axis(ax.yaxis, ticks=preferred_ticks if len(in_view) >= 2 else None)


def read_csv(csv_dir: Path, name: str) -> list[dict]:
    def _conv(val: str):
        if val == "":
            return None
        try:
            return float(val)
        except ValueError:
            return val

    with open(csv_dir / name) as fh:
        return [{k: _conv(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def fig_concept(fig_dir: Path) -> None:
    """1. Schematic of penalty energy barriers vs feasible-subspace XY ring mixer."""
    rng = np.random.default_rng(3)
    fm = np.sort(rng.normal(0, 1, 7))[::-1][[3, 0, 5, 2, 6, 1, 4]]
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, alpha, name in (
        (axes[0], 0.3, r"(small penalty, $\alpha=0.3$)"),
        (axes[1], 30.0, r"(large penalty, $\alpha=30$)"),
    ):
        scale = max(1.0, alpha)
        xf = np.arange(len(fm)) * 2
        xi = xf[:-1] + 1
        ef = fm / scale
        ei = (0.5 * (fm[:-1] + fm[1:]) - 2.0 + alpha * 1.0) / scale
        xs = np.concatenate([[xf[0]], np.ravel(np.column_stack([xi, xf[1:]]))])
        es = np.concatenate([[ef[0]], np.ravel(np.column_stack([ei, ef[1:]]))])
        ax.plot(xs, es, color=S.MUTED, lw=0.8, zorder=1)
        ax.plot(xf, ef, "o", color=S.BBO["alpha1"]["color"], ms=4, label="feasible", zorder=3)
        ax.plot(xi, ei, "x", color=S.BBO["alpha100"]["color"], ms=4, mew=1, label="infeasible", zorder=3)
        ax.set_xticks([])
        ax.set_ylabel("Normalised energy")
        ax.set_xlabel("States along a path\n" + name)
        ax.minorticks_off()
    axes[0].legend(loc="upper right", fontsize=6.5)

    ax = axes[2]
    n = 4
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 4
    for k, cx in enumerate((0.0, 2.6)):
        x, y = cx + np.cos(th), np.sin(th)
        for i in range(n):
            j = (i + 1) % n
            ax.annotate(
                "",
                xy=(x[j], y[j]),
                xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=S.BBO["XY-FMQAOA"]["color"],
                    lw=1.0,
                    shrinkA=6,
                    shrinkB=6,
                ),
            )
        for i in range(n):
            ax.plot(x[i], y[i], "o", ms=11, mfc="white", mec=S.INK, mew=0.8)
            ax.text(
                x[i],
                y[i],
                "".join("1" if t == i else "0" for t in range(n)),
                ha="center",
                va="center",
                fontsize=5,
            )
        ax.text(cx, -1.55, f"layer {k + 1}", ha="center", fontsize=7)
    ax.text(1.3, 1.45, r"XY mixer: $|1000\rangle\leftrightarrow|0100\rangle\leftrightarrow\cdots$", ha="center", fontsize=6.5)
    ax.set_xlim(-1.4, 4.0)
    ax.set_ylim(-1.8, 1.7)
    ax.axis("off")
    S.panel_letters(axes)
    fig.tight_layout(w_pad=0.8)
    S.save(fig, fig_dir, "concept")


def fig_paired(csv_dir: Path, fig_dir: Path) -> None:
    """2. Seed-paired final best ranks (Confirm 1)."""
    rows = read_csv(csv_dir, "confirm_final_ranks.csv")
    seeds = sorted({int(r["seed"]) for r in rows})
    get = {(r["condition"], int(r["seed"])): r["best_rank"] for r in rows}
    max_evals = int(max(r["evaluations_used"] for r in rows))
    comps = ["alpha0.3", "alpha100", "alpha1"]
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.4), sharey=True)
    for ax, c in zip(axes, comps):
        for s in seeds:
            a, b = get[(c, s)], get[("XY-FMQAOA", s)]
            col = (
                S.BBO["XY-FMQAOA"]["color"]
                if b < a
                else (S.MUTED if b == a else S.BBO["alpha100"]["color"])
            )
            ax.plot([0, 1], [a, b], "-", color=col, lw=0.8, alpha=0.9)
            ax.plot([0, 1], [a, b], "o", color=col, ms=3)
        w = sum(get[("XY-FMQAOA", s)] < get[(c, s)] for s in seeds)
        t = sum(get[("XY-FMQAOA", s)] == get[(c, s)] for s in seeds)
        ax.text(
            0.5,
            0.97,
            f"XY better {w}, tie {t}, worse {len(seeds) - w - t}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LABEL[c], "XY-FMQAOA"], fontsize=7)
        ax.set_xlim(-0.3, 1.3)
        ax.minorticks_off()
        ax.set_yscale("log")
        _smart_log_yaxis(ax, [1, 3, 10, 30, 100, 300])
    axes[0].set_ylabel(f"True rank of the best solution\nafter {max_evals} evaluations")
    S.panel_letters(axes)
    fig.tight_layout(w_pad=0.6)
    S.save(fig, fig_dir, "paired_ranks")


def fig_loop_alpha_sweep(csv_dir: Path, fig_dir: Path) -> None:
    """3. Closed-loop penalty coefficient alpha sweep on L = 7."""
    rows = read_csv(csv_dir, "loop_alpha_sweep.csv")
    alphas = [0.3, 1, 3, 10, 30, 100, 1000]

    def _stat(c: str):
        v = [r["best_rank"] for r in rows if r["condition"] == c]
        return (np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v)) if v else None

    fig, ax = plt.subplots(figsize=(S.COL_W, 2.6))
    sa = [(a, _stat(f"alpha{a:g}")) for a in alphas]
    sa = [(a, s) for a, s in sa if s]
    a_arr = np.array([x for x, _ in sa])
    m_arr = np.array([s[0] for _, s in sa])
    lo_arr = np.array([s[1] for _, s in sa])
    hi_arr = np.array([s[2] for _, s in sa])
    ax.fill_between(a_arr, lo_arr, hi_arr, color=S.BAND, lw=0, zorder=0)
    ax.plot(a_arr, m_arr, label=r"SA with penalty $\alpha$ (median, IQR)", zorder=3, **S.SA)

    xy = _stat("XY-FMQAOA")
    n_seeds = xy[3] if xy else 15
    if xy:
        ax.axhspan(xy[1], xy[2], color=S.BBO["XY-FMQAOA"]["color"], alpha=0.15, lw=0)
        ax.axhline(xy[0], label=r"XY-FMQAOA ($\lambda=0$)", **S.BBO["XY-FMQAOA"])
    ad = _stat("Adaptive")
    if ad:
        ax.axhline(ad[0], label=r"SA, adaptive $\alpha$", color=S.INK, ls=":", lw=1.0)
    rf = _stat("RandomFM")
    if rf:
        ax.axhline(rf[0], label="Random proposals + FM selection", **S.RANDOMFM)

    ax.set_xscale("log")
    S.plain_log_axis(ax.xaxis, ticks=alphas)
    ax.set_xlim(0.2, 1600)
    ax.set_yscale("log")
    S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100])
    y_max = max(300.0, float(np.max(hi_arr)) * 1.4) if len(hi_arr) else 300.0
    ax.set_ylim(0.8, y_max)
    ax.set_xlabel(r"Penalty coefficient $\alpha$ ($\lambda=\alpha S$)")
    ax.set_ylabel("True rank of the best solution\nafter budget")
    if n_seeds == 15:
        ax.set_ylabel("True rank of the best solution\nafter 100 evaluations")

    opt = lambda c: sum(1 for r in rows if r["condition"] == c and r["best_rank"] == 1)
    ax.text(
        0.98,
        0.04,
        f"optimum found: XY {opt('XY-FMQAOA')}/{n_seeds}, adaptive {opt('Adaptive')}/{n_seeds}, "
        + r"$\alpha$=1 "
        + f"{opt('alpha1')}/{n_seeds}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    ax.legend(loc="upper center", fontsize=6, ncol=1)
    fig.tight_layout()
    S.save(fig, fig_dir, "loop_alpha_sweep")


def fig_retune(csv_dir: Path, fig_dir: Path) -> None:
    """4. SA schedule retuning robustness on frozen FMs (L=6,7,8) and closed-loop L=7."""
    rows = read_csv(csv_dir, "fixedfm_schedules.csv")
    alphas = [1, 3, 10, 30, 100, 1000]
    fig, axes = plt.subplots(1, 4, figsize=(S.DCOL_W, 2.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for ax, L in zip(axes[:3], (6, 7, 8)):
        for sch, st in SCHEDULE_STYLE.items():
            m = [
                np.median(
                    [
                        r["sample_fm_pct"]
                        for r in rows
                        if r["layers"] == L
                        and r["sampler"] == f"SA-{sch}-a{a:g}"
                        and r["sample_fm_pct"] is not None
                    ]
                    or [np.nan]
                )
                for a in alphas
            ]
            ax.plot(alphas, m, label=SCHEDULE_LABEL[sch], ms=3.5, lw=1.0, **st)
        rf_vals = [r["sample_fm_pct"] for r in rows if r["layers"] == L and r["sampler"] == "RandomFM" and r["sample_fm_pct"] is not None]
        xy_vals = [r["sample_fm_pct"] for r in rows if r["layers"] == L and r["sampler"] == "XY-p4-cvar10" and r["sample_fm_pct"] is not None]
        if rf_vals:
            ax.axhline(np.median(rf_vals), **S.RANDOMFM, label="Random proposals")
        if xy_vals:
            ax.axhline(np.median(xy_vals), label=r"XY-FMQAOA $p=4$", **S.BBO["XY-FMQAOA"])
        ax.set_xscale("log")
        S.plain_log_axis(ax.xaxis, ticks=alphas)
        ax.set_ylim(-0.02, 0.56)
        ax.set_xlabel(r"$\alpha$")
        ax.text(0.03, 0.95, f"$L={L}$", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    axes[0].set_ylabel("Mean FM percentile of samples\n(0 = FM best, 0.5 = random)")

    loop = read_csv(csv_dir, "loop_alpha_sweep.csv")
    conds = [
        ("alpha100", "auto"),
        ("alpha100-fmscale", "fmscale"),
        ("alpha100-wide10x", "wide10x"),
        ("alpha100-fmscale10x", "fmscale10x"),
        ("alpha1", None),
        ("XY-FMQAOA", None),
    ]
    ax = axes[3]
    for i, (c, sch) in enumerate(conds):
        v = [r["best_rank"] for r in loop if r["condition"] == c]
        if not v:
            continue
        col = SCHEDULE_STYLE[sch]["color"] if sch else S.BBO[c]["color"]
        ax.plot(
            [i] * len(v) + 0.12 * np.random.default_rng(i).uniform(-1, 1, len(v)),
            v,
            "o",
            ms=2.5,
            color=col,
            alpha=0.6,
        )
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=col, lw=1.6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(
        [r"$\alpha$=100" + "\nauto", "FM-\nscaled", "auto\n10x", "FM-sc.\n10x", r"$\alpha$=1", "XY"],
        fontsize=5.5,
    )
    ax.minorticks_off()
    ax.set_yscale("log")
    S.plain_log_axis(ax.yaxis, ticks=[1, 3, 10, 30, 100])
    ax.set_ylabel("Closed-loop best rank (L=7)")
    S.panel_letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=3,
        fontsize=6.5,
        handlelength=2.6,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(w_pad=0.5, rect=(0, 0.15, 1, 1))
    S.save(fig, fig_dir, "retune_robustness")


def fig_energy_hist(csv_dir: Path, fig_dir: Path) -> None:
    """5. FM-percentile density histogram of feasible proposals on frozen FMs (L = 7)."""
    rows = read_csv(csv_dir, "energy_hist.csv")
    L = 7
    samplers = [
        ("SA-auto-a1", r"SA, $\alpha=1$", dict(color=S.INK, ls="-", lw=1.1)),
        ("SA-auto-a10", r"SA, $\alpha=10$", dict(color="#eb6834", ls="--", lw=1.1)),
        ("SA-auto-a100", r"SA, $\alpha=100$", dict(color="#1baf7a", ls="-.", lw=1.1)),
        ("XY-p4-cvar10", r"XY-FMQAOA $p=4$", dict(color=S.BBO["XY-FMQAOA"]["color"], ls="-", lw=1.6)),
        ("RandomFM", "Uniform random", dict(color=S.MUTED, ls=":", lw=1.1)),
    ]
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.5))
    for key, lab, st_orig in samplers:
        st = dict(st_orig)
        r = [x for x in rows if x["layers"] == L and x["sampler"] == key]
        if not r:
            continue
        lo = sorted({x["bin_lo"] for x in r})
        cnt = np.array([sum(x["count"] for x in r if x["bin_lo"] == b) for b in lo], dtype=float)
        if cnt.sum() == 0:
            continue
        dens = cnt / cnt.sum() / (lo[1] - lo[0])
        edges = np.append(lo, 1.0)
        ax.stairs(dens, edges, label=lab, lw=st.pop("lw", 1.1), **st)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("FM percentile of a feasible sample (0 = FM best)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    S.save(fig, fig_dir, "energy_hist")


def fig_loop_mechanism(csv_dir: Path, fig_dir: Path) -> None:
    """6. Per-cycle closed-loop mechanism trajectories (selected true rank, FM-min hit, diversity)."""
    rows = read_csv(csv_dir, "confirm_mechanism_per_cycle.csv")
    conds = ["XY-FMQAOA", "alpha1", "alpha100", "alpha0.3", "RandomFM"]
    max_cyc = int(max(r["cycle"] for r in rows)) if rows else 60
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.5))

    def _series(c: str, key: str, fn):
        cyc = sorted({int(r["cycle"]) for r in rows if r["condition"] == c})
        v = []
        for k in cyc:
            x = [r[key] for r in rows if r["condition"] == c and int(r["cycle"]) == k and r[key] is not None]
            v.append(fn(x) if x else np.nan)
        arr = np.array(v, dtype=float)
        w = min(5, max(1, len(arr)))
        sm = np.array([np.nanmean(arr[max(0, i - w + 1) : i + 1]) for i in range(len(arr))])
        return np.array(cyc), sm

    for c in conds:
        st = S.BBO[c]
        x, y = _series(c, "selected_true_rank", np.median)
        axes[0].plot(x, y, label=LABEL[c], **st)
        x, y = _series(c, "unseen_fm_min_hit", np.mean)
        axes[1].plot(x, y, **st)
        x, y = _series(c, "unique_new_candidates", np.median)
        axes[2].plot(x, y, **st)
    axes[0].set_yscale("log")
    _smart_log_yaxis(axes[0], [10, 30, 100, 300, 1000, 3000])
    axes[0].set_ylabel("True rank of the\nselected candidate")
    axes[1].set_ylabel("Fraction selecting the\nunseen FM minimum")
    axes[2].set_ylabel("Distinct new feasible\ncandidates per cycle")
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.set_xlabel("Proposal cycle")
        ax.set_xlim(1, max(2, max_cyc))
    S.panel_letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=5,
        fontsize=6.5,
        handlelength=2.6,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    S.save(fig, fig_dir, "loop_mechanism")


def fig_confirm2(csv_dir: Path, fig_dir: Path) -> None:
    """7. Second confirmation (L = 7, independent seeds, CatSA vs XY-FMQAOA & SA, no Boltzmann)."""
    rows = read_csv(csv_dir, "confirm2_curves.csv")
    conds = ["XY-FMQAOA", "CatSA", "alpha1", "alpha100", "alpha0.3"]
    evals_sorted = sorted({int(x["evaluations"]) for x in rows})
    n_min, n_max = evals_sorted[0], evals_sorted[-1]
    n_seeds = len({int(x["seed"]) for x in rows if x["condition"] == "XY-FMQAOA"}) or 30
    n_grid = np.arange(n_min, n_max + 1)

    fig, axes = plt.subplots(1, 2, figsize=(S.DCOL_W, 2.6))
    for c in conds:
        r = [x for x in rows if x["condition"] == c]
        if not r:
            continue
        axes[0].plot(
            n_grid,
            [np.median([x["best_rank"] for x in r if int(x["evaluations"]) == k]) for k in n_grid],
            label=SHAPE_LABEL[c],
            **SHAPE[c],
        )
        axes[1].plot(
            n_grid,
            [np.mean([x["best_rank"] == 1 for x in r if int(x["evaluations"]) == k]) for k in n_grid],
            **SHAPE[c],
        )
    axes[0].set_yscale("log")
    _smart_log_yaxis(axes[0], [1, 3, 10, 30, 100, 300])
    axes[0].set_ylabel(f"True rank of the best solution\n(median over {n_seeds} seeds)")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Fraction of seeds that found\nthe global optimum")
    for ax in axes:
        ax.set_xlim(n_min, n_max)
        ax.set_xlabel("Evaluations")
    S.panel_letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=3,
        fontsize=6.5,
        handlelength=2.6,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(w_pad=1.2, rect=(0, 0.14, 1, 1))
    S.save(fig, fig_dir, "confirm2")


def fig_si_explore(csv_dir: Path, fig_dir: Path) -> None:
    """8. Multi-layer closed-loop scaling across L = 6, 7, 8."""
    rows = read_csv(csv_dir, "explore_curves.csv")
    evals_sorted = sorted({int(x["evaluations"]) for x in rows})
    n_min, n_max = evals_sorted[0], evals_sorted[-1]
    n_seeds = len({int(x["seed"]) for x in rows if x["layers"] == 7 and x["condition"] == "XY-FMQAOA"}) or 5
    n_grid = np.arange(n_min, n_max + 1)

    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.6), sharey=True)
    for ax, L in zip(axes, (6, 7, 8)):
        for c in MAIN_CONDITIONS:
            r = [x for x in rows if int(x["layers"]) == L and x["condition"] == c]
            if not r:
                continue
            med = [np.median([x["best_rank"] for x in r if int(x["evaluations"]) == k]) for k in n_grid]
            ax.plot(n_grid, med, label=LABEL[c], **S.BBO[c])
        ax.set_yscale("log")
        _smart_log_yaxis(ax, [1, 3, 10, 30, 100, 300, 1000])
        ax.set_xlim(n_min, n_max)
        ax.set_xlabel("Evaluations")
        ax.text(
            0.97,
            0.95,
            f"$L={L}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )
    axes[0].set_ylabel(f"True rank of the best solution\n(median over {n_seeds} seeds)")
    S.panel_letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=3,
        fontsize=6.5,
        handlelength=2.6,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(w_pad=0.6, rect=(0, 0.14, 1, 1))
    S.save(fig, fig_dir, "si_closed_loop_L678")


def draw_all_figures(fig_dir: Path) -> list[str]:
    S.apply_style()
    fig_dir = Path(fig_dir)
    csv_dir = fig_dir / "csv"
    drawn = ["concept"]
    fig_concept(fig_dir)

    if (csv_dir / "confirm_final_ranks.csv").exists():
        fig_paired(csv_dir, fig_dir)
        drawn.append("paired_ranks")
    if (csv_dir / "loop_alpha_sweep.csv").exists():
        rows = read_csv(csv_dir, "loop_alpha_sweep.csv")
        if any(r["source"] == "paper-extra" for r in rows):
            fig_loop_alpha_sweep(csv_dir, fig_dir)
            drawn.append("loop_alpha_sweep")
    if (csv_dir / "fixedfm_schedules.csv").exists() and (csv_dir / "loop_alpha_sweep.csv").exists():
        fig_retune(csv_dir, fig_dir)
        drawn.append("retune_robustness")
    if (csv_dir / "energy_hist.csv").exists():
        fig_energy_hist(csv_dir, fig_dir)
        drawn.append("energy_hist")
    if (csv_dir / "confirm_mechanism_per_cycle.csv").exists():
        fig_loop_mechanism(csv_dir, fig_dir)
        drawn.append("loop_mechanism")
    if (csv_dir / "confirm2_curves.csv").exists():
        fig_confirm2(csv_dir, fig_dir)
        drawn.append("confirm2")
    if (csv_dir / "explore_curves.csv").exists():
        fig_si_explore(csv_dir, fig_dir)
        drawn.append("si_closed_loop_L678")
    return drawn


def main() -> None:
    default_run_dir, default_fig_dir = default_output_dirs("full")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir,
        help="Root directory containing suite outputs (truth, fixed_fm, confirm1, confirm2, explore_L678).",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=default_fig_dir,
        help="Output directory for csv/, pdf/, png/, and svg/.",
    )
    ap.add_argument(
        "--replot",
        action="store_true",
        help="Redraw figures directly from existing <fig-dir>/csv/ without reading --run-dir.",
    )
    args = ap.parse_args()
    if not args.replot:
        csvs = build_all_csvs(args.run_dir, args.fig_dir / "csv")
        print("CSV written:", ", ".join(csvs))
    drawn = draw_all_figures(args.fig_dir)
    print(f"Figures ({len(drawn)}) written to {args.fig_dir}: {', '.join(drawn)}")


if __name__ == "__main__":
    main()

