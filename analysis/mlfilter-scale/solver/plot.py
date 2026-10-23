"""Figures for the solver-level comparison on frozen FMs (experiments/mlfilter-scale, solver-fixedfm).

  python plot.py            rebuild the CSVs from the run summaries, then draw every figure
  python plot.py --replot   draw every figure from the existing CSVs only (seconds)

Every figure function reads only csv/; colours, fonts and sizes come from src/aps_style.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT/'src'))
import aps_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CONFIG = json.loads((HERE/'analysis_config.json').read_text())
CSV = HERE/'csv'
ALPHAS = [0.3, 1, 3, 10, 30, 100, 1000]
LAYERS = [6, 7, 8]
CANDIDATES = {6: 4096, 7: 16384, 8: 65536}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ CSV (from run summaries)
def build_csv():
    CSV.mkdir(exist_ok=True)
    per_fm, summary, exact = [], [], []
    for item in CONFIG['inputs']:
        path = PROJECT/item['summary']; assert sha(path)==item['sha256'], f'input changed: {path}'
        d = json.loads(path.read_text()); L = d['layers']; assert L==item['layers']
        for sampler, rows in d['per_fm'].items():
            for r in rows: per_fm.append({'layers':L,'sampler':sampler,**r})
        for sampler, s in d['summary'].items():
            for metric, v in s.items():
                if metric!='n_fm': summary.append({'layers':L,'sampler':sampler,'metric':metric,'median':v[0],'q25':v[1],'q75':v[2],'n_fm':s['n_fm']})
        for sampler, rows in d['per_fm_exact'].items():
            for r in rows: exact.append({'layers':L,'sampler':sampler,**r})
    for name, rows in [('solver_per_fm.csv',per_fm),('solver_summary.csv',summary),('solver_exact_per_fm.csv',exact)]:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(CSV/name,'w',newline='') as fh:
            w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    print('CSV written:', ', '.join(p.name for p in sorted(CSV.glob('*.csv'))))


def read(name):
    with open(CSV/name) as fh:
        return [{k:(None if v=='' else v if k in ('sampler','metric') else float(v)) for k,v in r.items()} for r in csv.DictReader(fh)]


def stat(rows, L, sampler, metric):
    r = [x for x in rows if x['layers']==L and x['sampler']==sampler and x['metric']==metric]
    return (r[0]['median'], r[0]['q25'], r[0]['q75']) if r else (np.nan, np.nan, np.nan)


def sa_series(rows, L, metric, alphas=ALPHAS):
    out = np.array([stat(rows, L, f'SA-a{a:g}', metric) for a in alphas]); return np.array(alphas, float), out


# ------------------------------------------------------------------ figures
def _alpha_axis(ax):
    ax.set_xscale('log'); ax.set_xlim(0.2, 1600); S.plain_log_axis(ax.xaxis, ticks=[0.3, 1, 3, 10, 30, 100, 1000])
    ax.set_xlabel(r'Penalty coefficient $\alpha$ ($\lambda = \alpha S$)')


def _references(ax, rows, L, metric, objective):
    h = [ax.axhline(stat(rows, L, 'RandomFM', metric)[0], label='Random proposals + FM selection', zorder=1, **S.RANDOMFM)]
    for p in (1, 2, 3):
        h.append(ax.axhline(stat(rows, L, f'XY-p{p}-{objective}', metric)[0], label=fr'XY-FMQAOA $p={p}$ ($\lambda=0$)', zorder=1, **S.XY_DEPTH[p]))
    return h


def _alpha_panel(ax, rows, L, metric, objective, alphas, ylim, xlim):
    a, v = sa_series(rows, L, metric, alphas); ok = np.isfinite(v[:,0])
    ax.fill_between(a[ok], v[ok,1], v[ok,2], color=S.BAND, lw=0, zorder=0)
    h = _references(ax, rows, L, metric, objective)
    hsa, = ax.plot(a[ok], v[ok,0], label=r'SA with penalty $\alpha$ (median, IQR band)', zorder=3, **S.SA)
    _alpha_axis(ax); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    # Direct labels at the left end of the XY lines; depths with (nearly) equal values share one label.
    vals = sorted((stat(rows, L, f'XY-p{p}-{objective}', metric)[0], p) for p in (1, 2, 3))
    groups = []
    for v, p in vals:
        if groups and abs(v - groups[-1][0][0]) < 0.045*(ylim[1]-ylim[0]): groups[-1].append((v, p))
        else: groups.append([(v, p)])
    for g in groups:
        ax.text(0.02, g[0][0], ' $p=' + ',\\,'.join(str(p) for _, p in sorted(g, key=lambda t: t[1])) + '$', transform=ax.get_yaxis_transform(),
                ha='left', va='bottom', fontsize=6, color=S.INK)
    ax.text(0.97, 0.05, f'$L={L}$ ({CANDIDATES[L]:,} candidates)', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    return [hsa, *h]


def _alpha_figure(stem, metric, objective, alphas, ylim, xlim, ylabel):
    rows = read('solver_summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(S.DCOL_W, 2.6), sharey=True)
    for ax, L in zip(axes, LAYERS): handles = _alpha_panel(ax, rows, L, metric, objective, alphas, ylim, xlim)
    axes[0].set_ylabel(ylabel); S.panel_letters(axes)
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=6.5, handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.09, 1, 1)); S.save(fig, HERE, stem)
    for L in LAYERS:   # independent panels
        fig, ax = plt.subplots(figsize=(S.COL_W, 2.9))
        handles = _alpha_panel(ax, rows, L, metric, objective, alphas, ylim, xlim); ax.set_ylabel(ylabel)
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=6.5, handlelength=2.6, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.17, 1, 1)); S.save(fig, HERE/"panels", f'{stem}_L{L}')


def fig_flattening(objective='cvar10'):
    """Mean FM percentile of the feasible samples vs alpha (0 = FM best, 0.5 = uniform random)."""
    _alpha_figure(f'penalty_flattening_{objective}', 'sample_fm_pct', objective, ALPHAS, (-0.02, 0.56), (0.2, 1600),
                  'Mean FM percentile of samples\n(0 = FM best, 0.5 = random)')


def fig_gap(objective='cvar10'):
    """Normalized FM gap of the selected candidate vs alpha (alpha >= 1; alpha = 0.3 has no feasible candidate at L = 6, 8)."""
    _alpha_figure(f'penalty_selected_gap_{objective}', 'norm_gap', objective, ALPHAS[1:], (-0.004, 0.105), (0.7, 1600),
                  'FM gap of the selected candidate\n(normalized by unseen FM range)')


def fig_mechanism():
    """Per-FM sample percentile vs FM contrast at the SA freezing point, dE_FM / lambda (proportional to 1/alpha)."""
    rows = [r for r in read('solver_per_fm.csv') if r['sampler'].startswith('SA-a') and r.get('sample_fm_pct') is not None]
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.5))
    ax.axhline(0.5, **S.RANDOMFM, label='Uniform random (0.5)')
    for L in LAYERS:
        rr = [r for r in rows if r['layers']==L]
        ax.plot([r['contrast_freeze'] for r in rr], [r['sample_fm_pct'] for r in rr], ls='none', ms=3.5, mfc='none', mew=0.8,
                label=f'$L={L}$', **S.LAYER[L])
    ax.set_xscale('log'); S.plain_log_axis(ax.xaxis, ticks=[0.001, 0.01, 0.1, 1])
    ax.set_xlabel(r'FM contrast at freezing, $\Delta E_\mathrm{FM}/\lambda$')
    ax.set_ylabel('Mean FM percentile of samples')
    ax.set_ylim(-0.02, 0.56); ax.legend(loc='upper right', fontsize=6.5)
    fig.tight_layout(); S.save(fig, HERE, 'penalty_mechanism')


def fig_feasibility():
    """Feasible fraction of the 100 samples vs alpha: SA per L, standard QAOA (X mixer + penalty, L = 6), XY = 1 by construction."""
    rows = read('solver_summary.csv')
    fig, ax = plt.subplots(figsize=(S.COL_W, 2.4))
    xy = np.nanmin([stat(rows, L, f'XY-p{p}-{o}', 'feasible')[0] for L in LAYERS for p in (1, 2, 3) for o in ('mean', 'cvar10')])
    ax.axhline(xy, label=r'XY-FMQAOA (all $L$, $p$; $\lambda=0$)', zorder=1, **S.XY_DEPTH[3])
    for L, ms in zip(LAYERS, (6, 4.5, 3)):
        a, v = sa_series(rows, L, 'feasible')
        ax.plot(a, v[:,0], label=f'SA, $L={L}$', ls='-', lw=0.8, ms=ms, mfc='none', mew=0.8, zorder=3, **S.LAYER[L])
    xa = [1, 10, 100]; xv = [stat(rows, 6, f'X-QAOA-a{a:g}', 'feasible')[0] for a in xa]
    ax.plot(xa, xv, label=r'Standard QAOA (X mixer), $L=6$', zorder=4, **S.XQAOA)
    _alpha_axis(ax); ax.set_ylim(-0.04, 1.08); ax.set_ylabel('Feasible fraction of samples')
    ax.legend(loc='center right', fontsize=6.5); fig.tight_layout(); S.save(fig, HERE, 'penalty_feasibility')


def _sweetspot_panel_single_L(ax_top, ax_bot, rows, L):
    a_cap, v_cap = sa_series(rows, L, 'capture')
    a_feas, v_feas = sa_series(rows, L, 'feasible')
    a_pct, v_pct = sa_series(rows, L, 'sample_fm_pct')
    ok_pct = np.isfinite(v_pct[:, 0])

    for ax in (ax_top, ax_bot):
        ax.axvspan(0.7, 3.8, color='#e5f3ee', lw=0, zorder=0)

    # Top panel: Capture rate (mountain shape peaking at alpha=1..3)
    ax_top.fill_between(a_cap, v_cap[:, 1], v_cap[:, 2], color=S.BAND, lw=0, zorder=1)
    h_cap, = ax_top.plot(a_cap, v_cap[:, 0], label=r'SA capture rate (median, IQR)', zorder=3, **S.SA)
    ax_top.text(0.97, 0.94, f'$L={L}$ ({CANDIDATES[L]:,})', transform=ax_top.transAxes, ha='right', va='top', fontsize=7)
    _alpha_axis(ax_top); ax_top.set_xlabel(''); ax_top.set_ylim(-0.05, 1.08)

    # Bottom panel: Trade-off between Feasibility (low alpha failure) and Sample FM percentile (high alpha flattening)
    ax_bot.fill_between(a_pct[ok_pct], v_pct[ok_pct, 1], v_pct[ok_pct, 2], color='#fadcd2', lw=0, zorder=1)
    h_feas, = ax_bot.plot(a_feas, v_feas[:, 0], color='#2a78d6', marker='s', ls='--', lw=1.1, ms=3.5,
                          mfc='white', mec='#2a78d6', label='Feasible fraction (1 = all valid)', zorder=3)
    h_pct, = ax_bot.plot(a_pct[ok_pct], v_pct[ok_pct, 0], color='#eb6834', marker='o', ls='-', lw=1.1, ms=3.5,
                         mfc='#eb6834', mec='#eb6834', label='Sample FM percentile (0 = best, 0.5 = random)', zorder=4)
    h_rand = ax_bot.axhline(0.5, label='Uniform random percentile (0.5)', zorder=2, **S.RANDOMFM)
    _alpha_axis(ax_bot); ax_bot.set_ylim(-0.05, 1.08)
    return [h_cap, h_feas, h_pct, h_rand]


def fig_sweetspot():
    """Sweet spot of the One-Hot penalty coefficient alpha on frozen FMs:
    (1) penalty_sweetspot: 2x3 grid for L = 6, 7, 8 separately (top: capture rate; bottom: feasibility & sample percentile)
    (2) penalty_sweetspot_combined: 2x1 stacked figure overlaying L = 6, 7, 8 on shared axes
    (3) panels/penalty_sweetspot_L{6,7,8}: independent 2x1 figure for each L
    """
    rows = read('solver_summary.csv')

    # 1. Separate L = 6, 7, 8 in a 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(S.DCOL_W, 4.5), sharey='row', sharex=True)
    for j, L in enumerate(LAYERS):
        handles = _sweetspot_panel_single_L(axes[0, j], axes[1, j], rows, L)
    axes[0, 0].set_ylabel('Unseen FM-min capture rate\n(1 = exact FM minimum found)')
    axes[1, 0].set_ylabel('Feasible fraction / Sample FM pct\n(trade-off across $\\alpha$)')
    S.panel_letters(axes.ravel())
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=6.5, handlelength=2.6,
               columnspacing=1.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(w_pad=0.6, h_pad=0.7, rect=(0, 0.09, 1, 1))
    S.save(fig, HERE, 'penalty_sweetspot')

    # 2. Independent 2x1 panel for each L
    for L in LAYERS:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(S.COL_W, 4.4), sharex=True)
        h = _sweetspot_panel_single_L(ax_top, ax_bot, rows, L)
        ax_top.set_ylabel('Unseen FM-min capture rate')
        ax_bot.set_ylabel('Feasible fraction / Sample FM pct')
        S.panel_letters([ax_top, ax_bot])
        fig.legend(handles=h, loc='lower center', ncol=1, fontsize=6.2, handlelength=2.6, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(h_pad=0.7, rect=(0, 0.16, 1, 1))
        S.save(fig, HERE / 'panels', f'penalty_sweetspot_L{L}')

    # 3. Combined overlay of L = 6, 7, 8 in a 2x1 figure
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(S.COL_W, 4.6), sharex=True)
    for ax in (ax_top, ax_bot):
        ax.axvspan(0.7, 3.8, color='#e5f3ee', lw=0, zorder=0)
    h_layers = []
    for L, ms in zip(LAYERS, (5.0, 4.2, 4.2)):
        a_cap, v_cap = sa_series(rows, L, 'capture')
        a_feas, v_feas = sa_series(rows, L, 'feasible')
        a_pct, v_pct = sa_series(rows, L, 'sample_fm_pct')
        ok_pct = np.isfinite(v_pct[:, 0])
        hl, = ax_top.plot(a_cap, v_cap[:, 0], label=f'$L={L}$ ({CANDIDATES[L]:,})', ls='-', lw=1.1,
                          ms=ms, mfc='none', mew=0.9, zorder=3, **S.LAYER[L])
        h_layers.append(hl)
        ax_bot.plot(a_pct[ok_pct], v_pct[ok_pct, 0], ls='-', lw=1.1, ms=ms, zorder=4, **S.LAYER[L])
        ax_bot.plot(a_feas, v_feas[:, 0], ls=':', lw=0.9, ms=ms, mfc='none', mew=0.8, zorder=3, **S.LAYER[L])
    _alpha_axis(ax_top); ax_top.set_xlabel(''); ax_top.set_ylim(-0.05, 1.08)
    ax_top.set_ylabel('Unseen FM-min capture rate')
    ax_top.legend(handles=h_layers, loc='upper right', fontsize=6.3)

    ax_bot.axhline(0.5, **S.RANDOMFM, zorder=2)
    h_style = [
        plt.Line2D([0], [0], color=S.INK, ls='-', marker='o', ms=3.5, label='Solid: Sample FM pct (0 = best, 0.5 = random)'),
        plt.Line2D([0], [0], color=S.INK, ls=':', marker='o', mfc='none', ms=3.5, label='Dotted: Feasible fraction (1 = all valid)'),
        plt.Line2D([0], [0], **S.RANDOMFM, label='Dashed gray: Uniform random pct (0.5)'),
    ]
    _alpha_axis(ax_bot); ax_bot.set_ylim(-0.05, 1.08)
    ax_bot.set_ylabel('Sample FM pct / Feasible fraction')
    ax_bot.legend(handles=h_style, loc='lower right', bbox_to_anchor=(0.99, 0.54), fontsize=5.8)
    S.panel_letters([ax_top, ax_bot])
    fig.tight_layout(h_pad=0.7)
    S.save(fig, HERE, 'penalty_sweetspot_combined')


DEPTHS = [1, 2, 3, 4, 5, 6]
DEPTH_METRICS = [('norm_gap', 'FM gap of the selected candidate\n(normalized)', (-0.004, 0.105)),
                 ('sample_fm_pct', 'Mean FM percentile of samples\n(0 = FM best, 0.5 = random)', (-0.02, 0.56))]


def _depth_panel(ax, rows, L, metric, ylim):
    h = []
    sa_labels = {
        1: r'SA, $\alpha=1$ (sweet spot)',
        3: r'SA, $\alpha=3$',
        10: r'SA, $\alpha=10$',
        100: r'SA, $\alpha=100$ (flattened)',
    }
    for a in (1, 3, 10, 100):
        h.append(ax.axhline(stat(rows, L, f'SA-a{a:g}', metric)[0], label=sa_labels[a], zorder=1, **S.SA_REF[a]))
    h.append(ax.axhline(stat(rows, L, 'RandomFM', metric)[0], label='Random proposals + FM selection', zorder=1, **S.RANDOMFM))
    for obj, name in (('cvar10', r'CVaR$_{0.1}$'), ('mean', 'mean')):
        v = np.array([stat(rows, L, f'XY-p{p}-{obj}', metric) for p in DEPTHS])
        ax.fill_between(DEPTHS, v[:,1], v[:,2], color=S.XY_OBJECTIVE[obj]['color'], alpha=0.18, lw=0, zorder=0)
        hh, = ax.plot(DEPTHS, v[:,0], label=f'XY-FMQAOA ({name} objective)', zorder=3, **S.XY_OBJECTIVE[obj]); h.append(hh)
    ax.set_xlim(0.6, 6.4); ax.set_xticks(DEPTHS); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_ylim(*ylim)
    ax.set_xlabel('XY-FMQAOA depth $p$')
    return h


def fig_depth_by_layer():
    """Redraw both depth metrics together for each L, using the original summary."""
    rows = read('solver_summary.csv')
    for L in LAYERS:
        fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.8), sharex=True)
        for ax, (metric, ylabel, ylim) in zip(axes, DEPTH_METRICS):
            handles = _depth_panel(ax, rows, L, metric, ylim)
            ax.set_ylabel(ylabel)
        axes[0].set_xlabel('')
        axes[0].text(.97, .95, f'$L={L}$', transform=axes[0].transAxes,
                     ha='right', va='top', fontsize=8)
        S.panel_letters(axes)
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7,
                   handlelength=3.0, bbox_to_anchor=(.5, .005))
        fig.tight_layout(h_pad=.8, rect=(0, .17, 1, 1))
        S.save(fig, HERE, f'xy_depth_L{L}')


def fig_depth():
    """XY-FMQAOA (lambda = 0) vs depth p = 1..6 against SA references, per L. Top: normalized FM gap; bottom: sample FM percentile."""
    rows = read('solver_summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(S.DCOL_W, 4.7), sharey='row', sharex=True)
    for i, (metric, ylabel, ylim) in enumerate(DEPTH_METRICS):
        for j, L in enumerate(LAYERS):
            ax = axes[i, j]; h = _depth_panel(ax, rows, L, metric, ylim)
            if i == 0: ax.text(0.97, 0.95, f'$L={L}$', transform=ax.transAxes, ha='right', va='top', fontsize=7); ax.set_xlabel('')
        axes[i, 0].set_ylabel(ylabel)
    S.panel_letters(axes.ravel())
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=6.8, handlelength=3.2, columnspacing=1.4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.09, 1, 1)); S.save(fig, HERE, 'xy_depth')
    for metric, ylabel, ylim in DEPTH_METRICS:   # independent panels
        for L in LAYERS:
            fig, ax = plt.subplots(figsize=(S.COL_W, 3.1)); h = _depth_panel(ax, rows, L, metric, ylim); ax.set_ylabel(ylabel)
            ax.text(0.97, 0.95, f'$L={L}$', transform=ax.transAxes, ha='right', va='top', fontsize=7)
            fig.legend(handles=h, loc='lower center', ncol=2, fontsize=6.4, handlelength=2.8, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
            fig.tight_layout(rect=(0, 0.26, 1, 1)); S.save(fig, HERE/'panels', f'xy_depth_{metric}_L{L}')

    fig_depth_by_layer()

def _xy_simple_panel(ax, rows, L, metric, ylim):
    # Highlight p=1 simple XY-QAOA region
    ax.axvspan(0.65, 1.35, color='#fdebe4', lw=0, zorder=0)
    h = []
    # Reference horizontal lines
    h.append(ax.axhline(stat(rows, L, 'SA-a1', metric)[0], label=r'SA, $\alpha=1$ (sweet spot)', zorder=1, **S.SA_REF[1]))
    h.append(ax.axhline(stat(rows, L, 'SA-a10', metric)[0], label=r'SA, $\alpha=10$', zorder=1, **S.SA_REF[10]))
    h.append(ax.axhline(stat(rows, L, 'SA-a100', metric)[0], label=r'SA, $\alpha=100$ (flattened)', zorder=1, **S.SA_REF[100]))
    h.append(ax.axhline(stat(rows, L, 'RandomFM', metric)[0], label='Random proposals + FM selection', zorder=1, **S.RANDOMFM))

    # XY depth curves (1024 angle evals): mean vs CVaR_0.1
    for obj, name in (('mean', 'Mean, 1024 evals'), ('cvar10', r'CVaR$_{0.1}$, 1024 evals')):
        v = np.array([stat(rows, L, f'XY-p{p}-{obj}', metric) for p in DEPTHS])
        ax.fill_between(DEPTHS, v[:, 1], v[:, 2], color=S.XY_OBJECTIVE[obj]['color'], alpha=0.18, lw=0, zorder=1)
        hh, = ax.plot(DEPTHS, v[:, 0], label=f'XY-FMQAOA ({name})', zorder=3, **S.XY_OBJECTIVE[obj])
        h.append(hh)

    # Simplest XY-QAOA at p=1: 7x7=49 grid search (Mean & CVaR_0.1)
    g_mean = stat(rows, L, 'XY-p1grid49-mean', metric)
    g_cvar = stat(rows, L, 'XY-p1grid49-cvar10', metric)
    hm, = ax.plot([0.88], [g_mean[0]], marker='D', ms=4.5, ls='none', color='#eb6834', mfc='#eb6834', mec='#000000', mew=0.5,
                  label=r'Simple XY ($p=1$, $7\times 7$ grid, Mean)', zorder=5)
    hc, = ax.plot([1.12], [g_cvar[0]], marker='^', ms=4.5, ls='none', color='#1baf7a', mfc='white', mec='#1baf7a', mew=1.0,
                  label=r'Simple XY ($p=1$, $7\times 7$ grid, CVaR$_{0.1}$)', zorder=5)
    h.extend([hm, hc])

    ax.set_xlim(0.55, 6.4); ax.set_xticks(DEPTHS); ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_ylim(*ylim)
    ax.set_xlabel('XY-FMQAOA circuit depth $p$')
    return h


def fig_xy_simple_failure():
    """Plots showing that simple XY-QAOA (p=1, 49-grid / Mean objective) performs no better than
    random proposals or flattened SA (alpha=100), and how deeper circuits + CVaR_0.1 improve performance:
    (1) xy_simple_failure: 2x3 grid for L = 6, 7, 8 separately
    (2) panels/xy_simple_failure_L{6,7,8}: independent 2x1 figure for each L
    (3) xy_simple_failure_combined: 2x1 stacked figure overlaying L = 6, 7, 8 on shared axes
    """
    rows = read('solver_summary.csv')

    # 1. Separate L = 6, 7, 8 in a 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(S.DCOL_W, 4.7), sharey='row', sharex=True)
    for i, (metric, ylabel, ylim) in enumerate(DEPTH_METRICS):
        for j, L in enumerate(LAYERS):
            ax = axes[i, j]
            h = _xy_simple_panel(ax, rows, L, metric, ylim)
            if i == 0:
                ax.text(0.97, 0.95, f'$L={L}$ ({CANDIDATES[L]:,})', transform=ax.transAxes, ha='right', va='top', fontsize=7)
                ax.text(0.03, 0.95, 'Simple $p=1$\n$\\approx$ random', transform=ax.transAxes, ha='left', va='top',
                        fontsize=5.8, color='#a83810')
                ax.set_xlabel('')
        axes[i, 0].set_ylabel(ylabel)
    S.panel_letters(axes.ravel())
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=6.2, handlelength=2.4,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(w_pad=0.6, h_pad=0.7, rect=(0, 0.09, 1, 1))
    S.save(fig, HERE, 'xy_simple_failure')

    # 2. Independent 2x1 panel for each L
    for L in LAYERS:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(S.COL_W, 4.6), sharex=True)
        h = _xy_simple_panel(ax_top, rows, L, DEPTH_METRICS[0][0], DEPTH_METRICS[0][2])
        _xy_simple_panel(ax_bot, rows, L, DEPTH_METRICS[1][0], DEPTH_METRICS[1][2])
        ax_top.set_ylabel(DEPTH_METRICS[0][1]); ax_top.set_xlabel('')
        ax_top.text(0.97, 0.95, f'$L={L}$ ({CANDIDATES[L]:,})', transform=ax_top.transAxes, ha='right', va='top', fontsize=7)
        ax_top.text(0.03, 0.95, 'Simple $p=1$\n$\\approx$ random', transform=ax_top.transAxes, ha='left', va='top',
                    fontsize=6.0, color='#a83810')
        ax_bot.set_ylabel(DEPTH_METRICS[1][1])
        S.panel_letters([ax_top, ax_bot])
        fig.legend(handles=h, loc='lower center', ncol=2, fontsize=5.8, handlelength=2.2,
                   columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(h_pad=0.7, rect=(0, 0.17, 1, 1))
        S.save(fig, HERE / 'panels', f'xy_simple_failure_L{L}')

    # 3. Combined overlay of L = 6, 7, 8 in a 2x1 figure
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(S.COL_W, 4.8), sharex=True)
    for ax in (ax_top, ax_bot):
        ax.axvspan(0.65, 1.35, color='#fdebe4', lw=0, zorder=0)

    # Shaded band at top for Random / Flattened SA (alpha=100) across L=6,7,8
    rand_gaps = [stat(rows, L, 'RandomFM', 'norm_gap')[0] for L in LAYERS] + [stat(rows, L, 'SA-a100', 'norm_gap')[0] for L in LAYERS]
    ax_top.axhspan(min(rand_gaps), max(rand_gaps), color='#e8e7e3', lw=0, zorder=0)
    ax_top.axhline(0.0, **S.SA_REF[1], zorder=1)
    ax_bot.axhline(0.5, **S.RANDOMFM, zorder=1)
    ax_bot.axhline(stat(rows, 7, 'SA-a1', 'sample_fm_pct')[0], **S.SA_REF[1], zorder=1)

    h_layers = []
    offsets = {6: -0.14, 7: 0.0, 8: 0.14}
    for L, ms in zip(LAYERS, (4.2, 3.8, 4.0)):
        c = S.LAYER[L]['color']
        mk = S.LAYER[L]['marker']
        v_cvar = np.array([stat(rows, L, f'XY-p{p}-cvar10', 'norm_gap')[0] for p in DEPTHS])
        v_mean = np.array([stat(rows, L, f'XY-p{p}-mean', 'norm_gap')[0] for p in DEPTHS])
        g49_gap = stat(rows, L, 'XY-p1grid49-mean', 'norm_gap')[0]

        p_cvar = np.array([stat(rows, L, f'XY-p{p}-cvar10', 'sample_fm_pct')[0] for p in DEPTHS])
        p_mean = np.array([stat(rows, L, f'XY-p{p}-mean', 'sample_fm_pct')[0] for p in DEPTHS])
        g49_pct = stat(rows, L, 'XY-p1grid49-mean', 'sample_fm_pct')[0]

        hl, = ax_top.plot(DEPTHS, v_cvar, color=c, marker=mk, ls='-', lw=1.1, ms=ms, mfc=c, mec=c,
                          label=f'$L={L}$ ({CANDIDATES[L]:,})', zorder=4)
        ax_top.plot(DEPTHS, v_mean, color=c, marker=mk, ls=':', lw=0.9, ms=ms, mfc='white', mec=c, mew=0.8, zorder=3)
        ax_top.plot([1.0 + offsets[L]], [g49_gap], color=c, marker='*', ls='none', ms=6.0, mfc=c, mec='#000000', mew=0.4, zorder=5)
        h_layers.append(hl)

        ax_bot.plot(DEPTHS, p_cvar, color=c, marker=mk, ls='-', lw=1.1, ms=ms, mfc=c, mec=c, zorder=4)
        ax_bot.plot(DEPTHS, p_mean, color=c, marker=mk, ls=':', lw=0.9, ms=ms, mfc='white', mec=c, mew=0.8, zorder=3)
        ax_bot.plot([1.0 + offsets[L]], [g49_pct], color=c, marker='*', ls='none', ms=6.0, mfc=c, mec='#000000', mew=0.4, zorder=5)

    ax_top.text(6.28, 0.064, 'Gray band: Random / SA $\\alpha=100$', ha='right', va='center', fontsize=5.8, color=S.INK)
    ax_top.text(0.98, 0.003, 'SA $\\alpha=1$ (0.000)', ha='right', va='bottom', fontsize=5.8, color=S.INK,
                transform=ax_top.get_yaxis_transform())
    ax_top.text(0.03, 0.96, 'Simple XY ($p=1$)\n$\\approx$ random', transform=ax_top.transAxes,
                ha='left', va='top', fontsize=6.0, color='#a83810')
    ax_top.set_xlim(0.55, 6.4); ax_top.set_xticks(DEPTHS); ax_top.xaxis.set_minor_locator(plt.NullLocator())
    ax_top.set_ylim(-0.005, 0.112); ax_top.set_ylabel(DEPTH_METRICS[0][1]); ax_top.set_xlabel('')
    ax_top.legend(handles=h_layers, loc='upper right', fontsize=6.0)

    ax_bot.text(0.98, 0.508, 'Uniform random / SA $\\alpha=100$ ($\\approx 0.5$)', ha='right', va='bottom', fontsize=5.8,
                color=S.MUTED, transform=ax_bot.get_yaxis_transform())
    ax_bot.text(0.98, 0.018, 'SA $\\alpha=1$ (sweet spot, $\\approx 0.00$)', ha='right', va='bottom', fontsize=5.8,
                color=S.INK, transform=ax_bot.get_yaxis_transform())
    h_style = [
        plt.Line2D([0], [0], color=S.INK, marker='*', ls='none', ms=5.5, mfc=S.INK, mec='#000000', mew=0.4,
                   label=r'Star ($p=1$): Simple XY ($7\times 7$ grid, Mean)'),
        plt.Line2D([0], [0], color=S.INK, marker='o', ls=':', ms=3.5, mfc='white', mec=S.INK,
                   label='Dotted (open): XY Mean (1024 evals)'),
        plt.Line2D([0], [0], color=S.INK, marker='o', ls='-', ms=3.5, mfc=S.INK, mec=S.INK,
                   label=r'Solid (filled): XY CVaR$_{0.1}$ (1024 evals)'),
    ]
    ax_bot.set_xlim(0.55, 6.4); ax_bot.set_xticks(DEPTHS); ax_bot.xaxis.set_minor_locator(plt.NullLocator())
    ax_bot.set_ylim(-0.02, 0.57); ax_bot.set_ylabel(DEPTH_METRICS[1][1])
    ax_bot.set_xlabel('XY-FMQAOA circuit depth $p$')
    ax_bot.legend(handles=h_style, loc='lower left', bbox_to_anchor=(0.02, 0.08), fontsize=5.7)

    S.panel_letters([ax_top, ax_bot])
    fig.tight_layout(h_pad=0.7)
    S.save(fig, HERE, 'xy_simple_failure_combined')


def draw_all():
    for obj in ('cvar10', 'mean'):
        fig_flattening(obj); fig_gap(obj)
    fig_mechanism(); fig_feasibility(); fig_sweetspot(); fig_depth(); fig_xy_simple_failure()
    print('figures written to', HERE)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replot', action='store_true', help='skip rebuilding the CSVs')
    if not ap.parse_args().replot: build_csv()
    draw_all()
