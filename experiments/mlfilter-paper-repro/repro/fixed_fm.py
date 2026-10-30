"""Frozen-FM schedule robustness and FM-percentile distribution runner (Suite 1)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import atomic_write_json, utc_now
from .optics import labels_for, onehot
from .solvers import create_xy_backend, interp_ladder_search, sample_random_feasible, sample_sa_schedule
from .surrogate import bits_to_ids, fit_fm, initial_ids, model_hash, predict_all, select_candidate


class FrozenFMContext:
    """Shared FM rank and percentile statistics across samplers on one frozen FM."""

    def __init__(self, pred: np.ndarray, seen: list[int], layers: int) -> None:
        self.pred = pred
        self.layers = layers
        self.seen = set(seen)
        mask = np.ones(len(pred), dtype=bool)
        mask[list(seen)] = False
        self.unseen_min = float(pred[mask].min())
        self.unseen_max = float(pred[mask].max())

        order = np.argsort(pred, kind="stable")
        pct = np.empty(len(pred), dtype=np.float64)
        pct[order] = np.arange(len(pred)) / (len(pred) - 1)
        self.pct = pct

        un = np.flatnonzero(mask)
        un_order = un[np.argsort(pred[un], kind="stable")]
        self.unseen_rank = np.full(len(pred), -1, dtype=np.int64)
        self.unseen_rank[un_order] = np.arange(len(un))
        self.n_unseen = len(un)

    def summarize_batch(self, raw: np.ndarray, x_all: np.ndarray) -> tuple[dict, list[float]]:
        chosen, filters = select_candidate(raw, self.pred, x_all, self.seen, self.layers)
        valid_ids = [i for i in bits_to_ids(raw, self.layers) if i is not None]
        span = max(self.unseen_max - self.unseen_min, 1e-300)
        pct_samples = self.pct[valid_ids].tolist() if valid_ids else []
        metrics = {
            "selected_id": chosen,
            "capture": bool(chosen is not None and abs(self.pred[chosen] - self.unseen_min) < 1e-8),
            "norm_gap": 1.0 if chosen is None else float((self.pred[chosen] - self.unseen_min) / span),
            "fm_rank_pct": (
                1.0 if chosen is None else float(self.unseen_rank[chosen] / max(self.n_unseen - 1, 1))
            ),
            "feasible": filters["raw_feasible_rate"],
            "unique_unseen": filters["unique_unseen_count"],
            "sample_fm_pct": float(np.mean(pct_samples)) if pct_samples else None,
        }
        return metrics, pct_samples


def run_fixed_fm_seed(
    layers: int,
    seed: int,
    truth: np.ndarray,
    *,
    initial_points: int = 40,
    reps: int = 10,
    schedules: list[str],
    sa_alphas: list[float],
    xy_cfg: dict,
    surrogate_cfg: dict,
    sa_cfg: dict,
    xy_backend_name: str = "auto",
) -> tuple[dict, dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    x_all = onehot(labels_for(layers))
    init_ids = initial_ids(seed, layers, count=initial_points)
    ys = [float(truth[i]) for i in init_ids]

    m, qd, q, b, s = fit_fm(
        x_all[init_ids],
        ys,
        seed,
        rank=surrogate_cfg["rank"],
        epochs=surrogate_cfg["epochs"],
        learning_rate=surrogate_cfg["learning_rate"],
        optimizer_name=surrogate_cfg["optimizer"],
        weight_decay=surrogate_cfg["weight_decay"],
    )
    pred = predict_all(x_all, q, b)
    ctx = FrozenFMContext(pred, init_ids, layers)

    num_reads = int(sa_cfg["num_reads"])
    num_sweeps = int(sa_cfg["num_sweeps"])
    batches: list[dict] = []
    dists: dict[str, list[float]] = {}

    def _record(label: str, raw: np.ndarray, extra: dict) -> None:
        metrics, pct_vals = ctx.summarize_batch(raw, x_all)
        batches.append({"sampler": label, **extra, **metrics})
        dists.setdefault(label, []).extend(pct_vals)

    # 1. SA across schedules x alphas
    for sch in schedules:
        for alpha in sa_alphas:
            for rep in range(reps):
                t0 = time.perf_counter()
                raw, sm = sample_sa_schedule(
                    qd,
                    b,
                    s,
                    float(alpha),
                    layers,
                    seed * 100000 + rep,
                    schedule=sch,
                    num_reads=num_reads,
                    num_sweeps=num_sweeps,
                )
                _record(
                    f"SA-{sch}-a{alpha:g}",
                    raw,
                    {
                        "alpha": alpha,
                        "schedule": sch,
                        "rep": rep,
                        "seconds": time.perf_counter() - t0,
                        "beta_range": sm["beta_range"],
                        "num_sweeps": sm["num_sweeps"],
                    },
                )

    # 2. RandomFM
    for rep in range(reps):
        raw, _ = sample_random_feasible(layers, seed * 100000 + rep, num_reads=num_reads)
        _record("RandomFM", raw, {"rep": rep})

    # 3. XY-FMQAOA (INTERP ladder to depth p, e.g. p=4 CVaR_0.1)
    depth = int(xy_cfg["depth"])
    objective = xy_cfg.get("objective", "cvar10")
    evals_per_depth = int(xy_cfg["evals_per_depth"])
    t_xy = time.perf_counter()
    backend = create_xy_backend(q / max(s, 1e-30), [4] * layers, xy_backend_name)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, 2020, depth]))
    best_angles, best_obj, prob, total_evals = interp_ladder_search(
        backend, pred, s, depth, objective, evals_per_depth, rng
    )
    xy_search_seconds = time.perf_counter() - t_xy

    cdf = np.cumsum(prob)
    # Always record under canonical label 'XY-p4-cvar10' so plotting schema stays uniform
    xy_label = "XY-p4-cvar10"
    for rep in range(reps):
        draws = np.minimum(
            np.searchsorted(
                cdf,
                np.random.default_rng(seed * 100000 + rep).random(num_reads) * cdf[-1],
                side="left",
            ),
            len(prob) - 1,
        )
        _record(xy_label, x_all[draws], {"rep": rep, "configured_depth": depth})

    summary = {
        "status": "completed",
        "layers": layers,
        "seed": seed,
        "model_sha256": model_hash(m),
        "S": s,
        "xy_search": {
            "label": xy_label,
            "configured_depth": depth,
            "objective": objective,
            "evals_per_depth": evals_per_depth,
            "total_evaluations": total_evals,
            "best_objective": best_obj,
            "best_angles": best_angles.tolist(),
            "backend": type(backend).__name__,
            "seconds": xy_search_seconds,
        },
        "batches": batches,
        "completed_utc": utc_now(),
    }
    npz_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in dists.items()}
    return summary, npz_arrays


def run_fixed_fm_task(task: dict) -> str:
    out_json = Path(task["out_json"])
    out_npz = Path(task["out_npz"])
    if out_json.exists() and out_npz.exists():
        try:
            existing = json.loads(out_json.read_text())
            if existing.get("status") == "completed":
                return f"skip {out_json.name}"
        except Exception:
            pass

    truth = np.load(task["truth_npz"])["fom"]
    summary, npz_arrays = run_fixed_fm_seed(
        layers=task["layers"],
        seed=task["seed"],
        truth=truth,
        initial_points=task["initial_points"],
        reps=task["reps"],
        schedules=task["schedules"],
        sa_alphas=task["sa_alphas"],
        xy_cfg=task["xy_cfg"],
        surrogate_cfg=task["surrogate_cfg"],
        sa_cfg=task["sa_cfg"],
        xy_backend_name=task["xy_backend"],
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out_npz.with_name(out_npz.stem + "_tmp.npz")
    np.savez_compressed(tmp_npz, **npz_arrays)
    tmp_npz.replace(out_npz)
    atomic_write_json(out_json, summary)
    return f"done {out_json.name}"

