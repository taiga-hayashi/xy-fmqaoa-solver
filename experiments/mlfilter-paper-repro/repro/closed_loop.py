"""Unified closed-loop FMQA / XY-FMQAOA / CatSA / baseline runner."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import atomic_write_json, utc_now
from .optics import labels_for, onehot
from .solvers import (
    AdaptivePenaltyTracker,
    angle_search,
    create_xy_backend,
    interp_ladder_search,
    random_search_trajectory,
    sample_categorical_sa,
    sample_random_feasible,
    sample_sa_schedule,
)
from .surrogate import fit_fm, initial_ids, model_hash, predict_all, select_candidate


def run_closed_loop_trajectory(
    seed: int,
    condition: dict,
    layers: int,
    truth: np.ndarray,
    *,
    budget: int = 100,
    initial_points: int = 40,
    cycle_cap: int = 60,
    surrogate_cfg: dict | None = None,
    sa_cfg: dict | None = None,
    adaptive_cfg: dict | None = None,
    xy_backend_name: str = "auto",
    random_stream_offset: int = 73101,
    x_all: np.ndarray | None = None,
) -> dict:
    """Execute one closed-loop BBO trajectory for any supported proposal solver."""
    torch.set_num_threads(1)
    sur_cfg = {"rank": 2, "optimizer": "AdamW", "weight_decay": 0.01, "learning_rate": 0.1, "epochs": 120}
    if surrogate_cfg:
        sur_cfg.update(surrogate_cfg)
    sa_params = {"num_reads": 100, "num_sweeps": 1000}
    if sa_cfg:
        sa_params.update({k: sa_cfg[k] for k in ("num_reads", "num_sweeps") if k in sa_cfg})
    ad_params = {"initial_alpha": 1.0, "min_alpha": 0.01, "max_alpha": 100.0}
    if adaptive_cfg:
        ad_params.update(adaptive_cfg)

    kind = condition["kind"]
    if kind == "random":
        return random_search_trajectory(
            seed,
            layers,
            truth,
            budget,
            initial_points=initial_points,
            random_stream_offset=random_stream_offset,
        )

    if x_all is None:
        x_all = onehot(labels_for(layers))
    ids = initial_ids(seed, layers, count=initial_points)
    ys = [float(truth[i]) for i in ids]
    tracker = AdaptivePenaltyTracker(**ad_params)
    events: list[dict] = []
    angles: np.ndarray | None = None
    start = time.perf_counter()

    for cycle in range(1, cycle_cap + 1):
        if len(ids) >= budget:
            break
        model_seed = seed + 100000 * (cycle - 1)
        t_train = time.perf_counter()
        m, qd, q, b, s = fit_fm(
            x_all[ids],
            ys,
            model_seed,
            rank=sur_cfg["rank"],
            epochs=sur_cfg["epochs"],
            learning_rate=sur_cfg["learning_rate"],
            optimizer_name=sur_cfg["optimizer"],
            weight_decay=sur_cfg["weight_decay"],
        )
        training_seconds = time.perf_counter() - t_train
        pred = predict_all(x_all, q, b)
        sampler_seed = seed * 100000 + (cycle - 1) * 1000

        t_samp = time.perf_counter()
        extra_meta: dict = {}
        if kind == "randomfm":
            raw, extra_meta = sample_random_feasible(
                layers, sampler_seed, num_reads=sa_params["num_reads"]
            )
        elif kind == "catsa":
            raw, extra_meta = sample_categorical_sa(
                pred,
                layers,
                sampler_seed,
                num_reads=sa_params["num_reads"],
                num_sweeps=sa_params["num_sweeps"],
            )
        elif kind in ("sa", "adaptive"):
            alpha = tracker.get_current_alpha() if kind == "adaptive" else float(condition["alpha"])
            schedule = condition.get("schedule", "auto")
            raw, extra_meta = sample_sa_schedule(
                qd,
                b,
                s,
                alpha,
                layers,
                sampler_seed,
                schedule=schedule,
                num_reads=sa_params["num_reads"],
                num_sweeps=sa_params["num_sweeps"],
            )
        elif kind == "xy":
            depth = int(condition["depth"])
            objective = condition.get("objective", "cvar10")
            backend = create_xy_backend(q / max(s, 1e-30), [4] * layers, xy_backend_name)
            rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 7, 2026]))
            if angles is None:
                a_prev, obj, prob, evals = interp_ladder_search(
                    backend,
                    pred,
                    s,
                    depth,
                    objective,
                    int(condition["first_evals"]),
                    rng,
                )
            else:
                a_prev, obj, prob, evals = angle_search(
                    backend,
                    pred,
                    s,
                    depth,
                    objective,
                    [angles],
                    int(condition["cycle_evals"]),
                    rng,
                    n_global=1,
                    n_local=1,
                    local_maxfev=int(condition["local_maxfev"]),
                )
            angles = a_prev
            cdf = np.cumsum(prob)
            draws = np.minimum(
                np.searchsorted(
                    cdf,
                    np.random.default_rng(sampler_seed).random(sa_params["num_reads"]) * cdf[-1],
                    side="left",
                ),
                len(prob) - 1,
            )
            raw = x_all[draws]
            unseen_mask_xy = np.ones(len(pred), dtype=bool)
            unseen_mask_xy[ids] = False
            unseen_min_xy = float(pred[unseen_mask_xy].min())
            targets_xy = np.flatnonzero(unseen_mask_xy & (np.abs(pred - unseen_min_xy) < 1e-8))
            extra_meta = {
                "sampler": "XY-FMQAOA",
                "backend": type(backend).__name__,
                "depth": depth,
                "objective": objective,
                "angles": angles.tolist(),
                "angle_objective": obj,
                "angle_evaluations": evals,
                "P_unseen_min": float(prob[targets_xy].sum()),
                "sampler_seed": sampler_seed,
            }
        else:
            raise ValueError(f"Unsupported condition kind: {kind}")

        sampling_seconds = time.perf_counter() - t_samp
        seen = set(ids)
        chosen, filters = select_candidate(raw, pred, x_all, seen, layers)
        unseen_mask = np.ones(len(pred), dtype=bool)
        unseen_mask[ids] = False
        unseen_min = float(pred[unseen_mask].min())

        event = {
            "cycle": cycle,
            "model_seed": model_seed,
            "model_sha256": model_hash(m),
            "S_fm": s,
            **extra_meta,
            **filters,
            "selected_id": chosen,
            "selected_value": None,
            "selected_FM_gap": None if chosen is None else float(pred[chosen] - unseen_min),
            "unseen_FM_min_hit": bool(chosen is not None and abs(pred[chosen] - unseen_min) < 1e-8),
            "evaluations_before": len(ids),
            "training_seconds": training_seconds,
            "sampling_seconds": sampling_seconds,
        }
        if kind == "adaptive":
            event["next_alpha"] = tracker.update(filters["raw_feasible_rate"])
        if chosen is not None:
            assert chosen not in seen and len(ids) < budget
            val = float(truth[chosen])
            ids.append(chosen)
            ys.append(val)
            event["selected_value"] = val
        event["evaluations_after"] = len(ids)
        event["best_so_far"] = float(min(ys))
        events.append(event)

    return {
        "seed": seed,
        "condition": condition,
        "layers": layers,
        "budget": budget,
        "cycle_cap": cycle_cap,
        "initial_ids": ids[:initial_points],
        "final_ids": ids,
        "final_values": ys,
        "actual_evaluations": len(ids),
        "final_best": float(min(ys)),
        "events": events,
        "total_seconds": time.perf_counter() - start,
        "stopping_reason": "evaluation_cap" if len(ids) >= budget else "proposal_cycle_cap",
    }


def run_closed_loop_task(task: dict) -> str:
    """Worker-friendly wrapper that checkpoints each (condition, seed) JSON atomically."""
    out_path = Path(task["out_path"])
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("status") == "completed":
                return f"skip {out_path.name}"
        except Exception:
            pass

    truth = np.load(task["truth_npz"])["fom"]
    res = run_closed_loop_trajectory(
        seed=task["seed"],
        condition=task["condition"],
        layers=task["layers"],
        truth=truth,
        budget=task["budget"],
        initial_points=task["initial_points"],
        cycle_cap=task["cycle_cap"],
        surrogate_cfg=task["surrogate_cfg"],
        sa_cfg=task["sa_cfg"],
        adaptive_cfg=task["adaptive_cfg"],
        xy_backend_name=task["xy_backend"],
        random_stream_offset=task["random_stream_offset"],
    )
    res["status"] = "completed"
    res["completed_utc"] = utc_now()
    atomic_write_json(out_path, res)
    return f"done {out_path.name} (best={res['final_best']:.4f}, evals={res['actual_evaluations']})"

