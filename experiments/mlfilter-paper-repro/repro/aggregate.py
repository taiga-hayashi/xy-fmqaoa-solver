"""Aggregate raw JSON / NPZ outputs from Suites 0..4 into clean CSVs for plotting."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import sha256_file

MAIN_CONDITIONS = ["XY-FMQAOA", "alpha1", "alpha100", "alpha0.3", "RandomFM", "Random"]


def _write_csv(csv_dir: Path, name: str, rows: list[dict]) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_dir / name, "w", newline="") as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_ranks(truth_npz: Path) -> np.ndarray:
    fom = np.load(truth_npz)["fom"]
    order = np.argsort(fom, kind="stable")
    ranks = np.empty(len(fom), dtype=np.int64)
    ranks[order] = np.arange(1, len(fom) + 1)
    return ranks


def _closed_loop_curve(run_file: Path, ranks: np.ndarray, budget: int) -> tuple[dict, np.ndarray]:
    data = json.loads(run_file.read_text())
    ids = data["final_ids"][:budget]
    best = np.minimum.accumulate(ranks[ids])
    if len(best) < budget:
        best = np.concatenate([best, np.full(budget - len(best), best[-1], dtype=best.dtype)])
    return data, best


def build_all_csvs(run_root: Path, csv_dir: Path) -> list[str]:
    """Read raw suite outputs under `run_root` and write plotting CSVs into `csv_dir`."""
    run_root = Path(run_root)
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: set[Path] = set()
    written: list[str] = []

    truth_dir = run_root / "truth"
    rank_cache: dict[int, np.ndarray] = {}
    for L in (6, 7, 8):
        tp = truth_dir / f"L{L}_truth.npz"
        if tp.exists():
            rank_cache[L] = _load_ranks(tp)
            manifest_files.add(tp)

    # ------------------------------------------------------------------
    # Suite 1: Fixed-FM schedules & FM-percentile histograms
    # ------------------------------------------------------------------
    fixed_dir = run_root / "fixed_fm"
    if fixed_dir.exists():
        fixed_rows: list[dict] = []
        hist_rows: list[dict] = []
        bins = np.linspace(0.0, 1.0, 21)
        for jf in sorted(fixed_dir.glob("L*_seed*.json")):
            d = json.loads(jf.read_text())
            if d.get("status") != "completed":
                continue
            manifest_files.add(jf)
            grouped: dict[str, list[dict]] = {}
            for row in d["batches"]:
                grouped.setdefault(row["sampler"], []).append(row)
            for smp, rows in grouped.items():
                def _mean_or_none(k: str) -> float | None:
                    vals = [r[k] for r in rows if r.get(k) is not None]
                    return float(np.mean(vals)) if vals else None

                fixed_rows.append(
                    {
                        "layers": d["layers"],
                        "seed": d["seed"],
                        "sampler": smp,
                        "sample_fm_pct": _mean_or_none("sample_fm_pct"),
                        "norm_gap": _mean_or_none("norm_gap"),
                        "feasible": _mean_or_none("feasible"),
                    }
                )
            npz_file = jf.with_name(jf.stem + "_pct.npz")
            if npz_file.exists():
                manifest_files.add(npz_file)
                pz = np.load(npz_file)
                for key in pz.files:
                    counts, _ = np.histogram(pz[key], bins=bins)
                    for i in range(len(counts)):
                        hist_rows.append(
                            {
                                "layers": d["layers"],
                                "seed": d["seed"],
                                "sampler": key,
                                "bin_lo": float(bins[i]),
                                "bin_hi": float(bins[i + 1]),
                                "count": int(counts[i]),
                            }
                        )
        if fixed_rows:
            _write_csv(csv_dir, "fixedfm_schedules.csv", fixed_rows)
            written.append("fixedfm_schedules.csv")
        if hist_rows:
            _write_csv(csv_dir, "energy_hist.csv", hist_rows)
            written.append("energy_hist.csv")

    # ------------------------------------------------------------------
    # Suite 2: Confirm 1 + Alpha Sweep + Retuned Closed Loop (L = 7)
    # ------------------------------------------------------------------
    c1_dir = run_root / "confirm1"
    if c1_dir.exists() and 7 in rank_cache:
        rank7 = rank_cache[7]
        finals: list[dict] = []
        mech: list[dict] = []
        cost: list[dict] = []
        loop_sweep: list[dict] = []

        for jf in sorted(c1_dir.glob("*_seed*.json")):
            raw_obj = json.loads(jf.read_text())
            if raw_obj.get("status") != "completed":
                continue
            manifest_files.add(jf)
            cid = raw_obj["condition"]["id"]
            seed = int(raw_obj["seed"])
            budget = int(raw_obj["budget"])
            d, best = _closed_loop_curve(jf, rank7, budget)
            final_rk = int(best[-1])
            src = "confirm" if cid in MAIN_CONDITIONS else "paper-extra"
            loop_sweep.append(
                {"condition": cid, "seed": seed, "best_rank": final_rk, "source": src}
            )
            if cid in MAIN_CONDITIONS:
                finals.append(
                    {
                        "condition": cid,
                        "seed": seed,
                        "best_rank": final_rk,
                        "evaluations_used": len(d["final_ids"]),
                    }
                )
                for ev in d.get("events", []):
                    sel = ev["selected_id"]
                    mech.append(
                        {
                            "condition": cid,
                            "seed": seed,
                            "cycle": ev["cycle"],
                            "selected_true_rank": None if sel is None else int(rank7[sel]),
                            "unseen_fm_min_hit": int(ev["unseen_FM_min_hit"]),
                            "unique_new_candidates": ev["unique_unseen_count"],
                            "feasible_rate": ev["raw_feasible_rate"],
                        }
                    )
                    cost.append(
                        {
                            "condition": cid,
                            "seed": seed,
                            "cycle": ev["cycle"],
                            "training_s": ev["training_seconds"],
                            "proposal_s": ev["sampling_seconds"],
                            "angle_evaluations": ev.get("angle_evaluations"),
                        }
                    )

        # Sort deterministically by MAIN_CONDITIONS order, then seed
        cond_order = {c: idx for idx, c in enumerate(MAIN_CONDITIONS)}
        finals.sort(key=lambda r: (cond_order.get(r["condition"], 99), r["seed"]))
        mech.sort(key=lambda r: (cond_order.get(r["condition"], 99), r["seed"], r["cycle"]))
        loop_sweep.sort(key=lambda r: (r["condition"], r["seed"]))

        if finals:
            _write_csv(csv_dir, "confirm_final_ranks.csv", finals)
            written.append("confirm_final_ranks.csv")
        if mech:
            _write_csv(csv_dir, "confirm_mechanism_per_cycle.csv", mech)
            written.append("confirm_mechanism_per_cycle.csv")
        if cost:
            agg_cost: dict[str, list[dict]] = {}
            for r in cost:
                agg_cost.setdefault(r["condition"], []).append(r)
            cost_rows = []
            for c in [k for k in MAIN_CONDITIONS if k in agg_cost]:
                v = agg_cost[c]
                later = [x["proposal_s"] for x in v if x["cycle"] > 1]
                later_evals = [
                    x["angle_evaluations"]
                    for x in v
                    if x["cycle"] > 1 and x["angle_evaluations"] is not None
                ]
                cost_rows.append(
                    {
                        "condition": c,
                        "cycles": len(v),
                        "median_proposal_seconds": float(np.median([x["proposal_s"] for x in v])),
                        "first_cycle_proposal_seconds_median": float(
                            np.median([x["proposal_s"] for x in v if x["cycle"] == 1])
                        ),
                        "later_cycles_proposal_seconds_median": (
                            float(np.median(later)) if later else None
                        ),
                        "median_training_seconds": float(np.median([x["training_s"] for x in v])),
                        "median_angle_evaluations_per_cycle": (
                            float(np.median(later_evals)) if later_evals else None
                        ),
                    }
                )
            _write_csv(csv_dir, "compute_cost.csv", cost_rows)
            written.append("compute_cost.csv")
        if loop_sweep:
            _write_csv(csv_dir, "loop_alpha_sweep.csv", loop_sweep)
            written.append("loop_alpha_sweep.csv")

    # ------------------------------------------------------------------
    # Suite 3: Confirm 2 (L = 7, 30 seeds, Boltzmann excluded)
    # ------------------------------------------------------------------
    c2_dir = run_root / "confirm2"
    if c2_dir.exists() and 7 in rank_cache:
        rank7 = rank_cache[7]
        c2_fin: list[dict] = []
        c2_cur: list[dict] = []
        for jf in sorted(c2_dir.glob("*_seed*.json")):
            raw_obj = json.loads(jf.read_text())
            if raw_obj.get("status") != "completed":
                continue
            manifest_files.add(jf)
            cid = raw_obj["condition"]["id"]
            if cid.startswith("Boltzmann"):
                continue
            seed = int(raw_obj["seed"])
            budget = int(raw_obj["budget"])
            init_pts = len(raw_obj["initial_ids"])
            _, best = _closed_loop_curve(jf, rank7, budget)
            c2_fin.append(
                {
                    "condition": cid,
                    "seed": seed,
                    "best_rank": int(best[-1]),
                    "success": int(best[-1] == 1),
                }
            )
            for n_eval in range(init_pts, budget + 1):
                c2_cur.append(
                    {
                        "condition": cid,
                        "seed": seed,
                        "evaluations": n_eval,
                        "best_rank": int(best[n_eval - 1]),
                    }
                )
        if c2_fin:
            _write_csv(csv_dir, "confirm2_final.csv", c2_fin)
            written.append("confirm2_final.csv")
        if c2_cur:
            _write_csv(csv_dir, "confirm2_curves.csv", c2_cur)
            written.append("confirm2_curves.csv")

    # ------------------------------------------------------------------
    # Suite 4: Multi-Layer Exploratory Closed-Loop (L = 6, 7, 8)
    # ------------------------------------------------------------------
    ex_dir = run_root / "explore_L678"
    if ex_dir.exists():
        ex_rows: list[dict] = []
        for jf in sorted(ex_dir.glob("L*_*.json")):
            raw_obj = json.loads(jf.read_text())
            if raw_obj.get("status") != "completed":
                continue
            L = int(raw_obj["layers"])
            if L not in rank_cache:
                continue
            manifest_files.add(jf)
            cid = raw_obj["condition"]["id"]
            seed = int(raw_obj["seed"])
            budget = int(raw_obj["budget"])
            init_pts = len(raw_obj["initial_ids"])
            _, best = _closed_loop_curve(jf, rank_cache[L], budget)
            for n_eval in range(init_pts, budget + 1):
                ex_rows.append(
                    {
                        "layers": L,
                        "condition": cid,
                        "seed": seed,
                        "evaluations": n_eval,
                        "best_rank": int(best[n_eval - 1]),
                    }
                )
        if ex_rows:
            _write_csv(csv_dir, "explore_curves.csv", ex_rows)
            written.append("explore_curves.csv")

    manifest_dict = {
        str(p.relative_to(run_root)): sha256_file(p) for p in sorted(manifest_files)
    }
    (csv_dir / "input_manifest.json").write_text(json.dumps(manifest_dict, indent=2) + "\n")
    return written

