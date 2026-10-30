"""Unified CLI runner for reproducing all 8 figures in `analysis/mlfilter-scale/paper/pdf`.

Examples:
    # Fast end-to-end smoke verification (~30-60s on laptop, generates all 8 PDFs)
    python run_pipeline.py --suite all --smoke --workers 4

    # Full production reproduction (resumable per-seed checkpoints, multi-core parallel)
    python run_pipeline.py --suite all --workers 16 --backend auto
"""
from __future__ import annotations

import os

# Prevent OpenMP / BLAS thread oversubscription on multi-core Linux servers when using --workers > 1
for _env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from plot_paper import draw_all_figures
from repro import atomic_write_json, default_output_dirs, utc_now
from repro.aggregate import build_all_csvs
from repro.closed_loop import run_closed_loop_task
from repro.fixed_fm import run_fixed_fm_task
from repro.optics import ensure_truth_table


def load_effective_config(smoke: bool) -> dict:
    cfg = json.loads((STUDY_DIR / "config.json").read_text())
    if not smoke:
        return cfg
    cfg = copy.deepcopy(cfg)
    sm = cfg["smoke"]
    cfg["surrogate"].update(sm["surrogate"])
    cfg["sa"].update(sm["sa"])

    # Apply suite-level smoke overrides
    cfg["suites"]["fixed_fm"]["seeds"] = sm["fixed_fm"]["seeds"]
    cfg["suites"]["fixed_fm"]["reps"] = sm["fixed_fm"]["reps"]
    cfg["suites"]["fixed_fm"]["xy"].update(sm["fixed_fm"]["xy"])

    for sname in ("confirm1", "confirm2", "explore_L678"):
        sc = cfg["suites"][sname]
        ov = sm[sname]
        sc["seeds"] = ov["seeds"]
        sc["budget"] = ov["budget"]
        sc["initial_points"] = ov["initial_points"]
        sc["proposal_cycle_cap"] = ov["proposal_cycle_cap"]
        for cond in sc["conditions"]:
            if cond["kind"] == "xy":
                cond.update(ov["xy_override"])
    return cfg


def _execute_tasks(fn, tasks: list[dict], workers: int, label: str) -> None:
    if not tasks:
        return
    print(f"[{label}] Running {len(tasks)} tasks (workers={workers})...", flush=True)
    t0 = time.perf_counter()
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            msg = fn(task)
            print(f"  [{i}/{len(tasks)}] {msg}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fn, t): t for t in tasks}
            done_cnt = 0
            for fut in as_completed(futs):
                done_cnt += 1
                msg = fut.result()
                print(f"  [{done_cnt}/{len(tasks)}] {msg}", flush=True)
    print(f"[{label}] Completed in {time.perf_counter() - t0:.1f}s.", flush=True)


def run_suite_truth(cfg: dict, run_dir: Path, tmm_backend: str) -> None:
    for L in cfg["optics"]["layers_list"]:
        t0 = time.perf_counter()
        fom, _ = ensure_truth_table(L, run_dir, tmm_backend=tmm_backend)
        print(
            f"[truth] L={L} ready ({len(fom)} states, min FOM={fom.min():.6f}, {time.perf_counter() - t0:.2f}s)",
            flush=True,
        )


def run_suite_fixed_fm(cfg: dict, run_dir: Path, xy_backend: str, workers: int) -> None:
    sc = cfg["suites"]["fixed_fm"]
    out_sub = run_dir / "fixed_fm"
    out_sub.mkdir(parents=True, exist_ok=True)
    tasks = []
    for L in sc["layers_list"]:
        truth_npz = run_dir / "truth" / f"L{L}_truth.npz"
        for seed in sc["seeds"]:
            tasks.append(
                {
                    "layers": L,
                    "seed": seed,
                    "truth_npz": str(truth_npz),
                    "initial_points": sc["initial_points"],
                    "reps": sc["reps"],
                    "schedules": sc["schedules"],
                    "sa_alphas": sc["sa_alphas"],
                    "xy_cfg": sc["xy"],
                    "surrogate_cfg": cfg["surrogate"],
                    "sa_cfg": cfg["sa"],
                    "xy_backend": xy_backend,
                    "out_json": str(out_sub / f"L{L}_seed{seed}.json"),
                    "out_npz": str(out_sub / f"L{L}_seed{seed}_pct.npz"),
                }
            )
    _execute_tasks(run_fixed_fm_task, tasks, workers, "fixed_fm")


def run_suite_closed_loop(
    suite_name: str,
    cfg: dict,
    run_dir: Path,
    xy_backend: str,
    workers: int,
) -> None:
    sc = cfg["suites"][suite_name]
    out_sub = run_dir / suite_name
    out_sub.mkdir(parents=True, exist_ok=True)
    layers_list = sc["layers_list"] if "layers_list" in sc else [sc["layers"]]
    multi_layer = "layers_list" in sc

    tasks = []
    for L in layers_list:
        truth_npz = run_dir / "truth" / f"L{L}_truth.npz"
        for cond in sc["conditions"]:
            for seed in sc["seeds"]:
                fname = (
                    f"L{L}_{cond['id']}_seed{seed}.json"
                    if multi_layer
                    else f"{cond['id']}_seed{seed}.json"
                )
                tasks.append(
                    {
                        "layers": L,
                        "seed": seed,
                        "condition": cond,
                        "truth_npz": str(truth_npz),
                        "budget": sc["budget"],
                        "initial_points": sc["initial_points"],
                        "cycle_cap": sc["proposal_cycle_cap"],
                        "surrogate_cfg": cfg["surrogate"],
                        "sa_cfg": cfg["sa"],
                        "adaptive_cfg": cfg["adaptive"],
                        "xy_backend": xy_backend,
                        "random_stream_offset": cfg["random_stream_offset"],
                        "out_path": str(out_sub / fname),
                    }
                )
    _execute_tasks(run_closed_loop_task, tasks, workers, suite_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--suite",
        choices=["all", "truth", "fixed_fm", "confirm1", "confirm2", "explore_L678", "plot"],
        default="all",
        help="Experiment suite to run (default: all).",
    )
    ap.add_argument(
        "--backend",
        choices=["auto", "qarpx", "numpy"],
        default="auto",
        help="XY-FMQAOA quantum circuit backend (default: auto -> qarpx if installed else numpy).",
    )
    ap.add_argument(
        "--tmm-backend",
        choices=["numpy", "tmm_fast"],
        default="numpy",
        help="Optical TMM backend for truth-table generation (default: numpy).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Run lightweight smoke configuration to verify end-to-end execution and all 8 figures quickly.",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for raw outputs (default: data/runs/mlfilter-paper-repro/<smoke|full>).",
    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Directory for aggregated CSVs and figures (default: analysis/mlfilter-paper-repro/<smoke|full>).",
    )
    args = ap.parse_args()

    mode_tag = "smoke" if args.smoke else "full"
    default_run_dir, default_fig_dir = default_output_dirs(mode_tag)
    run_dir = args.run_dir or default_run_dir
    fig_dir = args.fig_dir or default_fig_dir
    cfg = load_effective_config(args.smoke)

    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "pipeline_metadata.json",
        {
            "study": "mlfilter-paper-repro",
            "mode": mode_tag,
            "xy_backend_requested": args.backend,
            "tmm_backend": args.tmm_backend,
            "workers": args.workers,
            "updated_utc": utc_now(),
            "effective_config": cfg,
        },
    )

    if args.suite in ("all", "truth", "fixed_fm", "confirm1", "confirm2", "explore_L678"):
        run_suite_truth(cfg, run_dir, args.tmm_backend)
    if args.suite in ("all", "fixed_fm"):
        run_suite_fixed_fm(cfg, run_dir, args.backend, args.workers)
    if args.suite in ("all", "confirm1"):
        run_suite_closed_loop("confirm1", cfg, run_dir, args.backend, args.workers)
    if args.suite in ("all", "confirm2"):
        run_suite_closed_loop("confirm2", cfg, run_dir, args.backend, args.workers)
    if args.suite in ("all", "explore_L678"):
        run_suite_closed_loop("explore_L678", cfg, run_dir, args.backend, args.workers)

    if args.suite in ("all", "plot"):
        csvs = build_all_csvs(run_dir, fig_dir / "csv")
        drawn = draw_all_figures(fig_dir)
        print(f"[plot] CSVs ({len(csvs)}): {', '.join(csvs)}")
        print(f"[plot] Figures ({len(drawn)}) written to {fig_dir}/{{pdf,png,svg}}/")


if __name__ == "__main__":
    main()

