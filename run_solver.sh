#!/usr/bin/env bash
# L=6, 7, 8 の固定FM学習・SA・XY-FMQAOA (p=1..6) の計算のみを一括実行するスクリプト（プロットは行わない）
# 使い方:
#   bash run_all.sh          # L=6, 7, 8 すべての計算を一括実行（シード並列）
#   bash run_all.sh 7        # L=7 のみ計算を一括実行
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "/home/jovyan/FMA/.venv/bin/python" ]; then
  PY="/home/jovyan/FMA/.venv/bin/python"
else
  PY="python3"
fi

PARALLEL="${PARALLEL:-1}"

truth_dir_for() {
  case "$1" in
    6) echo "data/runs/mlfilter-scale/c-266f9e02f5c8/r-2295f36082ca" ;;
    7) echo "data/runs/mlfilter-scale/c-5e9cbb5483f7/r-f492fe73e4f9" ;;
    8) echo "data/runs/mlfilter-scale/c-94cc85879e73/r-00239d4f5cc3" ;;
    *) echo "Unsupported layer count: $1 (choose 6, 7, 8, or all)" >&2; exit 1 ;;
  esac
}

run_layer() {
  local L="$1"
  local TRUTH_DIR
  TRUTH_DIR="$(truth_dir_for "$L")"

  echo "============================================================"
  echo "[L=$L] Step 1/7: Initializing solver_fixedfm (p=1..3, SA)"
  echo "============================================================"
  local SOLVER_RUN
  SOLVER_RUN=$("$PY" experiments/mlfilter-scale/py/solver_fixedfm.py init --layers "$L" --truth-from "$TRUTH_DIR")
  echo "  -> SOLVER_RUN: $SOLVER_RUN"

  echo "[L=$L] Step 2/7: Training & freezing 5 FMs (seeds: 42, 101, 2024, 7, 19)"
  "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py freeze --run-dir "$SOLVER_RUN"

  echo "[L=$L] Step 3/7: Running SA, RandomFM, and XY-FMQAOA (p=1..3, Mean & CVaR0.1)"
  for SEED in 42 101 2024 7 19; do
    if [ "$PARALLEL" = "1" ]; then
      "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py sa --run-dir "$SOLVER_RUN" --seed "$SEED" &
      "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective mean &
      "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective cvar10 &
    else
      "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py sa --run-dir "$SOLVER_RUN" --seed "$SEED"
      "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective mean
      "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective cvar10
    fi
  done
  if [ "$L" -eq 6 ]; then
    for SEED in 42 101 2024 7 19; do
      for ALPHA in 1 10 100; do
        if [ "$PARALLEL" = "1" ]; then
          "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xstd --run-dir "$SOLVER_RUN" --seed "$SEED" --alpha "$ALPHA" &
        else
          "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xstd --run-dir "$SOLVER_RUN" --seed "$SEED" --alpha "$ALPHA"
        fi
      done
    done
  fi
  wait

  echo "[L=$L] Step 4/7: Aggregating solver_fixedfm (p=1..3) -> raw/summary.json"
  "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py analyze --run-dir "$SOLVER_RUN"

  echo "[L=$L] Step 5/7: Initializing xy_deep (p=4..6 warm-started from p=3)"
  local XYDEEP_RUN
  XYDEEP_RUN=$("$PY" experiments/mlfilter-scale/py/xy_deep.py init --solver-run "$SOLVER_RUN")
  echo "  -> XYDEEP_RUN: $XYDEEP_RUN"

  echo "[L=$L] Step 6/7: Running XY-FMQAOA (p=4..6, Mean & CVaR0.1)"
  for SEED in 42 101 2024 7 19; do
    if [ "$PARALLEL" = "1" ]; then
      "$PY" experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective mean &
      "$PY" experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective cvar10 &
    else
      "$PY" experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective mean
      "$PY" experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective cvar10
    fi
  done
  wait

  echo "[L=$L] Step 7/7: Aggregating xy_deep (p=4..6) -> raw/summary.json"
  "$PY" experiments/mlfilter-scale/py/xy_deep.py analyze --run-dir "$XYDEEP_RUN"
}

TARGET="${1:-all}"
if [ "$TARGET" = "all" ]; then
  for L in 6 7 8; do
    run_layer "$L"
  done
else
  run_layer "$TARGET"
fi

echo "Done!"
