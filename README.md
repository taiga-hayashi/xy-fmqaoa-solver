# XY-FMQAOA Solver Benchmark (`mlfilter-scale`)

## A. サーバー（`/home/jovyan/FMA/Fujitsu_OpenQarp`）に既にファイルがある状態で一括実行する場合

サーバーのターミナルに以下をそのままコピペすると、$L=6, 7, 8$ の全計算（固定FM学習・SA・XY-FMQAOA $p=1\sim 6$）から集計・プロット更新まで一括で回ります：

```bash
cd /home/jovyan/FMA/Fujitsu_OpenQarp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="/home/jovyan/FMA/.venv/bin/python"

declare -A TRUTH=(
  [6]="data/runs/mlfilter-scale/c-266f9e02f5c8/r-2295f36082ca"
  [7]="data/runs/mlfilter-scale/c-5e9cbb5483f7/r-f492fe73e4f9"
  [8]="data/runs/mlfilter-scale/c-94cc85879e73/r-00239d4f5cc3"
)

for L in 6 7 8; do
  SOLVER_RUN=$("$PY" experiments/mlfilter-scale/py/solver_fixedfm.py init --layers "$L" --truth-from "${TRUTH[$L]}")
  "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py freeze --run-dir "$SOLVER_RUN"

  for SEED in 42 101 2024 7 19; do
    "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py sa --run-dir "$SOLVER_RUN" --seed "$SEED" &
    "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective mean &
    "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective cvar10 &
  done
  if [ "$L" -eq 6 ]; then
    for SEED in 42 101 2024 7 19; do
      for ALPHA in 1 10 100; do
        "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py xstd --run-dir "$SOLVER_RUN" --seed "$SEED" --alpha "$ALPHA" &
      done
    done
  fi
  wait
  "$PY" experiments/mlfilter-scale/py/solver_fixedfm.py analyze --run-dir "$SOLVER_RUN"

  XYDEEP_RUN=$("$PY" experiments/mlfilter-scale/py/xy_deep.py init --solver-run "$SOLVER_RUN")
  for SEED in 42 101 2024 7 19; do
    "$PY" experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective mean &
    "$PY" experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective cvar10 &
  done
  wait
  "$PY" experiments/mlfilter-scale/py/xy_deep.py analyze --run-dir "$XYDEEP_RUN"
done
```

---

## B. 新規環境へクローンして一括実行する場合

```bash
git clone https://github.com/taiga-hayashi/xy-fmqaoa-solver.git
cd xy-fmqaoa-solver
bash run_all.sh
```
