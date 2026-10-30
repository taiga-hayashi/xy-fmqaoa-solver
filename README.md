# XY-FMQAOA Paper Reproduction & Solver Benchmark

## 論文全8枚プロット自己完結再現パッケージ (`experiments/mlfilter-paper-repro`)

どの Linux / macOS サーバーでも、以下のコマンドをそのままコピペするだけで、環境構築から全実験（光学真値生成・固定FM・Confirm 1・Confirm 2・`L=6,7,8` スケーリング）・全8枚の図（PDF / PNG / SVG）および CSV 出力までが自己完結で回ります（`Boltzmann` 比較除外済み、OpenQARP 未導入環境でも純粋 NumPy バックエンドで完全一致動作）：

```bash
git clone https://github.com/taiga-hayashi/xy-fmqaoa-solver.git
cd xy-fmqaoa-solver/experiments/mlfilter-paper-repro

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 1. 動作確認スモークテスト（約30〜60秒）
.venv/bin/python run_pipeline.py --suite all --smoke --workers 4

# 2. 本番フル並列実行（例: 16コア並列・バックグラウンド実行、中断再開対応）
nohup .venv/bin/python run_pipeline.py --suite all --workers 16 > run.log 2>&1 &
```

- 進捗確認: `tail -f run.log`
- 出力先: `experiments/mlfilter-paper-repro/outputs/full/figures/{pdf,png,svg,csv}/`
- 詳細な仕様・個別スイート実行方法は [`experiments/mlfilter-paper-repro/README.md`](experiments/mlfilter-paper-repro/README.md) を参照してください。

---

## 旧 Solver Benchmark (`mlfilter-scale`)

サーバー（`/home/jovyan/FMA/Fujitsu_OpenQarp`）のターミナルに以下をそのままコピペすると、$L=6, 7, 8$ の計算（固定FM学習・SA・XY-FMQAOA $p=1\sim 6$・`summary.json` 出力）のみが一括で回ります（プロットは実行しません）：

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
