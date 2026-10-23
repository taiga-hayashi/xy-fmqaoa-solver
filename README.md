# XY-FMQAOA Solver Benchmark (`mlfilter-scale`)

## 1. クローン
```bash
git clone https://github.com/taiga-hayashi/xy-fmqaoa-solver.git
cd xy-fmqaoa-solver
```

## 2. 保存済み CSV から図だけ再生成する場合（数秒）
```bash
python3 analysis/mlfilter-scale/solver/plot.py --replot
```
出力先: `analysis/mlfilter-scale/solver/pdf/`（`xy_depth.pdf` など）

---

## 3. 実験計算（SA・XY-FMQAOA $p=1\sim 6$）を1から実行する場合

### ワンライナー実行（付属スクリプト `run_solver.sh`）
```bash
# L=7 を直列実行
bash run_solver.sh 7

# L=7 をシード並列で高速実行
PARALLEL=1 bash run_solver.sh 7

# L=6, 7, 8 すべてをシード並列で実行
PARALLEL=1 bash run_solver.sh all
```

---

### 個別コマンドをコピペして実行する場合（例: $L=7$）

層数 $L$ ごとの真値テーブルパス（リポジトリに同梱済み）：
- `L=6`: `data/runs/mlfilter-scale/c-266f9e02f5c8/r-2295f36082ca`
- `L=7`: `data/runs/mlfilter-scale/c-5e9cbb5483f7/r-f492fe73e4f9`
- `L=8`: `data/runs/mlfilter-scale/c-94cc85879e73/r-00239d4f5cc3`

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

L=7
TRUTH_DIR="data/runs/mlfilter-scale/c-5e9cbb5483f7/r-f492fe73e4f9"

# ① 前半 (SA + XY p=1..3) の Run ディレクトリ作成
SOLVER_RUN=$(python3 experiments/mlfilter-scale/py/solver_fixedfm.py init \
  --layers "$L" --truth-from "$TRUTH_DIR")
echo "SOLVER_RUN = $SOLVER_RUN"

# ② 5本の固定 FM (Seeds: 42, 101, 2024, 7, 19) を学習して凍結
python3 experiments/mlfilter-scale/py/solver_fixedfm.py freeze --run-dir "$SOLVER_RUN"

# ③ 各 Seed で SA・RandomFM と XY-FMQAOA (p=1..3, Mean / CVaR_0.1) を実行
for SEED in 42 101 2024 7 19; do
  python3 experiments/mlfilter-scale/py/solver_fixedfm.py sa --run-dir "$SOLVER_RUN" --seed "$SEED"
  python3 experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective mean
  python3 experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective cvar10
done

# ④ 前半の結果を summary.json に集計
python3 experiments/mlfilter-scale/py/solver_fixedfm.py analyze --run-dir "$SOLVER_RUN"

# ⑤ 後半 (XY p=4..6) の Run ディレクトリ作成（p=3 の最良角度を引き継ぐ）
XYDEEP_RUN=$(python3 experiments/mlfilter-scale/py/xy_deep.py init --solver-run "$SOLVER_RUN")
echo "XYDEEP_RUN = $XYDEEP_RUN"

# ⑥ 各 Seed で XY-FMQAOA (p=4..6, Mean / CVaR_0.1) を実行
for SEED in 42 101 2024 7 19; do
  python3 experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective mean
  python3 experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective cvar10
done

# ⑦ 後半の結果を summary.json に集計
python3 experiments/mlfilter-scale/py/xy_deep.py analyze --run-dir "$XYDEEP_RUN"
```

### マルチコア環境で並列実行する場合（Step ③・⑥）
```bash
# Step ③（並列）
for SEED in 42 101 2024 7 19; do
  python3 experiments/mlfilter-scale/py/solver_fixedfm.py sa --run-dir "$SOLVER_RUN" --seed "$SEED" &
  python3 experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective mean &
  python3 experiments/mlfilter-scale/py/solver_fixedfm.py xy --run-dir "$SOLVER_RUN" --seed "$SEED" --objective cvar10 &
done
wait
python3 experiments/mlfilter-scale/py/solver_fixedfm.py analyze --run-dir "$SOLVER_RUN"

# Step ⑤〜⑦（並列）
XYDEEP_RUN=$(python3 experiments/mlfilter-scale/py/xy_deep.py init --solver-run "$SOLVER_RUN")
for SEED in 42 101 2024 7 19; do
  python3 experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective mean &
  python3 experiments/mlfilter-scale/py/xy_deep.py run --run-dir "$XYDEEP_RUN" --seed "$SEED" --objective cvar10 &
done
wait
python3 experiments/mlfilter-scale/py/xy_deep.py analyze --run-dir "$XYDEEP_RUN"
```
