# `mlfilter-paper-repro`: 論文図（全8枚）自己完結再現パッケージ

`analysis/mlfilter-scale/paper/pdf` に収録されている全 8 枚のプロット（および `csv/compute_cost.csv`）を、外部フォルダ（`Intern_Fujitsu` 等）への依存なし・重複計算なしでゼロから一貫して再現するパッケージです。温度付きボルツマン分布（`Boltzmann`）との比較は除外し、古典のペナルティ不要ベースラインは `CatSA`（Categorical SA）のみを対象としています。

## 生成される図・表一覧

1. `concept.{pdf,png,svg}` — 罰則付き QUBO のエネルギー障壁（小罰則 $\alpha=0.3$ / 大罰則 $\alpha=30$）と XY リングミキサの概念図
2. `paired_ranks.{pdf,png,svg}` — 第1確認実験（`L=7`, 予算 100, 15 seeds `5001..5015`）のシード対応最終順位比較
3. `loop_alpha_sweep.{pdf,png,svg}` — クローズドループにおける罰則係数 $\alpha \in \{0.3,\dots,1000\}$ スイープと `Adaptive`・`XY-FMQAOA` の比較
4. `retune_robustness.{pdf,png,svg}` — 凍結 FM（`L=6,7,8`）およびクローズドループ（`L=7`）での SA 温度スケジュール再調整に対する頑健性
5. `energy_hist.{pdf,png,svg}` — 凍結 FM（`L=7`）における実行可能サンプルの FM パーセンタイル密度分布
6. `loop_mechanism.{pdf,png,svg}` + `csv/compute_cost.csv` — 60 サイクルにわたる選択解の真の順位・未観測 FM 最小ヒット率・新規候補多様性と計算時間表
7. `confirm2.{pdf,png,svg}` — 第2確認実験（`L=7`, 独立 30 seeds `6001..6030`）における `XY-FMQAOA` vs `CatSA` vs `SA`
8. `si_closed_loop_L678.{pdf,png,svg}` — 層数 `L = 6, 7, 8`（5 seeds）にわたるクローズドループ収束曲線

## クイックスタート

### 0. 標準 `venv` のみでの環境構築（未作成の場合）

`conda`・`uv`・OpenQARP（`qarpx`）・Jupyter が無い素の Python 環境でも、標準の `python3 -m venv` と `pip` だけで全実験・全プロットが完結します（Google Drive 同期トラブル防止のため、仮想環境はローカルディスク上に作成してください）：

```bash
python3 -m venv ~/.venvs/mlfilter-repro
~/.venvs/mlfilter-repro/bin/pip install --upgrade pip
~/.venvs/mlfilter-repro/bin/pip install -r experiments/mlfilter-paper-repro/requirements.txt
```

### 1. 単体数値整合性テストの実行

```bash
python -m unittest discover -s experiments/mlfilter-paper-repro/tests -v
```

### 2. 高速 `--smoke` 検証（ローカル環境で約 30〜60 秒）

全スイート（光学真値生成 → 固定 FM → Confirm 1 → Confirm 2 → `L=6,7,8` 探索）を軽量設定で完走させ、8 枚すべての PDF / PNG / SVG を `analysis/mlfilter-paper-repro/smoke/` に出力します。

```bash
python experiments/mlfilter-paper-repro/run_pipeline.py --suite all --smoke --workers 4
```

### 3. 本番フル再現実行（マルチコア並列・中断再開対応）

完了した `(condition, seed)` の JSON は自動的にスキップされるため、途中停止しても同じコマンドで再開できます。
- `--backend auto`: `qarpx`（OpenQARP）がインストールされていれば `qarpx` を使用し、未導入環境では同一ユニタリの純粋 NumPy テンソル縮約状態ベクトルシミュレータに自動フォールバックします（`--backend numpy` または `--backend qarpx` で明示指定も可能）。

```bash
python experiments/mlfilter-paper-repro/run_pipeline.py --suite all --workers 16 --backend auto
```

スイート単位での個別実行も可能です：

```bash
python experiments/mlfilter-paper-repro/run_pipeline.py --suite truth
python experiments/mlfilter-paper-repro/run_pipeline.py --suite fixed_fm --workers 16
python experiments/mlfilter-paper-repro/run_pipeline.py --suite confirm1 --workers 16
python experiments/mlfilter-paper-repro/run_pipeline.py --suite confirm2 --workers 16
python experiments/mlfilter-paper-repro/run_pipeline.py --suite explore_L678 --workers 16
python experiments/mlfilter-paper-repro/run_pipeline.py --suite plot
```

### 4. 集計済み CSV からの再プロットのみ実行

```bash
python experiments/mlfilter-paper-repro/plot_paper.py --replot
```

## 別のサーバー（Linux 等）で実行する場合

光学スペクトル入力（`inputs/reference/{dielectric_ref.txt,solar_spectrum.txt}`）も本フォルダ内に同梱されているため、GitHub（`xy-fmqaoa-solver`）から認証なしでそのまま `git clone` / ZIP ダウンロードする、または **`mlfilter-paper-repro` フォルダ単体を `scp` / `rsync` するだけ**で、どの Linux / macOS サーバーでもそのまま動作します。

- ヘッドレス環境対応済み（`matplotlib` `Agg` バックエンド使用、X11 / `$DISPLAY` 不要）
- マルチコア Linux サーバーでのスレッド競合防止済み（`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` 自動設定）
- サーバー／単体配置時は出力先が自動的に `experiments/mlfilter-paper-repro/outputs/<smoke|full>/{runs,figures}`（単体フォルダ直下実行時は `./outputs/<smoke|full>/{runs,figures}`）になります（`--run-dir` / `--fig-dir` で明示指定も可）

### GitHub 経由でサーバー実行する場合（認証不要）

```bash
git clone https://github.com/taiga-hayashi/xy-fmqaoa-solver.git
cd xy-fmqaoa-solver/experiments/mlfilter-paper-repro

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 動作確認（約30〜60秒）
.venv/bin/python run_pipeline.py --suite all --smoke --workers 4

# 本番フル並列実行（例: 16 コア並列、バックグラウンド実行）
nohup .venv/bin/python run_pipeline.py --suite all --workers 16 > run.log 2>&1 &
```

### `scp` で単体フォルダのみ転送する場合

```bash
scp -r "experiments/mlfilter-paper-repro" user@remote-server:~/
```

