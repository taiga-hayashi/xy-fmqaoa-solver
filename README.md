# XY-FMQAOA Solver Benchmark (`mlfilter-scale`)

## 一括実行（クローンして全計算＋CSV・プロット生成まで1コマンド）

```bash
git clone https://github.com/taiga-hayashi/xy-fmqaoa-solver.git
cd xy-fmqaoa-solver
bash run_all.sh
```

- `bash run_all.sh` : $L=6, 7, 8$ すべての固定FM学習・SA・XY-FMQAOA ($p=1\sim 6$) をシード並列で実行し、最後に `analysis/mlfilter-scale/solver/` の CSV と全図（`pdf/xy_depth.pdf` 等）を更新します。
- `bash run_all.sh 6` : 特定の層数（`6`, `7`, `8` のいずれか）だけを一括計算します。
- `nohup bash run_all.sh > solver.log 2>&1 &` : バックグラウンドで全計算を流す場合。
- `python3 analysis/mlfilter-scale/solver/plot.py --replot` : 既存の `csv/` から図だけ再描画する場合（数秒）。
