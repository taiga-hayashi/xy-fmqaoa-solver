# analysis/mlfilter-scale/solver: ソルバー単体の比較(固定 FM)

大きいペナルティ係数で SA の探索がランダムに近づくこと(エネルギー地形の平準化)と、ペナルティ不要の XY-FMQAOA との比較を図にする。

- 入力: `data/runs/mlfilter-scale/` の solver-fixedfm run と solver-xydeep run(p=4〜6、L=6・7・8)の `raw/summary.json`。SHA-256 は [analysis_config.json](analysis_config.json)
- 条件: [experiments/mlfilter-scale/solver_fixedfm.json](../../../experiments/mlfilter-scale/solver_fixedfm.json)
- 入口: `python plot.py`(CSV を作り直して全図を描く)、`python plot.py --replot`(CSV から描くだけ)、`plot.ipynb`(同じ関数。各図の読み方つき)
- 共通の図の設定: [src/aps_style.py](../../../src/aps_style.py)(PRR/APS 形式、タイトルなし)

## 出力

| ファイル | 内容 |
|---|---|
| `penalty_flattening_{cvar10,mean}` | 実行可能なサンプルの FM 百分位の平均 vs α(主図)。XY の目的関数ごと |
| `penalty_selected_gap_{cvar10,mean}` | 選ばれた候補の正規化 FM ギャップ vs α |
| `penalty_mechanism` | 凍結時の FM コントラスト ΔE_FM/λ vs FM 百分位(FM ごとの点、L 別) |
| `penalty_feasibility` | 実行可能率 vs α(SA、通常 QAOA、XY) |
| `penalty_sweetspot` | ペナルティ係数 α のスイートスポット(L=6・7・8 別々の 2×3 図。上段: 未評価 FM 最小解の捕捉率、下段: 実行可能率とサンプルの FM 百分位のトレードオフ) |
| `penalty_sweetspot_combined` | ペナルティ係数 α のスイートスポット(L=6・7・8 を同一軸に重ねた 2×1 図) |
| `xy_depth` | XY-FMQAOA の深さ p=1〜6 と SA の基準線(上段: FM ギャップ、下段: サンプルの FM 百分位) |
| `xy_simple_failure` | 単純な XY-QAOA(p=1・7×7 格子・Mean 目的)ではランダム提案や平坦化 SA(α=100)と同程度に留まることと、深さ p=1〜6・CVaR 目的での改善(L=6・7・8 別々の 2×3 図) |
| `xy_simple_failure_combined` | 単純な XY-QAOA の性能不足と深さ・CVaR 目的での改善(L=6・7・8 を同一軸に重ねた 2×1 図) |
| `panels/` | 3 パネル図(`penalty_flattening_*`, `penalty_selected_gap_*`, `penalty_mechanism`, `penalty_feasibility`, `penalty_sweetspot_L*`, `xy_depth_L*`, `xy_simple_failure_L*`)の各 L を独立に描いたもの |
| `csv/solver_summary.csv` | 図の元の値(L・手法・指標ごとの 5 FM の中央値と IQR) |
| `csv/solver_per_fm.csv` | FM ごとの値(10 reps の平均) |
| `csv/solver_exact_per_fm.csv` | XY・通常 QAOA の厳密分布から計算した値 |

形式: `png/`(600 dpi)、`pdf/`、`svg/` に同名で保存。

## 状態

- 2026-10-20: 実データ(run の summary)から作図。p=4〜6 を追加して再実行(analysis_id a-e15858711587、旧 a-75ff8f852a1c は docs/records/analysis_runs/ に記録)。
- 2026-10-23: ペナルティ係数 α のスイートスポット図(`penalty_sweetspot`, `penalty_sweetspot_combined`, `panels/penalty_sweetspot_L{6,7,8}`)を追加して再作図(analysis_id a-c62ac6b99b06、旧 a-e15858711587 は docs/records/analysis_runs/ に記録)。Notebook は先頭から実行済み(エラーなし)。PNG は目視確認済み、PDF は未確認
- Drive 保存は未実施

- 2026-10-23: a-c6fea137c24a。`Shaded: sweet spot` の文字だけを削除。CSV変更なし。対象図と個別L図のPDF/PNG/SVGを再生成し、対象PDFの表示を確認。Notebookの該当表示PNGを更新（全セル再実行なし）。索引書込時のエラーは `docs/records/analysis_runs/index-recovery.json` に記録。

- L別の上下2段図: `pdf/xy_depth_L6.pdf`、`pdf/xy_depth_L7.pdf`、`pdf/xy_depth_L8.pdf`（PNG/SVGも同名）。元図と同じ中央値/IQR。`fig_depth_by_layer()`で再生成。解析ID: a-a1c16096b5d3。追加図だけ生成しNotebookに表示を追加。
