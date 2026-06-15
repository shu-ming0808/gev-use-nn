# 分析結果摘要

## 資料合併

- 來源資料：`觀測_月資料_臺灣_最高溫_1960.csv` 到 `觀測_月資料_臺灣_最高溫_2024.csv`，共 65 個年度檔。
- 合併後月資料：1980 年後共 540 個月份。
- 原始 pivot 可用網格點：1412 個。
- coverage 篩選門檻：有效月份比例至少 80%。
- 篩選後保留網格點：1412 個。

## 年最大值與 NN 估計

- 年最大值資料：1980 到 2024，共 45 年。
- NN 估計網格點數：1412 個。
- `mu_hat` 平均：29.247。
- `sigma_hat` 平均：0.372。
- `xi_hat` 平均：0.129。

## 主要輸出

- 合併後月資料：`data/processed/pivot_grid_monthly_max_temperature_all.csv`
- 篩選後月資料：`data/processed/pivot_grid_monthly_max_temperature_after_1980_clean.csv`
- 年最大值：`data/processed/annual_max_grid_temperature.csv`
- NN 估計參數：`data/processed/grid_station_gev_params_with_loc.csv`
- Kriging 平滑網格：`data/processed/kriging_grid_gev_params.csv`
- 圖檔：`results/figures/`

## 模擬網格驗證

- 模擬方式：使用臺灣經緯度範圍建立 0.5 度格點，共 80 個格點。
- 每個格點先生成真實 `mu`、`sigma`、`xi` surface，再模擬 45 年年最大值。
- 使用同一個 NN 估計流程預測 GEV 參數。
- 真實 surface 圖：`results/figures/simulation_true_surface_1x3.png`
- 預測 surface 圖：`results/figures/simulation_predicted_surface_1x3.png`
- 誤差表：`data/processed/simulation_error_summary.csv`
