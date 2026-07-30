# 分析結果摘要

## 資料合併

- 來源資料：`觀測_月資料_臺灣_最高溫_1960.csv` 到 `觀測_月資料_臺灣_最高溫_2024.csv`，共 65 個年度檔。
- 合併後月資料：1980 年後共 540 個月份。
- 原始 pivot 可用網格點：1412 個。
- coverage 篩選門檻：有效月份比例至少 80%。
- 篩選後保留網格點：1412 個。
- 原始經緯度已轉換為 TWD97 / TM2 zone 121 的 `x_km`、`y_km`。
- GP、K-means、variogram、buffer、Moran's I 與地圖均使用公里尺度。

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

- 模擬方式：先由臺灣範圍建立格點，再投影為 TWD97 公里座標進行空間場與 GP 計算。
- 每個格點先生成真實 `mu`、`sigma`、`xi` surface，再模擬 45 年年最大值。
- 使用同一個 NN 估計流程預測 GEV 參數。
- 真實 surface 圖：`results/figures/simulation_true_surface_1x3.png`
- 預測 surface 圖：`results/figures/simulation_predicted_surface_1x3.png`
- 誤差表：`data/processed/simulation_error_summary.csv`

## 高程、Isotropy 與候選模型

- 高程以 one-to-one GRID 鍵值併入，空間位置使用 `x_km`、`y_km`。
- baseline 為 linear elevation mean 加 stationary isotropic Matérn(1.5) covariance。
- isotropy screening 比較 0、45、90、135 度 directional variograms。
- 方向差異統計量會和 baseline parametric simulation 的 95% 上界比較。
- 超過上界只代表應加入 geometric anisotropy 候選，不代表直接選定 anisotropic GP。
- 同時檢查 nonlinear elevation、variance/range nonstationarity 與 spatial-elevation distance。
- 候選模型的最終選擇使用 buffered spatial-CV；AIC/BIC 為輔助診斷。
