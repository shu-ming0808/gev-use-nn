# 臺灣月最高溫網格資料 GEV NN 與 11 分位數比例分析

## 專案目的

本專案使用 TCCIP 臺灣月最高溫網格資料，整理成和原本 25 測站專案相同的 GEV 估計格式，並使用既有 neural network 模型估計每個網格點的 GEV 參數。

目前專案也新增一條教授建議的分位數比例法：

```text
11 個 empirical quantiles
-> 多組三分位比例 R = [Q(p3)-Q(p2)] / [Q(p2)-Q(p1)]
-> 數值解 xi
-> 反推 sigma 與 mu
-> 和 NN 結果比較
```

## 資料切法

### 1. 原始月資料整理

主要流程在 `main.py` 與 `main.ipynb`：

1. 讀取原始 `觀測_月資料_臺灣_最高溫_YYYY.csv`。
2. 將每年 12 個月份欄位轉成 long format：

```text
date, station, lon, lat, max_temp
```

3. 將 `-99.9` 等無效值視為缺值。
4. pivot 成：

```text
Date x grid-station
```

5. 只保留 1980 年以後資料。
6. 每個 grid-station 的有效月份比例至少要達到 80%。
7. 對每個 grid-station 取年度最大值：

```text
monthly maximum temperature -> annual block maxima
```

8. 每個 grid-station 至少要有 30 年 annual maxima，對齊原 NN 訓練資料的 sample size 下限。

主要輸出：

- `data/processed/monthly_long_grid_temperature.csv`
- `data/processed/pivot_grid_monthly_max_temperature_after_1980_clean.csv`
- `data/processed/annual_max_grid_temperature.csv`
- `data/processed/annual_grid_station_location.csv`

### 2. NN 使用的 11 個 quantiles

本專案沿用原始 NN 訓練架構，輸入不是任意挑 3 個 quantiles，而是固定 11 個 quantiles：

```text
0.0001, 0.001, 0.01, 0.1, 0.25,
0.5, 0.75, 0.9, 0.99, 0.999, 0.9999
```

每個樣本會先做 median / IQR robust standardization：

```text
z = (y - median(y)) / IQR(y)
```

再從標準化後的 `z` 取 11 個 quantiles 作為 NN input。

### 3. 11 分位數比例法

新增檔案：

```text
src/quantile_ratio_estimator.py
```

QR11 方法會先取同一組 11 個 quantiles，再從中產生多組 $(p_1,p_2,p_3)$，對每組計算：

```text
R = [Q(p3) - Q(p2)] / [Q(p2) - Q(p1)]
```

因為 GEV 分位數可寫成：

```text
Q(p) = mu + sigma * b(p, xi)
```

所以比例 $R$ 會消去 $\mu$ 和 $\sigma$，可以先數值解 $\xi$。得到 $\xi$ 後，再用 quantile gap 反推 $\sigma$ 與 $\mu$。

## 新增 Notebook

主要分析 notebook：

```text
notebooks/quantile_ratio_11_quantile_analysis.ipynb
```

內容包含：

- 完整中文 Markdown 說明資料切法與公式
- 顯示 NN 使用的 11 個 quantiles
- 模擬資料 QR11 數值解與 NN 比較
- 真實網格資料 QR11 數值解與 NN 比較
- 原本 25 測站真實資料 QR11 數值解與 NN 比較
- 每個資料區塊都使用 1 x 3 圖呈現 $\mu$、$\sigma$、$\xi$
- 空間圖使用 `geopandas` 讀取臺灣 shapefile，並在圖上疊加臺灣邊界

`fast_parameter_using_NN_window_data` 本身沒有 shapefile，因此 notebook 會自動使用 sibling 原專案中的檔案：

```text
../fast_parameter_using_NN/data/shapefile/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp
```

## 主要輸出

`data/processed/`：

- `simulation_station_true_vs_qr11.csv`
- `simulation_qr11_error_summary.csv`
- `real_grid_station_qr11_estimates.csv`
- `real_station_qr11_estimates.csv`
- `real_grid_kriging_qr11_params.csv`
- `real_station_kriging_nn_params.csv`
- `real_station_kriging_qr11_params.csv`

`results/figures/`：

- `qr11_simulation_true_1x3.png`
- `qr11_simulation_nn_1x3.png`
- `qr11_simulation_qr_1x3.png`
- `qr11_real_grid_nn_1x3.png`
- `qr11_real_grid_qr_1x3.png`
- `qr11_real_station_nn_1x3.png`
- `qr11_real_station_qr_1x3.png`
- `qr11_real_grid_kriging_1x3.png`
- `qr11_real_station_nn_kriging_surface_1x3.png`
- `qr11_real_station_qr11_kriging_surface_1x3.png`

## 建議執行順序

先跑原本 window-data 主流程：

```bash
python main.py
```

再開啟並執行：

```text
notebooks/quantile_ratio_11_quantile_analysis.ipynb
```

## 作者

Shu-Ming Chang  
National Central University
