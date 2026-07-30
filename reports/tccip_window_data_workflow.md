# TCCIP 網格資料 GEV 參數估計實驗

本文件說明主專案 `fast_parameter_using_NN` 的 TCCIP GRID 流程。目前已下載的
TCCIP 月最高溫產品代表「月平均日最高溫」，不是「月內最高的日最高溫」。
舊流程把每年 12 筆月平均日最高溫再取最大，可作為「年度最暖月份」分析，
但不能稱為真正的月／年 block maxima。

若研究問題是比較月最大值與年最大值，必須改用 TCCIP 網格化觀測**日**
最高溫：先由每日資料計算月最大值，再從同一年 12 筆月最大值計算年最大值。
對應 notebook：

```text
notebooks/annual_monthly_max_comparison.ipynb
```

日資料前處理程式：

```text
src/prepare_daily_tmax_block_maxima.py
```

本實驗同時加入教授建議的 11 分位數比例法 QR11，用來比較：

```text
NN 直接估計 GEV 參數
QR11 先用分位數比例估 xi，再反推 sigma 與 mu
```

## 主要檔案架構

```text
fast_parameter_using_NN/
├── src/
│   ├── tccip_grid_preprocessing.py
│   ├── elevation_gp_analysis.py
│   ├── terrain_predictors.py
│   ├── prepare_daily_tmax_block_maxima.py
│   └── quantile_ratio_estimator.py
├── notebooks/
│   ├── real_TCCIP_grid_data.ipynb
│   ├── elevation_gp_model_comparison.ipynb
│   ├── annual_monthly_max_comparison.ipynb
│   └── quantile_ratio_11_quantile_analysis.ipynb
├── data/
│   ├── processed/
│   └── spatial_predictors/
└── results/
    ├── figures/
    └── tables/
```

各資料夾用途如下：

| 路徑 | 內容 |
|---|---|
| `src/tccip_grid_preprocessing.py` | TCCIP 網格資料前處理、NN 估計、Kriging 與基本視覺化流程 |
| `notebooks/tccip_grid_preprocessing.ipynb` | 主流程的 notebook 版本，方便逐步檢查資料處理結果 |
| `src/quantile_ratio_estimator.py` | QR11 分位數比例估計器 |
| `src/elevation_gp_analysis.py` | 高程 mean structure、isotropy screening、GP 選模、SKCV 與 RL 分析 |
| `src/terrain_predictors.py` | 從 DEM 產生坡度、坡向、northness、eastness、地形起伏與 ruggedness |
| `notebooks/real_TCCIP_grid_data.ipynb` | 真實 GRID 的 variogram、fold、buffer、SKCV/RLO 與 residual diagnostics |
| `notebooks/elevation_gp_model_comparison.ipynb` | T0/T1、高程、kernel、isotropy 與 mixed return-level pipeline |
| `notebooks/quantile_ratio_11_quantile_analysis.ipynb` | QR11 主要分析 notebook，包含模擬資料與真實網格資料 |
| `data/processed/` | 前處理後的 annual maxima、參數估計表與比較表 |
| `models/` | NN 模型權重，通常不建議直接放 Git |
| `results/analysis_summary.md` | 主要分析結果摘要 |
| `results/figures/` | 1 x 3 參數圖與 Kriging surface 圖 |

## 資料處理流程

目前已下載的原始資料為 TCCIP 月尺度最高溫網格產品，主要流程如下：

```text
原始「月平均日最高溫」CSV
-> long format: date, station, lon, lat, max_temp
-> 經緯度轉成 TWD97 / TM2 zone 121 的 x_km, y_km
-> 移除無效值
-> pivot 成 Date x grid-station
-> 保留 1980 年以後資料
-> 保留有效月份比例至少 80% 的網格點
-> 每個網格點取年度最暖月份數值
-> 保留至少 30 年 annual maxima 的網格點
```

其中 `lon`、`lat` 只用於原始 GRID 配對及 CRS 轉換。以下空間計算全部
使用 `x_km`、`y_km`：

- GP covariance 與 kernel length scale；
- coordinate-based K-means folds；
- variogram、directional variogram 與 Moran's I；
- SKCV buffer 與 RLO；
- 高程模型、residual map 與 return-level map。

座標進入 GP 前只做平移置中，不做 per-axis standardization，因此
Euclidean distance、buffer radius 與 kernel length scale 均保留 km 單位。

## Isotropy 與空間結構 Screening

高程分析的 baseline 為：

```text
T1 linear elevation mean
+ stationary isotropic Matern(nu=1.5)
```

screening 使用四方向 directional variograms（0、45、90、135 度），比較
相同距離 bin 中的 semivariance。觀測方向差異會與 stationary isotropic
baseline 產生的 parametric simulation envelope 比較：

```text
observed statistic <= simulated q95
-> 暫時保留 isotropy

observed statistic > simulated q95
-> 將 geometric anisotropy 加入候選模型
```

這不是看圖後主觀選模型，也不是直接宣稱 anisotropy 顯著。它是候選模型
產生階段；候選模型仍必須使用相同 spatial folds 與 buffer，交由 spatial-CV
RMSE、MAE、Bias、fold stability、residual Moran's I 和 residual variogram
共同驗證。

對應檔案：

```text
src/elevation_gp_analysis.py
notebooks/elevation_gp_model_comparison.ipynb
results/tables/elevation_gp_baseline_diagnostic_screening.csv
results/figures/elevation_04_baseline_diagnostic_screening.png
```

主要輸出包含：

```text
data/processed/monthly_long_grid_temperature.csv
data/processed/pivot_grid_monthly_max_temperature_after_1980_clean.csv
data/processed/annual_max_grid_temperature.csv
data/processed/annual_grid_station_location.csv
```

## 11 個 Quantile 架構

本實驗沿用原本 fast parameter estimation NN 的輸入架構。每個樣本不是任意取少數 quantile，而是固定使用 11 個 empirical quantiles：

```text
0.0001, 0.001, 0.01, 0.1, 0.25,
0.5, 0.75, 0.9, 0.99, 0.999, 0.9999
```

在取 quantile 前，會先對每個樣本做 median / IQR robust standardization：

```text
z = (y - median(y)) / IQR(y)
```

因此模擬資料與真實網格資料都使用同一組 11 個 quantiles，這樣 NN 與 QR11 的比較才會一致。

## QR11 分位數比例法

GEV 分位數可寫成：

```text
Q(p) = mu + sigma * b(p, xi)
```

因此對三個分位數 $p_1 < p_2 < p_3$，可以計算比例：

```text
R = [Q(p3) - Q(p2)] / [Q(p2) - Q(p1)]
```

這個比例會消去 $\mu$ 和 $\sigma$，所以可以先用數值方法估計 $\xi$。得到 $\xi$ 後，再反推 $\sigma$ 與 $\mu$。

本實驗使用 11 個 quantiles 產生多組 $(p_1,p_2,p_3)$，對每組估計一組 GEV 參數，最後用穩健的彙整方式得到 QR11 結果。

## 主要 Notebook

QR11 的完整分析在：

```text
notebooks/quantile_ratio_11_quantile_analysis.ipynb
```

內容包含：

- 中文 Markdown 說明資料切法、11 quantiles 與 QR11 公式
- 模擬資料：true、NN、QR11 的數值比較與 1 x 3 圖
- 真實網格資料：NN 與 QR11 的參數比較、Kriging surface 與 1 x 3 圖
- 使用 `geopandas` 讀取臺灣 shapefile，並在空間圖上疊加臺灣邊界
- 所有輸出空間圖以 TWD97 Easting/Northing（km）為座標軸

臺灣邊界 shapefile 會從主專案讀取：

```text
data/shapefile/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp
```

## 主要輸出

`data/processed/` 中的重要表格：

```text
simulation_station_true_vs_qr11.csv
simulation_qr11_error_summary.csv
real_grid_station_qr11_estimates.csv
real_grid_qr11_vs_nn_summary_rows.csv
real_grid_kriging_qr11_params.csv
```

`results/figures/` 中的重要圖片：

```text
qr11_simulation_true_1x3.png
qr11_simulation_nn_1x3.png
qr11_simulation_qr_1x3.png
qr11_real_grid_nn_1x3.png
qr11_real_grid_qr_1x3.png
qr11_real_grid_kriging_1x3.png
```

## 建議執行順序

先執行網格資料主流程：

```bash
python src/tccip_grid_preprocessing.py
```

再執行 QR11 notebook：

```text
notebooks/quantile_ratio_11_quantile_analysis.ipynb
```

如果只想重現 QR11 分析，需先確認 `data/processed/` 中已經存在目前資料
定義下的年度序列與 NN 估計結果。若論文改以真正的月／年極值為主，必須先
下載日最高溫資料並改用 `prepare_daily_tmax_block_maxima.py` 產生輸入。

## Git 上傳建議

建議放進 Git：

```text
README.md
requirements.txt
src/
notebooks/
reports/
results/figures/
```

通常不建議放進 Git：

```text
models/
data/raw/
大型原始資料
可重新產生的大型中間檔
```

如果需要保留大型資料，建議改用雲端硬碟、release asset，或只在 README 中說明資料取得位置與重現流程。

## 作者

Shu-Ming Chang  
National Central University
