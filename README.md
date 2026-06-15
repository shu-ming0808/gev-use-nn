# 使用神經網路快速估計 GEV 分布參數

## 專案目的

本專案實作「使用神經網路快速估計廣義極值分布（GEV）參數」的方法，並延伸到臺灣測站資料、空間 Kriging 推估與重現期水準圖。

核心目標：

- 使用模擬 GEV 樣本訓練神經網路
- 用 11 個樣本分位數估計 $\mu, \sigma, \xi$
- 套用到臺灣 25 個測站的年最大高溫資料
- 將測站參數用 Gaussian process / Kriging 推估到空間網格
- 比較 NN 與分位數比例法的參數估計表現

參考論文：

```text
Fast parameter estimation of generalized extreme value distribution using neural networks
```

## 快速開始

```bash
conda env create -f environment.yml
conda activate gev-nn

python src/prepare_annual_max.py
python src/estimate_real_params.py
python src/merge_station_data.py
python src/kriging_params.py
python src/plot_gev_maps.py
python src/compute_return_level.py
```

若要執行空間模擬驗證：

```bash
python src/simulate_spatial_gev.py
```

## NN 訓練架構

原始 NN 不是只使用 3 個分位數，而是固定使用 11 個 quantiles 作為輸入：

```text
0.0001, 0.001, 0.01, 0.1, 0.25,
0.5, 0.75, 0.9, 0.99, 0.999, 0.9999
```

每筆樣本會先做 median / IQR robust standardization：

```text
z = (y - median(y)) / IQR(y)
```

再從標準化後的樣本 $z$ 取 11 個分位數，作為 NN 輸入。

NN 的概念可以寫成：

```text
11 個 standardized quantiles -> GEV parameters
```

注意：`scipy.stats.genextreme` 的 shape 參數是 `c`，和極值理論常用的 $\xi$ 相反：

```text
xi = -c
```

因此整理結果時要確認欄位定義：

- `shape_c_hat`：scipy 的 shape 參數
- `xi_hat`：極值理論定義下的 shape 參數

## 模擬訓練資料切法

模擬資料程式在：

```text
src/simulate_data.py
```

資料產生方式：

- 總資料數：`340000`
- 訓練資料：前 `300000`
- 驗證資料：後 `40000`
- sample size：

```text
30, 72, 173, 416, 1000
```

每一組 GEV 參數會複製到五種 sample size，讓 NN 學習不同樣本大小下的分位數型態。

參數範圍：

- `mu`：均勻分布
- `sigma`：log-uniform
- `c`：均勻分布，其中 `c = -xi`

## 真實 25 測站資料切法

真實資料整理程式在：

```text
src/prepare_annual_max.py
```

流程：

1. 讀取 `data/original_data/pivot_25stations.csv`
2. 將第一欄日期轉為 datetime
3. 只保留 1980 年以後資料
4. 對每個測站做 annual block maxima

輸出：

```text
data/processed/annual_max_25stations.csv
```

資料形狀為：

```text
45 年 × 25 測站
```

## 真實資料參數估計

估計程式：

```text
src/estimate_real_params.py
```

輸出：

```text
data/processed/station_gev_params.csv
```

主要欄位：

- `mu_hat`
- `sigma_hat`
- `log_sigma_hat`
- `xi_hat`
- `shape_c_hat`

## 空間 Kriging

測站參數會先和測站經緯度合併：

```bash
python src/merge_station_data.py
```

再使用 Gaussian process / Kriging 推估到規則空間網格：

```bash
python src/kriging_params.py
```

輸出：

```text
data/processed/grid_gev_params.csv
```

主要欄位：

- `lon`
- `lat`
- `mu`
- `sigma`
- `log_sigma`
- `xi`

## 100 年重現期水準

計算程式：

```text
src/compute_return_level.py
```

GEV 的 $T$-year return level：

```text
z_T(s) = mu(s) + sigma(s) / xi(s) * {[-log(1 - 1/T)]^(-xi(s)) - 1}
```

輸出：

```text
data/processed/grid_return_level.csv
```

## 分位數比例法

教授建議的分位數比例為：

```text
R = [Q(p3) - Q(p2)] / [Q(p2) - Q(p1)]
```

GEV 分位數可寫成：

```text
Q(p) = mu + sigma * b(p, xi)
```

所以比例 $R$ 會消去 $\mu$ 與 $\sigma$，可以先數值解 $\xi$，再反推 $\sigma$ 與 $\mu$。

早期單組比例可以使用：

```text
p1 = 0.25
p2 = 0.50
p3 = 0.75
```

但正式分析應該對齊 NN 的 11 quantile 架構。完整 QR11 分析已放在 sibling 專案：

```text
../fast_parameter_using_NN_window_data/notebooks/quantile_ratio_11_quantile_analysis.ipynb
```

QR11 的做法：

1. 先取和 NN 相同的 11 個 empirical quantiles
2. 從 11 個 quantiles 中形成多組 $(p_1,p_2,p_3)$
3. 對每組比例數值解 $\xi$
4. 對成功解取 median，得到穩健的 $\hat\mu, \hat\sigma, \hat\xi$

## 空間模擬驗證

空間模擬程式：

```text
src/simulate_spatial_gev.py
```

目的：

- 使用真實 25 測站座標
- 產生已知的空間 GEV 參數場
- 模擬 annual maxima 與 monthly maxima
- 用 NN 估計測站參數
- 用 RBF / Matern Kriging 推估空間場
- 和 true parameter field 比較 RMSE、MAE、correlation

輸出位置：

```text
data/simulated/spatial_gev/
```

重要檔案：

- `spatial_station_true_params.csv`
- `spatial_annual_max_25stations.csv`
- `spatial_monthly_max_25stations.csv`
- `spatial_annual_station_nn_estimates.csv`
- `spatial_monthly_station_nn_estimates.csv`
- `spatial_annual_station_error_summary.csv`
- `spatial_monthly_station_error_summary.csv`
- `spatial_annual_monthly_grid_error_summary.csv`

## 專案結構

```text
fast_parameter_using_NN/
│
├── data/
│   ├── original_data/
│   ├── processed/
│   └── simulated/
│
├── models/
│   ├── best_baseline_model.pth
│   └── best_weighted_model.pth
│
├── notebooks/
│   └── main.ipynb
│
├── results/
│   └── figures/
│
├── src/
│   ├── baseline_train.py
│   ├── weighted_train.py
│   ├── simulate_data.py
│   ├── prepare_annual_max.py
│   ├── estimate_real_params.py
│   ├── merge_station_data.py
│   ├── kriging_params.py
│   ├── plot_gev_maps.py
│   ├── compute_return_level.py
│   └── simulate_spatial_gev.py
│
└── README.md
```

## 後續工作

- 使用 QR11 notebook 比較 NN 與分位數比例法
- 將 TCCIP gridded temperature data 與 25 測站資料放在同一個架構下比較
- 測試不同 $(p_1,p_2,p_3)$ 組合對 $\xi$ 與 $\sigma$ 的穩定性
- 比較 NN、MLE、L-moments、QR11 的 station-level 與 grid-level 表現

## 作者

Shu-Ming Chang  
National Central University
