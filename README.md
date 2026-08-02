# 使用神經網路快速估計 GEV 分布參數

## 專案目的

本專案實作「使用神經網路快速估計廣義極值分布（GEV）參數」的方法，並以 TCCIP 臺灣月最高溫網格資料建立空間 Gaussian process、空間驗證與重現期水準分析。

核心目標：

- 使用模擬 GEV 樣本訓練神經網路
- 用 11 個樣本分位數估計 $\mu, \sigma, \xi$
- 將 TCCIP 月最高溫網格序列轉成 11 個 empirical quantiles
- 估計每個 GRID 的 GEV 參數並建立 Gaussian process
- 將 WGS84 經緯度轉成研究國家的公里座標，再進行空間距離、
  isotropy 與 anisotropy 診斷
- 比較 NN 與分位數比例法的參數估計表現

## 主要參考文獻

以下文獻不是全部做相同的事情，而是分別支援本研究的 NN 參數估計、
GEV 理論、空間切分、buffer distance、空間變數選擇與 isotropy 診斷。

### GEV、神經網路與限制條件

1. Rai, S., Hoffman, A., Lahiri, S., Nychka, D. W., Sain, S. R., &
   Bandyopadhyay, S. (2024). Fast parameter estimation of generalized
   extreme value distribution using neural networks. *Environmetrics, 35*(3),
   e2845. [https://doi.org/10.1002/env.2845](https://doi.org/10.1002/env.2845)
   - 本專案 11 個 empirical quantiles、median/IQR 標準化、NN 架構與
     GEV 參數反轉換的主要依據。

2. Galib, A. H., McDonald, A., Wilson, T., Luo, L., & Tan, P.-N. (2022).
   DeepExtrema: A deep learning approach for forecasting block maxima in time
   series data. *Proceedings of IJCAI-22*, 2980-2986.
   [https://doi.org/10.24963/ijcai.2022/413](https://doi.org/10.24963/ijcai.2022/413)
   - 本專案 constraint-penalty 實驗中 GEV support constraint 的參考；
     本專案並未直接重製完整 DeepExtrema 架構。

### Spatial cross-validation 與空間資訊洩漏

3. Roberts, D. R., Bahn, V., Ciuti, S., et al. (2017). Cross-validation
   strategies for data with temporal, spatial, hierarchical, or phylogenetic
   structure. *Ecography, 40*(8), 913-929.
   [https://doi.org/10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)
   - 說明具有空間依賴的資料不應直接使用 random CV，並支持使用 spatial
     blocking 評估空間泛化誤差。

4. Brenning, A. (2012). Spatial cross-validation and bootstrap for the
   assessment of prediction rules in remote sensing: The R package
   `sperrorest`. *2012 IEEE International Geoscience and Remote Sensing
   Symposium*, 5372-5375.
   [https://doi.org/10.1109/IGARSS.2012.6352393](https://doi.org/10.1109/IGARSS.2012.6352393)
   - 支援 spatial resampling，以及以投影座標建立 geographically
     concentrated folds 的方法背景。

5. Pohjankukka, J., Pahikkala, T., Nevalainen, P., & Heikkonen, J. (2017).
   Estimating the prediction performance of spatial models via spatial
   k-fold cross validation. *International Journal of Geographical Information
   Science, 31*(10), 2001-2019.
   [https://doi.org/10.1080/13658816.2017.1346255](https://doi.org/10.1080/13658816.2017.1346255)
   - 本專案 SKCV、test fold 周圍 dead zone，以及 SKCV-RLO
     樣本數控制比較的主要依據。
   - `arXiv:2005.14263` 是 2020 年上傳版本；正式文章的出版年份是 2017。

6. Valavi, R., Elith, J., Lahoz-Monfort, J. J., &
   Guillera-Arroita, G. (2019). `blockCV`: An R package for generating
   spatially or environmentally separated folds for k-fold cross-validation
   of species distribution models. *Methods in Ecology and Evolution, 10*(2),
   225-232.
   [https://doi.org/10.1111/2041-210X.13107](https://doi.org/10.1111/2041-210X.13107)
   - 支援用 variogram spatial autocorrelation range 建立候選 block/buffer
     尺度。雖然案例是物種分布模型，作者明確說明方法可用於其他空間模型。


### 國家尺度投影座標

7. Snyder, J. P. (1987). *Map Projections—A Working Manual*. U.S.
   Geological Survey Professional Paper 1395.
   [https://doi.org/10.3133/pp1395](https://doi.org/10.3133/pp1395)
   - 支援依研究地區選擇合適的 projected CRS，而不是直接以經緯度角度或
     Web Mercator 代替實際距離。本專案將投影輸出的線性單位統一轉成 km。

### 空間候選變數與 covariance structure

8. Meyer, H., Reudenbach, C., Wöllauer, S., & Nauss, T. (2019).
   Importance of spatial predictor variable selection in machine learning
   applications - Moving from data reproduction to spatial prediction.
   *Ecological Modelling, 411*, 108815.
   [https://doi.org/10.1016/j.ecolmodel.2019.108815](https://doi.org/10.1016/j.ecolmodel.2019.108815)
   - 支援在 spatial CV 內進行 predictor selection，避免只靠經緯度或高度
   空間自相關的變數重製訓練資料。本專案的 elevation 與 terrain
   predictor 候選流程以此作為方法背景。

## 空間解釋變數資料來源

- 土地覆蓋比例來自
  [ESA Climate Change Initiative Land Cover v2.0.7cds](https://climate.esa.int/en/projects/land-cover/data/)；
  使用 2000 年 reference epoch、300 m FAO LCCS 圖資，計算每個 TCCIP
  GRID 內都市、森林、農業、水域與其他類別的連續面積比例。
- 海岸距離來自
  [NOAA/NCEI GSHHG v2.3.7](https://www.ngdc.noaa.gov/mgg/shorelines/shorelines.html)；
  使用 intermediate-resolution level-1 land/ocean boundary，將海岸線與
  GRID 中心投影至 TWD97 / TM2（EPSG:3826）後，計算最近海岸距離（km）。
- 原始檔案、處理後欄位與重製指令詳見
  [`data/spatial_predictors/README.md`](data/spatial_predictors/README.md)。

## 快速開始

```bash
conda env create -f environment.yml
conda activate gev-nn

python src/tccip_grid_preprocessing.py
python src/terrain_predictors.py
python src/coast_distance_predictor.py
python src/spatial_predictor_selection.py
```

真實 GRID 的 canonical preprocessing 已整合為：

```text
notebooks/data_preprocessing.ipynb
src/data_preprocessing_pipeline.py
```

預設沿用已建立的 annual-max table 並重建 11 quantiles、正確 shape sign、
地形、土地覆蓋、海岸距離及 raw spatial diagnostics：

```bash
python src/data_preprocessing_pipeline.py
```

只有在原始月資料變更時才需要從 65 年 CSV 全部重建：

```bash
python src/data_preprocessing_pipeline.py --rebuild-temperature
```

舊測站式空間模擬可執行：

```bash
python src/simulate_spatial_gev.py
```

論文主流程的獨立 downstream simulation 使用：

```text
notebooks/downstream_spatial_simulation.ipynb
src/downstream_spatial_simulation.py
```

此流程固定既有 NN 權重，從已知 RBF、Matérn 與
elevation/anisotropy 三種 GEV 空間真值重新產生 45 年 annual maxima，
再以 nested buffered spatial CV 選擇 GP，分別評估參數與
$RL_{50}$、$RL_{100}$ 對已知真值的誤差。pilot 完成後再將 Monte Carlo
replicates 提高至 100--200 次。
`SimulationConfig(n_jobs=-2)` 會以 replicate 為單位使用全部 CPU 邏輯核心
減一，並在每次模擬完成時輸出目前完成數。

若要訓練加入 DeepExtrema-style constraint penalty 的模型：

```bash
python src/constraint_penalty_train.py
```

若要對 safety margin 做 grid search：

```bash
python src/grid_search_safety_margin.py --mode xi --margins 0.0,0.01,0.03,0.05
```

若要比較 Fast transformation 中 `exp(delta)` 的安全距離，請分開跑另一組：

```bash
python src/grid_search_safety_margin.py --mode delta --delta-margins 0.0,0.01,0.03,0.05
```

這兩組建議分開比較，不建議一開始做雙參數 grid search，因為研究問題是要判斷「限制 $\xi$ 可行區間」或「限制 `exp(delta)` slack」哪一種對參數 RMSE 與 return level 較有幫助。

## NN 訓練架構

原始 NN 不是只使用 3 個分位數(理論可證明三個 quantile 可以 shape 出 GEV function)，而是固定使用 11 個 quantiles 作為輸入：

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

## Constraint Penalty 版本

新增程式：

```text
src/constraint_penalty_train.py
```

這個版本保留原本 Fast NN 的輸出形式：

```text
11 個 standardized quantiles -> sc_loc, delta, c
```

其中 `c` 是 `scipy.stats.genextreme` 的 shape 參數，和極值理論中的 $\xi$ 關係為：

```text
xi = -c
```

DeepExtrema 的想法是從 GEV support constraint：

```text
1 + xi * (y - mu) / sigma > 0
```

推出對 $\xi$ 的可行上下界：

```text
xi_lower <= xi <= xi_upper
```

因此本實驗在原本 Fast loss 上加入 soft penalty：

```text
penalty =
ReLU(xi_lower - xi)^2
+
ReLU(xi - xi_upper)^2
```

新的訓練目標為：

```text
total_loss = fast_loss + lambda * penalty
```

其中 `fast_loss` 仍然是 `sc_loc`、`delta`、`c` 的 MSE。這代表模型架構沒有改成 DeepExtrema，而是把 DeepExtrema 的 $\xi$ 可行範圍作為訓練時的額外限制。

目前程式使用 safety margin 版本，也就是不只要求 $\xi$ 落在原始上下界內，而是要求它離上下界各至少保留 0.01：

```text
xi_lower + 0.01 <= xi <= xi_upper - 0.01
```

這比直接縮小整個可行區間溫和，目的是避免預測值貼近 GEV support boundary，同時降低丟失極端尾端資訊的風險。

因為計算 $\xi_{\text{lower}}$ 和 $\xi_{\text{upper}}$ 需要每筆樣本的最小值與最大值，所以此程式會另外產生含有 standardized sample min/max 的訓練資料：

```text
data/simulated/gev_train_valid_constraint_seed111.npz
```

訓練完成後會輸出：

```text
models/best_constraint_penalty_model.pth
results/histories/constraint_penalty_history.csv
```

訓練完後可用以下 notebook 比較 baseline 與 constraint penalty 模型：

```text
notebooks/constraint_penalty_comparison.ipynb
```

比較內容包含 validation set 上的 standardized $\mu$、$\sigma$、$\xi$ 誤差，以及 $\xi$ 是否違反 DeepExtrema-style 可行上下界。

若要自動比較不同 safety margin，可執行：

```text
src/grid_search_safety_margin.py
```

它會依序訓練多個 margin，例如 `m=0.0, 0.01, 0.03, 0.05`，並輸出：

```text
results/tables/safety_margin_grid_search_xi_summary.csv
results/tables/safety_margin_grid_search_xi_metrics.csv
results/tables/safety_margin_grid_search_delta_summary.csv
results/tables/safety_margin_grid_search_delta_metrics.csv
models/best_constraint_penalty_m0p0300_model.pth
```

其中 `safety_margin_grid_search_metrics.csv` 會列出每個 margin 在 validation set 上的 $\mu$、$\sigma$、$\xi$ RMSE、MAE、Bias 與 Correlation。

此 grid search 也會計算 return level 誤差：

```text
RL10
RL50
RL100
```

因為極端事件預測最終通常關心高重現期水準，所以除了參數誤差，也應比較 return level RMSE，檢查 constraint 或 margin 是否壓掉極端尾端資訊。

## Constraint 顯著性檢定

若要檢驗最佳 $\xi$ margin 與 $\exp(\delta)$ margin 相較於 Baseline 的改善是否顯著，可執行：

```bash
python src/test_constraint_significance.py --bootstrap 2000
```

此檢定使用同一批 validation samples，因此採用配對方法：

```text
paired bootstrap 95% CI for RMSE difference
paired squared-error t-test
Holm multiple-testing correction
Wilcoxon signed-rank test as sensitivity analysis
```

主要虛無假設為：

```text
H0: candidate RMSE >= baseline RMSE
H1: candidate RMSE < baseline RMSE
```

輸出：

```text
results/tables/constraint_significance_tests.csv
```

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

## TCCIP 真實 GRID 資料流程

正式資料流程位於：

```text
src/tccip_grid_preprocessing.py
notebooks/tccip_grid_preprocessing.ipynb
notebooks/real_TCCIP_grid_data.ipynb
```

流程會讀取 `data/original_data/觀測_月資料_臺灣_最高溫/`，依 GRID
整理月最高溫序列、檢查 coverage、計算年最大值、產生 11 個 quantiles，
再用預訓練 NN 估計 $\hat\mu$、$\widehat{\log\sigma}$ 與 $\hat\xi$。
舊的 25 測站執行程式與 notebook 已移除；其原始與衍生資料仍保留在
`data/`，供資料追溯使用。

## 空間預測與驗證

GRID 參數使用 Gaussian process 建模，並以 buffered spatial
cross-validation 比較候選 mean structure 與 covariance kernel。最終選擇
依據是 nested buffered spatial cross-validation 的 out-of-fold 參數與
return-level 誤差。Inner loop 選 predictor、kernel 與超參數；outer loop
只負責估計未見地理區域的泛化誤差。

### 單一國家的統一空間尺度

經緯度只保留作為原始資料鍵值、GRID ID 與 CRS 轉換來源。所有具有
物理距離意義的計算，均先轉換到適合研究國家的 projected CRS：

```text
WGS84 longitude / latitude (EPSG:4326)
-> country-specific projected CRS
-> x_km, y_km
```

目前內建的國家設定為：

| 研究國家 | Projected CRS | 投影座標單位 |
|---|---|---|
| 臺灣 | TWD97 / TM2 zone 121 (`EPSG:3826`) | metre，再轉成 km |
| 冰島 | ISN2016 / Lambert 2016 (`EPSG:8088`) | metre，再轉成 km |

共用介面位於 `src/spatial_coordinates.py`：

```python
from spatial_coordinates import project_lonlat_to_km

taiwan_xy_km = project_lonlat_to_km(lon, lat, country="taiwan")
iceland_xy_km = project_lonlat_to_km(lon, lat, country="iceland")
```

其他單一國家必須明確提供該國官方 projected CRS：

```python
country_xy_km = project_lonlat_to_km(
    lon,
    lat,
    target_crs="EPSG:XXXX",
)
```

程式不會對未知國家自動套用 Web Mercator 或任意 UTM zone，以避免在
研究者不知情時引入距離變形。原有的 `project_lonlat_to_twd97_km()` 與
`add_twd97_km_columns()` 保留為臺灣分析的相容介面。

適用範圍包括：

- Gaussian-process covariance distance；
- coordinate-based K-means folds；
- semivariogram 與 directional variogram；
- buffered spatial cross-validation；
- Moran's I 鄰接距離與 residual spatial diagnostics；
- 高程候選模型與 return-level 空間圖。

GP 座標只減去 training-set 的中心，不分別除以兩軸標準差。因此空間距離
不會被扭曲，isotropic kernel 的 length scale 仍以 km 表示。現行優化設定
使用 `50 km` 作為初始 length scale，搜尋範圍為 `1--500 km`；固定距離
grid search 使用 `5--200 km` 候選值。

## 100 年重現期水準

GEV 的 $T$-year return level：

```text
z_T(s) = mu(s) + sigma(s) / xi(s) * {[-log(1 - 1/T)]^(-xi(s)) - 1}
```

實際的 $RL_{50}$、$RL_{100}$ 與 mixed-pipeline 比較位於
`notebooks/real_TCCIP_grid_data.ipynb` 與
`notebooks/elevation_gp_model_comparison.ipynb`。

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

但正式分析應該對齊 NN 的 11 quantile 架構。完整 QR11 分析已整理到主專案的延伸實驗資料夾：

```text
notebooks/quantile_ratio_11_quantile_analysis.ipynb
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
- 先模擬 monthly maxima，再由同一年 12 筆 monthly maxima 取最大值形成 annual maxima
- 依 GEV max-stability 分別建立 monthly 與 annual 的正確真實參數；兩者不共用相同的 `mu`、`sigma`
- 用 NN 估計測站參數
- 用 RBF / Matern Kriging 推估空間場
- 和 true parameter field 比較 RMSE、MAE、correlation

## 高程與 Isotropy 診斷

真實 TCCIP GRID 的高程與 GP 模型比較位於：

```text
notebooks/elevation_gp_model_comparison.ipynb
src/elevation_gp_analysis.py
src/terrain_predictors.py
```

baseline screening 先假設：

```text
linear elevation mean
+ stationary isotropic Matern(nu=1.5) covariance
```

再以 `x_km, y_km` 建立四個方向（0、45、90、135 度）的 directional
variograms。對每個 distance bin 計算四方向 semivariance 的差距，並在
stationary isotropic baseline 下進行 parametric simulations。若觀測統計量
超過模擬分布的 95% 上界，才將 geometric anisotropy 加入候選模型集合。

同一階段也檢查：

- nonlinear elevation mean；
- elevation-dependent variance；
- elevation-dependent spatial range；
- spatial-elevation distance。

這個 screening 只負責產生可辯護的候選模型，不直接決定最終模型。最終
選擇使用 nested buffered spatial cross-validation：inner loop 依 OOF RMSE
選擇 predictor 與 covariance kernel，outer loop 獨立評估最終表現。

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

年度／月度比較的可執行 notebook：

```text
notebooks/annual_monthly_max_comparison.ipynb
```

注意：上述模擬假設 12 個月份的 block maxima 同分布，用來控制 block
size 與樣本數。真實臺灣溫度有季節性，不應將 1–12 月直接合併成單一
stationary GEV。

## 專案結構

```text
fast_parameter_using_NN/
│
├── .gitignore                              # 排除模型暫存、快取與本機產生檔案
├── README.md                               # 專案方法、執行方式與檔案架構說明
├── environment.yml                        # Conda 環境與套件版本
├── requirements.txt                       # pip 安裝所需的 Python 套件
├── mle.R                                  # 以 ismev 計算 GEV MLE 信賴區間寬度
│
├── data/
│   ├── original_data/                     # 原始 TCCIP 月資料及保留的舊測站資料
│   ├── interim/                           # Figure 4 等分析尚未彙整的中間樣本
│   ├── processed/                         # 清理後的 GRID、GEV、GP 與 RL 分析資料
│   ├── simulated/                         # NN 訓練資料與空間 GEV 模擬結果
│   ├── spatial_predictors/                # 高程資料與衍生地形候選解釋變數
│   │   ├── raw/                           # 未加工的臺灣 GRID 高程資料
│   │   ├── processed/                     # 高程、坡度、坡向、起伏度等處理結果
│   │   └── README.md                      # 地形資料來源、欄位與產製方式
│   └── shapefile/                         # 臺灣邊界裁切與地圖繪製所需圖層
│
├── models/
│   ├── best_baseline_model.pth            # 原始 Fast GEV baseline NN 權重
│   └── best_constraint_penalty_model.pth  # 加入 GEV support penalty 的 NN 權重
│
├── notebooks/
│   ├── tccip_grid_preprocessing.ipynb     # TCCIP 原始月 GRID 清理與極值資料前處理
│   ├── real_TCCIP_grid_data.ipynb         # 真實 GRID 的 NN、GP、SKCV、RLO 與殘差驗證
│   ├── elevation_gp_model_comparison.ipynb # 高程、isotropy 與 GP 候選模型比較
│   ├── data_preprocessing.ipynb             # 真實 GRID、GEV 參數與空間 predictors 的統一前處理
│   ├── spatial_predictor_selection.ipynb    # 地形、土地覆蓋與海岸距離的 spatial FFS
│   ├── downstream_spatial_simulation.ipynb  # 已知真值下驗證 frozen NN、nested GP 與 RL
│   ├── annual_monthly_max_comparison.ipynb # 年度 45 筆與月度 540 筆模擬敏感度比較
│   ├── quantile_ratio_11_quantile_analysis.ipynb # NN 與 11 分位數比例估計法比較
│   └── constraint_penalty_comparison.ipynb # Baseline 與 constraint-penalty NN 比較
│
├── results/
│   ├── figures/                           # 論文、簡報與 notebook 使用的圖形
│   ├── tables/                            # Spatial CV、檢定、Moran's I 與 RL 表格
│   └── histories/                         # NN 訓練與 penalty 搜尋的 loss histories
│
├── src/
│   ├── annual_monthly_max_comparison.py   # 比較獨立年度與月度樣本的 NN/GP 估計
│   ├── baseline_train.py                  # 訓練原始 11-quantile Fast GEV NN
│   ├── block_maxima_comparison.py         # 建立相依的月最大值與年最大值比較資料
│   ├── bootstrap_nn.py                    # NN bootstrap、GEV MLE 與信賴區間工具
│   ├── constraint_penalty_train.py        # 訓練加入 GEV support penalty 的 NN
│   ├── directional_kernel_tests.py        # 執行六組 RBF/Matérn 方向性假設檢定
│   ├── downstream_spatial_simulation.py    # 獨立空間 GEV Monte Carlo 與 nested buffered CV
│   ├── elevation_gp_analysis.py           # 高程 GP、buffered SKCV、RL 與殘差診斷
│   ├── data_preprocessing_pipeline.py      # 原始 GRID 到 model-ready parameter table
│   ├── spatial_diagnostics.py              # raw／OOF directional 與 regional variograms
│   ├── generate_nn_bootstrap_csv.py       # 批次執行 NN bootstrap 並輸出 CSV
│   ├── generate_samples.py                # 產生論文 Figure 4 使用的模擬樣本
│   ├── gev_nn.py                          # 共用 NN 架構、輸入標準化與參數反轉換
│   ├── grid_search_safety_margin.py       # 搜尋 constraint safety/delta margin
│   ├── kriging_kernel_gridsearch.py       # 搜尋模擬資料 RBF/Matérn GP kernel
│   ├── plot_variograms.py                 # 計算並繪製模擬參數 empirical variogram
│   ├── prepare_daily_tmax_block_maxima.py # 從每日最高溫建立月與年 block maxima
│   ├── project_paths.py                   # 集中管理資料、模型與結果的標準路徑
│   ├── quantile_ratio_estimator.py        # 實作 3 與 11 分位數比例 GEV 估計
│   ├── simulate_data.py                   # 產生及切分 NN 使用的 GEV 模擬資料
│   ├── simulate_spatial_gev.py            # 模擬空間 GEV 曲面並驗證 NN/QR/GP
│   ├── spatial_coordinates.py             # WGS84 轉單一國家的 projected km 座標
│   ├── tccip_grid_preprocessing.py        # 真實 TCCIP GRID 批次前處理主程式
│   ├── terrain_predictors.py              # 由 DEM 計算坡度、坡向、起伏與粗糙度
│   └── test_constraint_significance.py    # 檢定 constraint NN 是否改善參數與 RL
│
├── reports/
│   ├── directional_kernel_test_report.md  # 六組 GP kernel 假設檢定報告
│   ├── tccip_analysis_summary.md           # TCCIP 資料與主要分析結果摘要
│   └── tccip_window_data_workflow.md       # 真實 GRID 完整資料與驗證流程
│
└── reference_paper/
    └── 3. Fast parameter estimation ...   # 11 分位數 Fast GEV NN 原始論文

```

## 後續工作

持續優化 non-isotropy 的問題，以及針對參數選擇建模優化

## 作者

Shu-Ming Chang  
National Central University
