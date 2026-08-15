# 使用神經網路快速估計 GEV 參數

## 專案目的

本專案使用 11 個經驗分位數與預訓練神經網路估計 GEV 參數，再以 Gaussian process（GP）重建臺灣 TCCIP GRID 的空間參數曲面與 return levels。

```text
逐日最高溫
  → 月最高溫與發生日
  → 年最大值
  → 11 個分位數
  → NN 估計 μ、log σ、ξ
  → 空間變數與 GP 候選模型
  → Buffered Spatial CV
  → OOF 參數、RL50、RL100 與殘差診斷
```

## 資料與空間變數

| 類別 | 資料來源 | 處理 |
|---|---|---|
| 逐日最高溫 | TCCIP 0.05° GRID | 建立月最高溫、發生日與年最大值 |
| 逐日降雨 | TCCIP 0.05° GRID | 降雨氣候值與極端高溫日降雨 |
| 高程 | 內政部 DTM | 高程、坡度、坡向、TPI、起伏度與崎嶇度 |
| 土地覆蓋 | ESA CCI Land Cover 2000 | 都市、森林、農業與水域的連續面積比例 |
| 海岸線 | GSHHG | GRID 中心至最近海岸距離 |
| 風速、太陽輻射、雲量 | Copernicus CDS（AgERA5） | 先配對月最高溫發生日，再彙整為 GRID-level 平均候選變數 |

土地覆蓋保留連續比例，不用 `0.5` 門檻轉成單一類別。所有資料對齊至同一個 TCCIP GRID；臺灣本島座標使用 TWD97/TM2（EPSG:3826）並轉為 km。

## 現行選模邏輯

| 項目 | 設定 |
|---|---|
| Response | NN-derived $\hat\mu$、$\widehat{\log\sigma}$、$\hat\xi$ |
| Geographic folds | Coordinate K-means，正式流程 $K=5$ |
| Spatial separation | Response-specific buffer distance |
| Training cap | 每 fold 最多 800 GRID |
| Predictor selection | Buffered Spatial-CV 內的 grouped forward feature selection |
| Collinearity | 每個 training fold 檢查 VIF，預設上限 5 |
| GP kernels | RBF、Matérn $\nu=0.5,1.5,2.5$ |
| Primary criterion | Pooled out-of-fold RMSE |
| Diagnostics | MAE、Bias、fold stability、Moran's $I$、residual variogram |
| Final quantities | $RL_{50}$ 與 $RL_{100}$ |

Predictor set 與 kernel 在相同 folds、buffer 與 training cap 下一起比較。AIC/BIC 不是主要選模依據。現行結果屬開發階段；正式泛化誤差應再使用 repeated nested buffered Spatial CV。

GP 座標只減去 training-set 中心，不分別除以兩軸標準差，因此 isotropic length scale 仍以 km 表示。預設初始 length scale 為 50 km，優化範圍為 1--500 km。

## 快速開始

```powershell
conda env create -f environment.yml
conda activate gev-nn
```

### 1. 下載或續傳大氣資料

```powershell
python .\src\atmospheric_predictors.py --download --download-only `
  --start-year 1980 --end-year 2024 --batch-years 9
```

### 2. 檢查資料完整性

```powershell
python .\src\real_grid_modeling_pipeline.py --check-only
```

### 3. 建立 model-ready GRID，但不選模

```powershell
python .\src\real_grid_modeling_pipeline.py --prepare-only --n-jobs -2
```

### 4. 執行完整選模

```powershell
python .\src\real_grid_modeling_pipeline.py --n-jobs -2
```

### 5. $K=3,4,5,6,7$ 敏感度實驗

```powershell
python .\src\k_sensitivity_experiment.py `
  --k-values 3 4 5 6 7 --n-jobs -2
```

### 6. 輸出四階段 OOF 比較圖

```powershell
python .\src\compare_four_stage_oof.py --n-jobs -2 `
  --output-directory "C:\Users\User.DESKTOP-4RV84M1\Desktop\picture"
```

## 主要檔案

### 程式

| 檔案 | 用途 |
|---|---|
| `src/real_grid_modeling_pipeline.py` | 真實資料正式入口 |
| `src/data_preprocessing_pipeline.py` | 年最大值、NN 參數與 model-ready table |
| `src/atmospheric_predictors.py` | AgERA5 下載、解壓、日期配對與彙整 |
| `src/terrain_predictors.py` | 高程與地形變數 |
| `src/land_cover_predictors.py` | 土地覆蓋面積比例 |
| `src/coast_distance_predictor.py` | 海岸距離 |
| `src/rainfall_predictors.py` | 降雨氣候與事件日變數 |
| `src/spatial_coordinates.py` | 投影座標與 km 尺度 |
| `src/spatial_diagnostics.py` | Directional / regional variograms |
| `src/spatial_predictor_selection.py` | VIF、FFS、kernel 與 buffered Spatial-CV |
| `src/k_sensitivity_experiment.py` | $K=3$--$7$ 實驗 |
| `src/downstream_spatial_simulation.py` | 已知真值的獨立空間模擬 |
| `src/compare_four_stage_oof.py` | 四種 predictor structures 的 OOF 比較 |

### Notebooks

| Notebook | 用途 |
|---|---|
| `notebooks/data_preprocessing.ipynb` | 資料前處理與候選變數稽核 |
| `notebooks/real_TCCIP_grid_data.ipynb` | Variogram、folds、buffered SKCV 與 residual diagnostics |
| `notebooks/spatial_predictor_selection.ipynb` | 候選變數與 kernel 選擇 |
| `notebooks/elevation_gp_model_comparison.ipynb` | 無變數、高程與多變數 GP 比較 |
| `notebooks/downstream_spatial_simulation.ipynb` | 參數、kernel 與 return-level 恢復模擬 |

## 主要輸出

| 路徑 | 內容 |
|---|---|
| `data/processed/` | 月／年最大值、GEV 參數與 model-ready tables |
| `data/spatial_predictors/processed/` | 地形、覆蓋、海岸、降雨與大氣變數 |
| `results/spatial_predictor_selection/` | 候選模型、OOF predictions 與 return levels |
| `results/k_sensitivity/` | 候選 $K$ 的誤差、穩定性與選模結果 |
| `results/figures/` | 程式產生的圖 |

## 參考文獻

1. Rai et al. (2024). *Fast parameter estimation of generalized extreme value distribution using neural networks*. Environmetrics, 35(3), e2845.
2. Roberts et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography, 40(8), 913--929.
3. Brenning (2012). *Spatial cross-validation and bootstrap for the assessment of prediction rules in remote sensing: The R package sperrorest*.
4. Pohjankukka et al. (2017). *Estimating the prediction performance of spatial models via spatial k-fold cross validation*.
5. Valavi et al. (2019). *blockCV: An R package for generating spatially or environmentally separated folds*.
6. Meyer et al. (2019). *Importance of spatial predictor variable selection in machine learning applications*.
7. Snyder (1987). *Map Projections—A Working Manual*. USGS Professional Paper 1395.
8. Tibshirani, Walther, and Hastie (2001). *Estimating the number of clusters in a data set via the gap statistic*.

## 注意事項

- `RL50` 與 `RL100` 的真實資料誤差是相對 NN-derived reference，不是相對不可觀測的真實 GEV 參數。
- 殘差 Moran's $I$ 或 variogram 顯著代表尚有未解釋空間結構，不等於 fold 設計必然錯誤。
- 目前大氣候選變數是極端高溫發生日的 GRID-level 平均。未來若保留逐月變化，應改建 nonstationary spatiotemporal GEV。
