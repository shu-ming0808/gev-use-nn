# TCCIP GRID 處理流程

## 正式流程

```text
TCCIP 逐日最高溫
  → 月最高溫與發生日
  → 年最大值
  → 11 個經驗分位數
  → 預訓練 NN
  → μ、log σ、ξ
  → 對齊空間候選變數
  → Buffered Spatial CV + FFS + kernel selection
  → OOF 參數與 RL50 / RL100
  → Moran's I 與 residual variogram
```

## 主要入口

| 檔案 | 用途 |
|---|---|
| `src/prepare_daily_tmax_block_maxima.py` | 建立月最高溫、發生日與年最大值 |
| `src/data_preprocessing_pipeline.py` | NN 參數與 model-ready GRID |
| `src/real_grid_modeling_pipeline.py` | 前處理、稽核與正式選模 |
| `src/spatial_predictor_selection.py` | VIF、FFS、kernel 與 OOF 預測 |

## 執行

```powershell
python .\src\real_grid_modeling_pipeline.py --check-only
python .\src\real_grid_modeling_pipeline.py --prepare-only --n-jobs -2
python .\src\real_grid_modeling_pipeline.py --n-jobs -2
```

## 分析 Notebooks

| Notebook | 用途 |
|---|---|
| `notebooks/data_preprocessing.ipynb` | 資料與候選變數稽核 |
| `notebooks/real_TCCIP_grid_data.ipynb` | Folds、buffer、SKCV 與殘差診斷 |
| `notebooks/spatial_predictor_selection.ipynb` | Predictor/kernel 選擇 |
| `notebooks/elevation_gp_model_comparison.ipynb` | Predictor structures 與 return levels 比較 |
| `notebooks/downstream_spatial_simulation.ipynb` | 已知真值的獨立模擬 |
