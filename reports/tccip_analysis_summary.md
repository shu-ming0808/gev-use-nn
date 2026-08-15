# TCCIP 分析摘要

## 資料

| 項目 | 結果 |
|---|---:|
| 分析年份 | 1980--2024 |
| 年最大值樣本數 | 45 |
| 原始 GRID | 1412 |
| 本島選模 GRID | 1385 |
| Coverage 門檻 | 80% |
| 空間座標 | TWD97/TM2，km |

## 核心輸出

| 路徑 | 內容 |
|---|---|
| `data/processed/annual_max_grid_temperature.csv` | 年最大值 |
| `data/processed/grid_station_gev_params_with_loc.csv` | NN-derived GEV 參數 |
| `data/processed/model_ready_grid_parameters.csv` | 參數與空間候選變數 |
| `results/spatial_predictor_selection/` | OOF 預測、選模與 return levels |
| `results/figures/` | 診斷與比較圖 |

## 驗證

- 以 buffered Spatial CV 同時比較 predictor set 與 GP kernel。
- 以 OOF RMSE、MAE、Bias、Moran's $I$ 與 residual variogram 評估。
- 獨立 downstream simulation 檢查參數、kernel 與 return-level 恢復能力。
