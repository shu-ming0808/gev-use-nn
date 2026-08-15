# 空間候選變數

本資料夾管理 GP mean structure 候選變數。經緯度保留為鍵值；距離、坡度與空間鄰域使用 TWD97/TM2（EPSG:3826）。

## 資料來源與處理

| 變數 | 來源 | 處理 |
|---|---|---|
| Elevation | 本地臺灣高程 GRID | 對齊 TCCIP GRID |
| Slope / aspect / TPI / relief / ruggedness | Elevation | 投影後由 DEM 計算 |
| Urban / forest / agriculture / water ratio | ESA CCI Land Cover 2000，300 m | 計算每個 TCCIP GRID 內面積比例 |
| Distance to coast | GSHHG coastline | GRID 中心至最近海岸距離 |
| Rainfall | TCCIP 逐日降雨 | 氣候值與月最高溫事件日降雨 |
| Wind / solar radiation / cloud | Copernicus CDS AgERA5 | 配對 `max_date`，再彙整為 GRID-level 平均 |

`max_date` 由 Tmax response 決定，屬 outcome-conditioned 變數。若目標是完全未知地區的盲目預測，應排除這類變數或改建時空 GEV。

## 建立變數

```powershell
python .\src\terrain_predictors.py
python .\src\land_cover_predictors.py --force-download
python .\src\coast_distance_predictor.py --force-download
python .\src\rainfall_predictors.py --start-year 1980
```

AgERA5 先測試一年：

```powershell
python .\src\atmospheric_predictors.py --download `
  --start-year 1980 --end-year 1980 --batch-years 9
```

確認成功後續傳：

```powershell
python .\src\atmospheric_predictors.py --download --download-only `
  --start-year 1981 --end-year 2024 --batch-years 9
```

## 選模前稽核

| 檢查 | 規則 |
|---|---|
| Pairwise correlation | $|\rho|\geq0.7$ 標示；$|\rho|>0.9$ 視為近乎重複 |
| VIF | 每個 buffered training fold 的上限為 5 |
| Model comparison | 通過稽核後，以 pooled OOF RMSE 比較 predictor set 與 kernel |

```powershell
python .\src\real_grid_modeling_pipeline.py --check-only
python .\src\real_grid_modeling_pipeline.py --n-jobs -2
```
