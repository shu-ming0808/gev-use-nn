# Spatial predictor data

本資料夾集中管理 GP mean structure 與 spatial variable selection 使用的
候選解釋變數。經緯度仍保留作為資料鍵值，但所有距離、坡度與空間鄰域
計算統一使用 TWD97 / TM2 zone 121（EPSG:3826）。

## 目錄

- `raw/taiwan_grid_elevation_001deg.csv`
  - 原始臺灣 0.01 度高程網格。
- `processed/taiwan_terrain_predictors_001deg.csv`
  - 完整 0.01 度地形 predictor。
- `processed/tccip_grid_terrain_predictors.csv`
  - 對齊目前 TCCIP GRID 的地形 predictor，供 GP 與 Spatial FFS 使用。

## 衍生欄位

- `elevation_m`：高程，單位為公尺。
- `slope_deg`：以投影後公尺距離計算的坡度。
- `aspect_deg`：最陡下降方向，從北方順時針計算；平坦地為缺值。
- `northness`：`cos(aspect)`；平坦地設為 0。
- `eastness`：`sin(aspect)`；平坦地設為 0。
- `local_relief_m`：預設 5 x 5 原始網格內最高與最低高程之差。
- `terrain_ruggedness_m`：中心網格與周圍八格高程差的 RMS。

使用下列指令重新產生：

```powershell
python src/terrain_predictors.py
```
