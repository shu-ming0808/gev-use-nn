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
- `raw/land_cover/esa_cci_land_cover_2000_taiwan.tif`
  - 從 ESA CCI Land Cover 2000 全球 COG 擷取的臺灣範圍。
  - 解析度為 300 m，分類系統為 FAO LCCS。
- `processed/tccip_grid_land_cover_2000.csv`
  - 每個 0.05 度 TCCIP GRID 內的面積加權土地覆蓋比例。
  - 包含都市、森林、農業、水域與其他類別，五類比例加總為 1。
- `raw/coastline/GSHHS_i_L1/`
  - GSHHG v2.3.7 intermediate-resolution level-1 海岸線。
  - 原始資料使用 WGS84，僅在計算時投影為 TWD97 / TM2。
- `processed/tccip_grid_coast_distance.csv`
  - 每個 TCCIP GRID 中心點至最近海岸線的距離，單位為公里。

## 衍生欄位

- `elevation_m`：高程，單位為公尺。
- `slope_deg`：以投影後公尺距離計算的坡度。
- `aspect_deg`：最陡下降方向，從北方順時針計算；平坦地為缺值。
- `northness`：`cos(aspect)`；平坦地設為 0。
- `eastness`：`sin(aspect)`；平坦地設為 0。
- `local_relief_m`：預設 5 x 5 原始網格內最高與最低高程之差。
- `terrain_ruggedness_m`：中心網格與周圍八格高程差的 RMS。
- `urban_ratio`：ESA CCI class 190（urban areas）的面積比例。
- `forest_ratio`：classes 50--90 與 160、170 的 tree-cover 面積比例。
- `agriculture_ratio`：classes 10、11、12、20、30 的農業面積比例。
- `water_ratio`：class 210（水體，包含海水）的面積比例。
- `other_ratio`：未被上述四組納入的有效像元比例。
- `coast_distance_km`：GRID 中心點至 GSHHG level-1 最近海岸線的
  TWD97 平面距離，單位為公里。

土地覆蓋固定使用 2000 年參考面；該產品是以 2000 年為中心的 reference
epoch，而非只使用 2000 日曆年的單一年影像。這些變數是空間解釋變數，不代表
1980--2024 年間土地覆蓋維持不變。沿海 GRID 的 `water_ratio` 可能包含
海域；這是刻意保留的 coastal exposure，而不是缺值。
建模時直接保留各土地覆蓋類型的連續面積比例，不使用 0.5 門檻轉成單一類別；
`other_ratio` 作為組成資料的參考類別，避免五個比例同時進入線性 mean
structure 所造成的完全共線性。

使用下列指令重新產生：

```powershell
python src/terrain_predictors.py
```

重新下載臺灣範圍並計算 2000 年土地覆蓋比例：

```powershell
python src/land_cover_predictors.py --force-download
```

下載 GSHHG 海岸線並計算 GRID 中心的最近海岸距離：

```powershell
python src/coast_distance_predictor.py --force-download
```

使用固定 K-means 五區與 response-specific buffer，對所有候選變數執行
spatial forward feature selection：

```powershell
python src/spatial_predictor_selection.py
```

此步驟將 kernel 固定為前一階段的選擇，專門比較 GP mean structure；
每一步預設必須使整體 OOF RMSE 至少下降 1%，避免固定分割下極小的數值改善
造成過度選取；這是預先設定的 practical-improvement rule，不是顯著水準。
輸出的 RMSE 是變數選擇階段的分數，不直接當成最終無偏泛化誤差。完成
predictor 與 kernel 選擇後，仍應另外執行 repeated 或 nested buffered
spatial CV。

資料來源：

- [ESA Climate Change Initiative Land Cover](https://climate.esa.int/en/projects/land-cover/data/)
  v2.0.7cds：1992--2020 年全球年度土地覆蓋圖，300 m。
- [Microsoft Planetary Computer ESA CCI catalog](https://planetarycomputer.microsoft.com/dataset/group/esa-cci-lc)：
  提供依區域讀取的 Cloud Optimized GeoTIFF。
- [NOAA/NCEI Shoreline and Coastline Databases](https://www.ngdc.noaa.gov/mgg/shorelines/shorelines.html)：
  GSHHG 結合 GSHHS 海岸線與 WDBII 河流、國界資料；本研究使用
  v2.3.7 intermediate-resolution level-1 land/ocean boundary。
- [Generic Mapping Tools GSHHG release](https://github.com/GenericMappingTools/gshhg-gmt/releases/tag/2.3.7)：
  本專案實際下載的 GSHHG v2.3.7 ESRI Shapefile 發布頁。
