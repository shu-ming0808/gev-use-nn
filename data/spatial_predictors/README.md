# Spatial predictor data

本資料夾集中管理 GP mean structure 與 spatial variable selection 使用的
候選解釋變數。經緯度仍保留作為資料鍵值，但所有距離、坡度與空間鄰域
計算統一使用 TWD97 / TM2 zone 121（EPSG:3826）。

## 目錄

- `raw/taiwan_grid_elevation_001deg.csv`
  - 原始臺灣 0.01 度高程網格。
  - 此檔目前是既有本地輸入；原始發布機關、網址與版本仍須由下載紀錄補齊，
    在確認前不宣稱為特定版本的 20 m DTM。
- `processed/taiwan_terrain_predictors_001deg.csv`
  - 完整 0.01 度地形 predictor。
- `processed/tccip_grid_terrain_predictors.csv`
  - 對齊目前 TCCIP GRID 的地形 predictor，包含 TPI，供 GP 與 Spatial FFS 使用。
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
- `processed/tccip_grid_tmax_event_day_rainfall.csv`
  - 以 `station + max_date` 將每個 GRID 的月最高溫發生日對到同日雨量。
- `processed/tccip_grid_rainfall_predictors.csv`
  - 1980 年後逐日雨量氣候指標，以及月最高溫事件日雨量的 GRID 摘要。
- `raw/atmosphere/`
  - CDS 下載的 AgERA5 0.1 度日風速、太陽輻射及 24 小時平均雲覆蓋。
  - 1980 年單年測試也是正式資料；完整下載時保留並跳過既有檔案，不需覆蓋。
- `processed/tccip_grid_atmospheric_predictors.csv`
  - 依每個 GRID 的月最高溫日期配對每日網格，以 bilinear interpolation
    評估於 TCCIP GRID 中心，再計算每個 GRID 的極端高溫事件平均。
- `processed/tccip_grid_tmax_event_day_atmosphere.csv`
  - 逐 GRID、逐年月保存 `max_date` 及同日風速、太陽輻射與雲覆蓋值；
    這是上述 GRID-level 平均 predictor 的可追溯明細。
- `processed/tccip_grid_atmospheric_alignment_audit.csv`
  - 保存來源解析度、涵蓋邊界、插值方法、外插數量與最近來源格點距離。

## 衍生欄位

- `elevation_m`：高程，單位為公尺。
- `slope_deg`：以投影後公尺距離計算的坡度。
- `aspect_deg`：最陡下降方向，從北方順時針計算；平坦地為缺值。
- `northness`：`cos(aspect)`；平坦地設為 0。
- `eastness`：`sin(aspect)`；平坦地設為 0。
- `local_relief_m`：預設 5 x 5 原始網格內最高與最低高程之差。
- `tpi_m`：中心格高程減去周圍 5 x 5 鄰域（不含中心格）的平均高程；
  正值偏山脊、負值偏谷地。
- `terrain_ruggedness_m`：中心網格與周圍八格高程差的 RMS。
- `urban_ratio`：ESA CCI class 190（urban areas）的面積比例。
- `forest_ratio`：classes 50--90 與 160、170 的 tree-cover 面積比例。
- `agriculture_ratio`：classes 10、11、12、20、30 的農業面積比例。
- `water_ratio`：class 210（水體，包含海水）的面積比例。
- `other_ratio`：未被上述四組納入的有效像元比例。
- `coast_distance_km`：GRID 中心點至 GSHHG level-1 最近海岸線的
  TWD97 平面距離，單位為公里。
- `mean_annual_precip_mm`：1980 年後各年有效日雨量總和的跨年平均。
- `rain_wet_day_ratio`：有效日中雨量至少 1 mm 的比例。
- `tmax_event_rain_mean_mm`：各 GRID 月最高溫發生日的同日雨量平均。
- `tmax_event_rain_wet_ratio`：月最高溫事件日中雨量至少 1 mm 的比例。
- `tmax_event_wind_mean_mps`：各 GRID 月最高溫發生日的 AgERA5 10 m
  日平均 scalar wind speed，再跨事件取平均，單位為 m/s。
- `tmax_event_solar_radiation_mean_mj_m2`：各 GRID 月最高溫發生日的
  AgERA5 地表太陽能量，再跨事件取平均；由 J/m2/day 轉為 MJ/m2/day。
- `tmax_event_agera5_cloud_cover_mean_fraction`：各 GRID 月最高溫發生日的
  AgERA5 24 小時平均總雲覆蓋比例，再跨事件取平均，範圍為 0--1。
  此欄位不是「有雲時數頻率」，既有下載檔名稱中的 `cloud_frequency`
  僅為歷史命名。

事件日雨量與事件日大氣變數皆使用由 Tmax 決定的 `max_date`，屬
outcome-conditioned predictor。只有在應用階段同樣能取得該 GRID 的 Tmax
發生日時才可使用；若研究目標是完全未知地區的盲目空間預測，應避免使用
test response 才能決定的日期資訊。

目前空間 GP 每個 GRID 只有一組 GEV 參數，所以逐月事件日大氣資料先取
GRID-level mean，形成固定的 mean-structure predictor。這個平均是「極端
高溫發生日的平均大氣條件」，不是所有日子的氣候平均。未來若要保留每個
事件的時間差異，應改建 nonstationary spatiotemporal GEV，例如讓 GEV
參數隨事件日風速、輻射與雲量改變，而不是先取 GRID-level mean。

土地覆蓋固定使用 2000 年參考面；該產品是以 2000 年為中心的 reference
epoch，而非只使用 2000 日曆年的單一年影像。這些變數是空間解釋變數，不代表
1980--2024 年間土地覆蓋維持不變。沿海 GRID 的 `water_ratio` 可能包含
海域；這是刻意保留的 coastal exposure，而不是缺值。
建模時直接保留各土地覆蓋類型的連續面積比例，不使用 0.5 門檻轉成單一類別；
`other_ratio` 作為組成資料的參考類別，避免五個比例同時進入線性 mean
structure 所造成的完全共線性。

AgERA5 的原生解析度為 0.1 度，比 TCCIP 0.05 度粗。程式不切 polygon 或
複製成假的 0.05 度觀測值，而是把
大氣場視為連續場，在來源網格涵蓋範圍內用 bilinear interpolation 評估
TCCIP cell centre；禁止外插，並輸出 alignment audit。因此結果只能稱為
「AgERA5 在 TCCIP 中心的插值值」，不能稱為原生 TCCIP 解析度資料。

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

以月最高溫發生日配對逐日雨量，並建立降雨候選變數：

```powershell
python src/rainfall_predictors.py --start-year 1980
```

下載並建立風速、太陽輻射及雲候選變數（需先設定 CDS API key 並接受
AgERA5 授權條款）：

```powershell
# 先確認帳號、授權與請求欄位皆可使用
python src/atmospheric_predictors.py --download --start-year 1980 --end-year 1980 --batch-years 9

# 1980 成功後下載剩餘正式資料
python src/atmospheric_predictors.py --download --download-only --start-year 1981 --end-year 2024 --batch-years 9
```

若已從 CDS 網頁下載 NetCDF/zip 至 `raw/atmosphere`，省略 `--download`
即可只做解壓縮、事件日配對、中心插值、GRID 平均與邊界稽核。

CDS 原始資料不得放進 `data/original_data/`；該目錄只保存 TCCIP 原始觀測。
本下載器將 CDS zip 與解壓內容放在 `raw/atmosphere/`，並將最終 predictor
表與 alignment audit 寫入 `processed/`。每個成功檔案以資料集、變數與年份
命名；重新執行時，已存在的檔案會顯示 `skip existing`，因此可由中斷處續跑。
下載器預設每個變數以九個連續年份為一組，再按月份拆成 cost-safe CDS
requests；既有完整年度檔與已完成的月別批次檔都會納入續傳判斷。1980
已完成後，1981--2024 會形成五個年份組，三個變數最多共 180 次月別
request。這樣避免九年全年單一請求觸發 CDS `cost limits exceeded`。
單年測試可產生事件配對表供檢查，但 Spatial FFS 會核對每個 GRID 的事件
涵蓋率；1980--2024 尚未完整時會停止，避免把 1980 單年平均誤當成正式
長期 predictor。

使用固定 K-means 五區與 response-specific buffer，對所有候選變數與
RBF、Matérn $\nu=0.5,1.5,2.5$ 執行聯合 spatial forward feature
selection：

```powershell
python src/spatial_predictor_selection.py --n-jobs -2
```

全部下載完成後，也可用單一入口先稽核再執行到選模：

```powershell
python src/real_grid_modeling_pipeline.py --check-only
python src/real_grid_modeling_pipeline.py --n-jobs -2
```

只重算相關性與 VIF、不執行耗時 GP：

```powershell
python src/spatial_predictor_selection.py --audit-only
```

每一步對每個 predictor set 使用完全相同的 folds、buffer 與 training cap
比較全部 kernel，選擇 pooled OOF RMSE 最小的 predictor-set/kernel 組合；
新變數加入後仍可改選 kernel。每一步預設必須使整體 OOF RMSE 至少下降 1%，
避免固定分割下極小的數值改善
造成過度選取；這是預先設定的 practical-improvement rule，不是顯著水準。
輸出的 RMSE 是變數選擇階段的分數，不直接當成最終無偏泛化誤差。完成
predictor 與 kernel 選擇後，仍應另外執行 repeated 或 nested buffered
spatial CV。

共線性不是只靠 FFS 自動消失。本流程另輸出 Pearson、Spearman 與 VIF：

1. 候選池先檢查 $|\rho|\geq0.7$，而 $|\rho|>0.9$ 視為近乎重複訊息；
2. 每個 proposed predictor set 都在各 buffered training fold 內計算 VIF；
3. 任一 training fold 的最大 VIF 超過 5，該 predictor set 不進行 GP 比較；
4. 通過後才以 pooled OOF RMSE 執行 joint predictor/kernel FFS。

這個順序仿照 Kagawa-Viviani and Giambelluca (2020) 的公開分析程式：先看
covariate correlation matrix、移除相關大於 0.9 的冗餘變數，再檢查候選模型
VIF，並將 VIF 5 作為警戒。相關與 VIF是 mean-structure 可解釋性的保護；
buffered Spatial CV 才是本研究的預測選模依據。

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
- TCCIP 網格化觀測日資料（降雨量）：與最高溫相同的 0.05 度 GRID，
  以座標建立一致的 `station` 後逐日配對。
- [Copernicus AgERA5](https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators?tab=overview)：
  0.1 度、1979 年至今的 daily wind speed、solar radiation flux 與
  24-hour mean cloud cover；本研究使用 1980--2024。AgERA5 `cloud_cover`
  的正確定義是日平均雲覆蓋，不是雲發生頻率。
- Kagawa-Viviani, A. K. and Giambelluca, T. W. (2020),
  *Spatial Patterns and Trends in Surface Air Temperatures and Implied Changes
  in Atmospheric Moisture Across the Hawaiian Islands, 1905--2017*,
  DOI: 10.1029/2019JD031571；公開程式 DOI: 10.5281/zenodo.3592085。
- Dormann et al. (2013), *Collinearity: a review of methods to deal with it
  and a simulation study evaluating their performance*, DOI:
  10.1111/j.1600-0587.2012.07348.x。
