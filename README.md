# 使用神經網路快速估計時間非定常 GEV 參數

## 專案目的與目前狀態

本專案以 **ERA5 全球 1976–2025 年、每個 GRID 的 50 筆年最大值**為分析目標。Cheng NN 快速估計時間變動 GEV 的五個係數，再銜接時間結構診斷與 GP／Spatial CV，最後估計指定年份的 return levels。

目前選定 **21D 輸入、三層小模型、10 萬組四情境混合模擬資料**。train／validation／test = **80,000／10,000／10,000**。checkpoint 由 validation 選定，已在同一批 10,000 組 test 與五係數 MLE 比較。這是模擬估計能力的評估，**不代表全球真實資料或五係數 Spatial CV 已完成**。

舊版「11 分位數 → 三個固定 GEV 參數」NN、兩個 `.pth` 權重及依賴它的臺灣前處理／校準模擬入口已移除。原始資料與已完成的歷史結果保留於本機。含時間的 17D／21D／50D 對照實驗保留，用於重現選型依據；它們不等於已退役的不含時間 NN。

## 最終模型參數

| 項目 | 選定設定 |
|---|---|
| 每組觀測 | 50 筆有時間順序的年最大值，1976–2025 |
| 模擬總數／分割 | 100,000；train 80,000／validation 10,000／test 10,000 |
| 四種情境 | 完全定常、只有位置變動、只有尺度變動、兩者變動，各 25% |
| 序列標準化 | 全段 median／IQR；不在各時段分別標準化 |
| Input | 21D：全段 11 分位數＋5 個時段各一組 median、IQR |
| 時段 | 1976–1985、1986–1995、1996–2005、2006–2015、2016–2025 |
| Hidden layers | `128 → 128 → 64`，3 層 |
| 完整架構 | `21 → 128 → 128 → 64 → 5` |
| 可訓練參數 | 27,909 |
| Activation | 隱藏層 ReLU；輸出層 Linear，不使用 softmax |
| Output | $\mu_0,\beta_\mu,\eta_0=\log\sigma_0,\beta_\sigma,\xi_0$ |
| Loss | 五係數經 training-only target scaling 後的 MSE |
| Optimizer | Adam |
| Initial learning rate | $10^{-3}$ |
| L2／Dropout | Adam `weight_decay=1e-4`／`p=0` |
| Batch size | 128 |
| LR scheduler | ReduceLROnPlateau，factor=0.5、patience=6、min_lr=$10^{-6}$ |
| Early stopping | validation loss，patience=20，最多 300 epochs |
| 選定 checkpoint | epoch 83；由 validation 決定，非 test |
| Training seed | 20260929，最終比較只用此單一 seed |
| Simulation seed | 20261006；各 split 使用不同子種子 |

時間與 GEV 係數的定義：

$$
t_c=\frac{\mathrm{year}-2000.5}{10},\qquad
\mu(t)=\mu_0+\beta_\mu t_c,\qquad
\log\sigma(t)=\eta_0+\beta_\sigma t_c,\qquad
\xi(t)=\xi_0.
$$

$\beta_\mu$ 的單位為 °C／十年，$\beta_\sigma$ 為 log-scale／十年。shape 可正、可負，不強制 $\xi>0$。五個輸出先逆轉 training target scaling，再逆轉每條序列的 median／IQR 標準化，還原原尺度係數。

### 模擬係數抽樣

| 係數 | 抽樣設計 |
|---|---|
| $\mu_0$ | Uniform $[-50,60]$ °C |
| $\eta_0$ | Uniform $[\log(0.2),\log(8)]$，即 $\sigma_0$ 採 log-uniform |
| $\beta_\mu$ | 有位置趨勢時抽 $\beta_\mu/\sigma_0\sim U[-0.5,0.5]$，否則為 0 |
| $\beta_\sigma$ | 有尺度趨勢時抽 $U[-0.2,0.2]$，否則為 0 |
| $\xi_0$ | Uniform $[-0.4,0.4]$，時間固定 |

四情境比例是訓練設計，不是推定全球有 25% 格點屬於每一類；scenario 標籤不作為 NN input。21D 摘要不是已證明的充分統計量。

### 最終權重與資料位置

- 權重：`results/cheng_nn_17d/mixed_sampling_20261006/input21_seed20260929/best.pt`。
- SHA-256：`534e3877a15826f5b443f71d74804edc8628821a3f0fec4b32b192c2162134b2`。
- 資料：`data/simulated/cheng_nn_mixed_100k_20261006/`。
- Test／MLE 結果：`results/cheng_nn_21d/test_mle_100k_20261007/`。

`cheng_nn_17d` 是歷史父資料夾名稱，**上述 `input21_...` checkpoint 實際為 21D**。不搬動舊實驗檔，避免破壞保存的來源與資料 hash。`notebooks/cheng_NN.ipynb` 保留共用模型／訓練函式與早期預設；直接執行它的預設訓練不等於重現此 21D 設定。21D 特徵及訓練設定由 `compare_cheng_nn_inputs.py`、`compare_cheng_nn_mixed_sampling.py` 指定。

### 獨立 test：NN 與 MLE 的五係數 RMSE

| 係數 | MLE | 21D NN | NN 相對改善 |
|---|---:|---:|---:|
| $\mu_0$ | 0.48582 | 0.49639 | −2.17% |
| $\beta_\mu$ | 0.29963 | 0.30605 | −2.14% |
| $\eta_0$ | 0.13364 | 0.13493 | −0.97% |
| $\beta_\sigma$ | 0.08126 | 0.06702 | +17.52% |
| $\xi_0$ | 0.11666 | 0.10904 | +6.54% |

改善率為 $100(1-\mathrm{RMSE}_{NN}/\mathrm{RMSE}_{MLE})\%$，正值代表 NN 較好。兩者使用相同 10,000 組 test 序列，各組 50 年；NN 權重凍結，不重新訓練。

MLE 是**與模擬係數範圍相同的受限 MLE**，每組估計完整五係數，四個 data-only 初始值，不用生成真值、NN 預測或情境標籤協助配適。10,000 組全部收斂，其中 3,226 組至少一個係數在邊界附近；多起點不保證全域最優。NN 參數曲線均有效，但 2 組有觀測超出預測 support，仍保留在係數 RMSE 計算中。

這是單一 training seed、四情境各 25% 的整體結果，未做顯著性檢定，不能宣稱 NN 全面優於 MLE。$\beta_\sigma$ 的整體優勢主要來自零尺度斜率案例；非零尺度斜率兩情境中，NN RMSE 仍較 MLE 高約 16%–18%。分情境結果見 `comparison_by_scenario.csv`。此 test 已開封，不應再用它反覆調參。

## 研究流程圖

全球資料 → Cheng NN → 時間非定常性 → GP／Spatial CV → Return levels。以下五個區塊中，時間結構判斷與全球五係數 GP 仍待串接與驗證。

### 1. 全球資料：逐時溫度 → 每格年最大值

~~~mermaid
flowchart LR
    A["ERA5 全球逐時溫度<br/>0.25°；1976–2025"] --> B["檢查月份與小時完整性"]
    B --> C["逐 GRID 取年最大值<br/>每格 50 筆"]
    C --> D["LSM > 0.5 篩選陸地<br/>格網中心接上 AR6 區域"]
    D --> E["對齊候選空間變數<br/>輸出每格序列與變數"]
~~~

不先做區域平均。保留原始海陸資料，陸地分析另套遮罩；區域外島嶼另列，不自動派到最近區域。候選變數包括地形、土地覆蓋、海岸距離、降雨、風、日射與雲量。

### 2. Cheng NN：21 維摘要 → 5 個 GEV 時間係數

~~~mermaid
flowchart TD
    A["每格 50 年最大值"] --> B["全段 median / IQR 標準化"]
    B --> C["21D<br/>11 分位數＋五期 median / IQR"]
    C --> D["小模型 128 → 128 → 64<br/>Adam；L2 = 0.0001"]
    S["四情境混合 10 萬組<br/>train / val / test = 8 / 1 / 1"] -.-> D
    D --> E["逆轉換原尺度五係數<br/>μ₀、βμ、η₀、βσ、ξ₀"]
~~~

每期 10 年；不對各期分別標準化。真實資料與模擬採相同轉換，缺年或長度不同時不直接套用這個 50 年模型。

### 3. 時間非定常性：判斷時間結構

~~~mermaid
flowchart LR
    A["年最大值序列<br/>NN 係數供初步參考"] --> B["配適候選 GEV<br/>定常／位置變動／尺度變動／兩者"]
    B --> C["比較 AICc 與時間驗證<br/>檢查係數及 RL 穩定性"]
    C --> D["採用時間結構<br/>保留相應配適係數"]
~~~

非零 NN 斜率不等於統計顯著。改採定常結構時應重新配適受限模型，不能只將斜率歸零；NN 估計值也不能直接套入假定 MLE 的檢定。`grill.ipynb` 保留作獨立診斷，其中定常模型是必要對照，不是待刪除的舊 NN。

### 4. GP／Spatial CV：重建係數的空間分布

~~~mermaid
flowchart LR
    A["GEV 係數＋候選變數<br/>依區域建立 GP"] --> B["外層 Spatial CV<br/>保留 test GRID"]
    B --> C["內層 buffered Spatial CV<br/>只用 training GRID 選變數與 kernel"]
    C --> D["外層 OOF 預測<br/>RMSE、MAE、Bias 與殘差診斷"]
~~~

時間結構若依資料選擇，也限制在 training 資料內。AR6 區域不是 CV folds；距離採測地線或合適區域投影的 km，不能將 0.25° 當固定公里，也不能將臺灣 EPSG:3826 用於全球。保留的 GP 模組與 notebook 仍以歷史三參數表為主要介面，**尚未改成全球五係數正式流程**。

### 5. Return levels：指定年份的極端溫度水準

~~~mermaid
flowchart LR
    A["GP 係數＋指定年份 t"] --> B["重建當年 GEV<br/>μ、σ、ξ"]
    B --> C["檢查數值與 support"]
    C --> D["計算年度 RL50、RL100"]
    D --> E["模擬：比較已知真值<br/>真實：評估預測與不確定性"]
~~~

$RL_T(s,t)$ 是指定年份分布的 $1-1/T$ 分位數，不代表未來 $T$ 年必然發生一次。年最大值不需月尺度轉年尺度。真實資料的 GP-vs-NN 誤差只代表與 NN 估計的一致性，不是對未知真值的誤差。

## 快速開始

在專案根目錄使用 uv：

~~~powershell
uv sync --locked
uv run python --version
~~~

`pyproject.toml` 定義依賴，`uv.lock` 鎖定版本，`.python-version` 指定 Python。資料、權重、圖片、PDF 與結果表留在本機，不隨 Git 傳送；另一台電腦需另外準備資料及權重。

### 1. 下載／續傳全球資料

先查看請求計畫，確認輸出磁碟：

~~~powershell
uv run python .\src\download_global_predictors.py `
  --start-year 1976 --end-year 2025 `
  --components all `
  --output-directory "D:\論文資料\ERA5\1975-2025_global_hourly" `
  --dry-run
~~~

移除 `--dry-run` 才實際下載。`all` 包含逐時溫度、LSM／地表位勢、ESA CCI 2000 土地覆蓋、GSHHG 海岸線及每日風、雲、日射、降雨。CDS 認證需在執行電腦另外設定，不將金鑰寫入專案。重跑會驗證並跳過完整檔案。下載目錄名稱中的 `1975` 是歷史命名，分析期間仍為 1976–2025；換電腦時依實際磁碟改路徑。

### 2. ERA5 逐時資料 → 年最大值

以下為先處理已下載 1976–2004 的例子；延伸年份前須確認原始月份完整：

~~~powershell
uv run python .\src\prepare_era5_annual_maxima.py `
  --input-directory "D:\論文資料\ERA5\1975-2025_global_hourly" `
  --output-directory "D:\論文資料\ERA5\1976-2004_global_annual_maxima" `
  --start-year 1976 --end-year 2004 --workers 2
~~~

檢查每年 12 個月、全部整點時間及格網一致性後，以 UTC 年逐格計算最大值並將 Kelvin 轉為 °C。保留原始檔、逐年 NetCDF、有效時數、最大值發生時間、LSM 及陸地寬表；任何格點缺少有效小時時，該格年度值留空。未完成年度從一月重算，已驗證年度不重跑；用 `run_status.json`、`progress_YYYY.json` 查進度。`--audit-only` 只做結構／時間檢查，不代表全檔數值檢查。

1976–2004 只有 29 年，可先整理，**不可直接套用 50 年 NN**。合併後的 `era5_t2m_annual_maxima_1976_2004_land.csv` 每格一列，以 `LSM > 0.5` 篩選；原始 longitude 為 0–360°，另有 `longitude_180`。

### 3. 建立與套用 AR6 分區

~~~powershell
uv run python .\src\global_climate_regions.py prepare `
  --lsm "D:\論文資料\ERA5\1975-2025_global_hourly\static\era5_land_sea_mask_global_025.nc"

uv run python .\src\global_climate_regions.py apply `
  --input ".\data\processed\era5_annual_maxima.csv" `
  --output ".\data\processed\era5_annual_maxima_ar6.csv" `
  --land-only
~~~

CSV 名稱是範例，請換成實際資料。以經緯度格點中心分區，不插值、不覆蓋輸入；支援 0–360° 和 −180–180°。輸出位於 `data/processed/global_regions/ar6_025/`，包括 `era5_ar6_grid.nc`、`region_catalogue.csv`、`land_grid_lookup.csv.gz`、區域外島嶼清單及 manifest。AR6 v4 有 58 個唯一區域，46 個支援陸地、15 個支援海洋，CAR／MED／SEA 兩者兼用。共用邊界以最小官方 ID 分配；$10^{-9}$ 度容差不是 CV buffer。

使用 `spatial_cv_eligible` 去除極點重複位置。更換 LSM／門檻請另指定輸出資料夾。沒有 LSM 可用 `prepare --without-lsm --output-directory ...` 建立純幾何模板，但未知遮罩不能當海洋，也不能執行 `--land-only`。

### 4. 重現／續跑最終 NN 與 MLE 比較

本機已有四情境資料、訓練 manifest 和 checkpoint 時：

~~~powershell
uv run python .\src\benchmark_cheng_21d_test_mle.py --n-jobs 6
~~~

此指令不訓練 NN，不做 RL 評估。它核對來源與資料 hash，對同一 10,000 組 test 估計五係數；每 100 組原子保存 MLE 結果，中斷後以相同指令續跑。輸出 `comparison.csv`、`comparison_by_scenario.csv`、`mle_fits.csv`、support 檢查及 `protocol.json`。沒有本機資料／權重時不會自動下載或改用其他模型。

`compare_cheng_nn_mixed_sampling.py` 保留完整 17D／21D 歷史對照設計；不帶參數執行會安排多 seed 對照，**不是只跑目前選定的一個模型**。一般使用既有最終 checkpoint，不需重跑整組選型實驗。

## 專案結構

~~~text
fast_parameter_using_NN/
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── conftest.py                          # 測試暫存放系統 Temp，結束後清理
├── data/                               # 本機資料，不推送
│   ├── original_data/                  # 原始觀測資料保留
│   ├── processed/global_regions/       # AR6／LSM 對照
│   ├── simulated/cheng_nn_mixed_100k_20261006/
│   └── spatial_predictors/             # 地形／土地／海岸／大氣變數
├── models/                             # 既有含 t 權重保留，不推送
├── notebooks/
│   ├── cheng_NN.ipynb                  # 共用 time-varying NN 訓練定義
│   ├── cheng_NN_diagnostics.ipynb       # 訓練與誤差診斷
│   ├── global_region_partition.ipynb   # AR6／LSM 分區
│   ├── grill.ipynb                     # 時間結構與定常對照
│   └── 其他 GP／分位數比較 notebook    # 歷史結果，不是全球五係數入口
├── src/
│   ├── cheng_nn_simulation.py          # 時間 GEV 抽樣與共用轉換
│   ├── train_cheng_nn.py               # 共用訓練、checkpoint 與早停
│   ├── compare_cheng_nn_mixed_sampling.py # 四情境生成與17D／21D對照
│   ├── compare_cheng_nn_inputs.py      # 17D／21D／50D轉換與對照
│   ├── benchmark_cheng_21d_test_mle.py  # 最終21D：10k test／MLE
│   ├── cheng_nn_mle.py                 # 五係數受限MLE
│   ├── cheng_nn_evaluation.py          # GEV support與條件RL工具
│   ├── cheng_nn_diagnostics.py         # NN診斷
│   ├── compare_cheng_nn_*.py           # 正則化、架構、LR與資料量對照
│   ├── report_cheng_nn_*.py            # 實驗報表
│   ├── download_global_predictors.py   # 全球來源資料下載／續傳
│   ├── prepare_era5_annual_maxima.py    # 逐時ERA5轉年最大值
│   ├── global_climate_regions.py       # AR6分區與表格接合
│   ├── prepare_daily_tmax_block_maxima.py # 保留獨立TCCIP日資料處理
│   ├── *_predictors.py                 # 地形、海岸、土地與大氣變數
│   ├── spatial_*.py                    # 空間距離、CV與殘差工具
│   ├── *_gp_analysis.py                # 歷史三參數GP工具，待銜接五係數
│   ├── quantile_ratio_estimator.py     # 非NN的歷史分位數比例方法
│   └── project_paths.py                # 共用路徑
├── scripts/download_era5_global_t2m.py  # 逐時溫度下載工具
├── tests/                              # 現行NN、MLE、ERA5與空間工具測試
└── results/                            # 本機結果，不推送
    ├── cheng_nn_17d/mixed_sampling_20261006/input21_seed20260929/
    └── cheng_nn_21d/test_mle_100k_20261007/
~~~

## 分析限制與後續工作

- 完成全球 1976–2025 年最大值與候選變數的完整性／對齊檢查，再做真實格點推論。
- 21D 與四情境比例是經驗設計。關注非零時間斜率的偏差、分情境表現與 GEV support 相容性，不能只看混合後整體 RMSE。
- 時間結構判斷需校準、時間外驗證及不確定性分析。線性時間項不涵蓋所有變點或非線性趨勢。
- 將 GP／nested buffered Spatial CV 延伸到五係數；區域距離、buffer、fold 與 kernel 需另驗證，不將歷史臺灣三參數結果當成全球證據。
- AR6 分區不保證區內定常／等向性，也不自動解決洋流影響。全球土地覆蓋目前是 2000 年靜態層，仍需評估年代代表性。
- RL 與整體下游流程另行評估，不用本次 NN 係數 RMSE 代替最終 RL 的驗證。

## 參考論文

- **Rai et al. (2024).** *Fast parameter estimation of generalized extreme value distribution using neural networks.* 用途：NN 估計 GEV 的背景，不是本專案五係數時間模型的直接驗證。
- **Zhou and Wu (2009).** *Local linear quantile estimation for nonstationary time series.* **The Annals of Statistics, 37**(5B), 2696–2729. [DOI](https://doi.org/10.1214/08-AOS636)、[公開全文](https://arxiv.org/abs/0908.3576)。用途：時間分位數／IQR 曲線的方法參考；不是 21D 摘要充分性的證明。
- **Hersbach et al. (2020).** *The ERA5 global reanalysis.* **Quarterly Journal of the Royal Meteorological Society, 146**, 1999–2049. [DOI](https://doi.org/10.1002/qj.3803)。用途：ERA5 資料來源。
- **Iturbide et al. (2020).** *An update of IPCC climate reference regions for subcontinental analysis of climate model data: definition and aggregated datasets.* **Earth System Science Data, 12**, 2959–2970. [DOI](https://doi.org/10.5194/essd-12-2959-2020)、[官方邊界](https://github.com/SantanderMetGroup/ATLAS/tree/devel/reference-regions)。用途：第3節及 Figure 1(b) 的 AR6 WGI v4 分區；程式固定版本與 hash。
- **Roberts et al. (2017).** *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.* 用途：結構化資料交叉驗證。
- **Brenning (2012).** *Spatial cross-validation and bootstrap for the assessment of prediction rules in remote sensing: The R package sperrorest.* 用途：空間重抽樣。
- **Pohjankukka et al. (2017).** *Estimating the prediction performance of spatial models via spatial k-fold cross validation.* 用途：空間 CV。
- **Valavi et al. (2019).** *blockCV: An R package for generating spatially or environmentally separated folds.* 用途：區塊與自相關距離。
- **Meyer et al. (2019).** *Importance of spatial predictor variable selection in machine learning applications.* 用途：空間變數選擇。
- **Snyder (1987).** *Map Projections—A Working Manual.* 用途：投影與距離換算。
- **Tibshirani, Walther, and Hastie (2001).** *Estimating the number of clusters in a data set via the gap statistic.* 用途：歷史空間分析的群數診斷。
- **Hanel, Buishand, and Ferro (2009).** *A nonstationary index flood model for precipitation extremes in transient regional climate model simulations.* 用途：空間極值模型背景。
