# 使用神經網路快速估計時間非定常 GEV 參數

## 專案目的與目前狀態

本專案以 **ERA5 全球 1976–2025 年、每個 GRID 的 50 筆年最大值**為分析目標。Cheng NN 快速估計時間變動 GEV 的五個係數，直接銜接 GP／buffered Spatial CV，最後估計指定年份的 return levels。

**目前主分支流程**：已合併 `review/nn-direct-spatial-cv`，移除獨立的時間非定常診斷／選模階段，不逐格重跑四個 MLE 候選模型，也暫不檢查變點。NN 的線性時間表示、五個輸出、四情境模擬訓練、support 約束及基本有效性檢查均保留；不是改回不含時間的 NN。此合併不代表下游五係數空間模擬已完成，也不替換正式模型權重。

目前選定 **21D 輸入、三層小模型、10 萬組四情境混合模擬資料**。train／validation／test = **80,000／10,000／10,000**。checkpoint 由 validation 選定，已在同一批 10,000 組 test 與五係數 MLE 比較。這是模擬估計能力的評估，**不代表全球真實資料的估計準確率已驗證，或五係數 Spatial CV 已完成**。

舊版「11 分位數 → 三個固定 GEV 參數」NN、兩個 `.pth` 權重及依賴它的臺灣前處理／校準模擬入口已移除。原始資料與已完成的歷史結果保留於本機。含時間的 17D／21D／50D 對照實驗保留，用於重現選型依據；它們不等於已退役的不含時間 NN。

## 兩個研究工作區

- **`simulation/`**：NN 訓練／診斷，以及 **ERA5 臺灣 1976–2025、50 年**校準的模擬／bootstrap 驗證。臺灣驗證不再使用舊 45 年 TCCIP 作為這個 NN 的直接輸入。
- **`global/`**：全球真實資料整理、NN 推論、buffered spatial CV 與 RL。
- **`src/`**：依用途分為 `taiwan_simulation/`（臺灣資料處理與驗證工具）、`global_pipeline/`（全球資料處理與推論）、`parameters/`（候選變數下載與處理）；共用 NN、MLE、GP 與空間工具仍放在 `src/` 根目錄，不複製兩份。各工作區的 `stage.json` 記錄共用入口、輸出位置和待完成事項；`available_tools` 只表示工具存在，`planned` 表示流程尚未串接。

1976–2025 完整 50 年年最大值已整理；`src/global_pipeline/infer_global_cheng_nn.py` 提供固定 21D 模型的全球陸地批次推論、資料轉換、有效性檢查與計時。**推論工具可用不代表 bootstrap 或五係數 Spatial CV 已完成**。原始資料留在外接磁碟；已完成的 `data/`、`results/`、模型權重及來源雜湊不搬移、不覆蓋。新下游結果預設寫入 `results/simulation/` 與 `results/global/`，也可指定 D 槽新目錄。

唯讀查看設定（從專案根目錄執行；不是完整性檢查或執行整套流程）：

~~~powershell
uv run python .\src\research_workflows.py --workspace simulation
uv run python .\src\research_workflows.py --workspace global
~~~

### 臺灣 bootstrap／已知真值模擬的用途

保留原本「生成已知真值 → 固定 NN 推論 → nested buffered spatial CV → OOF RMSE」的概念，延伸為五係數、50 年。每個 replicate 重新標準化、計算 21D、執行固定 NN（含所用 decoder），再重做空間 predictors／kernel 選擇。四種生成情境仍保留，用來分組報告係數偏差、RMSE、RL 誤差及 support 超界率；**它們是模擬標籤，不是 NN 的四分類輸出，也不需要先做 MLE**。目前不計算時間結構選回率、趨勢檢定力或變點檢定。

以真實臺灣資料估計參數再生成的曲面是 **real-data-calibrated simulated truth**，不是已知的真實氣候參數。真實資料 bootstrap 可衡量 predictors／kernel 的選取穩定性，但沒有已知氣候真值。若只固定既有 predictors/kernel 重算，得到的是條件於既定空間模型的不確定性；即使重做空間選模，目前結果仍條件於固定的線性時間模型。時間與空間相依的生成方式需另校準，不能打亂年份或把同一年跨 GRID 的事件全部當獨立。

ERA5 0.25° 的臺灣格點數／密度不同於 TCCIP 0.05°，需重新檢查 fold 數、buffer 和保留訓練格點比例；不得沿用舊 1,385 格或 8.5 km 鄰接規則。臺灣驗證不保證全球所有氣候條件的有效性。

### 臺灣 Cheng NN 新入口（2026-10-13）

入口：`simulation/taiwan_validation/taiwan_cheng_workflow.ipynb`。共用程式為 `src/taiwan_simulation/prepare_taiwan_cheng.py`；使用同一份全球 ERA5 年最大值，依既有 Natural Earth `ADM0_A3=TWN` 多邊形篩選。預設只留最大多邊形（臺灣本島），格網中心必須在多邊形內或邊界且 `LSM > 0.5`。這是可重現的製圖範圍，不是法律疆界；海岸格網中心在外側會被排除。沒有重採樣、插值或沿用舊 0.05° 網格連通規則。

本次原模型 pilot 保留 **50 GRID × 50 年（1976–2025）**，完成攝氏／完整小時數檢查、全段 median/IQR 標準化、21D、五係數推論、support 診斷、EPSG:3826 公里座標與係數空間圖／直方圖。**這是臺灣真實資料試跑，不是已知真值模擬，也不是完整下游流程已完成。** 圖中的分布是 GRID 間估計值分布，不是 bootstrap 抽樣分布。

~~~powershell
# 從專案根目錄執行；output 必須是尚不存在的新目錄
uv run python .\src\taiwan_simulation\prepare_taiwan_cheng.py --model original --output .\results\simulation\taiwan_validation_era5_1976_2025\pilot_new_original
# 僅使用實驗 BOUND checkpoint，不替換正式權重
uv run python .\src\taiwan_simulation\prepare_taiwan_cheng.py --model bound --output .\results\simulation\taiwan_validation_era5_1976_2025\pilot_new_bound
~~~

每次另存 `annual_maxima.npy`、`inputs_21d.npy`、`sample_location_scale.npy`、五係數、`spatial_responses.csv`、診斷圖及含來源／模型 SHA256 的 `run_report.json`。計時分開記錄讀檔、模型載入、21D、NN／decoder、還原尺度、診斷及總時間。BOUND 會使用同條 50 年觀測；後續 replicate 必須重新計算標準化及 decoder，不能沿用真實序列的界線。

五係數空間介面為 `src/taiwan_simulation/cheng_spatial_cv.py`，重用既有 FFS／GP 核心，不複製舊三參數流程。由內層 buffered CV RMSE 選 predictors × kernel，外層只產生 OOF；**不使用 AIC**。輸入須以 `grid_id` 精確對齊完整 predictors，五個 buffer 必須明確指定，所有內外層先檢查 training retention，不足就停止而不自動放寬。`spatial_cv_config.example.json` 中的候選、fold 數及 min_train 只是模板，null buffer 必須先校準，不能當作已驗證設定。

目前只支援明確的 `temporal_policy="full_no_selection"`（保留五係數、直接做空間選模）。臺灣候選變數尚未齊備，buffer／fold 與空間相依模擬尚待校準，因此尚未跑正式五係數 Spatial CV、RL 或 bootstrap；不再以時間選模作為前置條件。真實資料 OOF 誤差命名為 `RMSE_vs_NN_reference`，不能當作未知氣候真值的 RMSE。臺灣資料／已完成的歷史結果均保留。

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

**參數定義**：保留 $\eta_0=\log\sigma_0$ 表示基準年的 log-scale，$\log$ 為自然對數；隨時間變動的量為 $\log\sigma(t)=\eta_0+\beta_\sigma t_c$。原模型直接估計上述五係數；實驗 BOUND 版的第三個內部輸出是 $\delta$，由 decoder 在序列標準化尺度上計算 $\eta_0=\log(A+e^\delta)$，再以 training target scaling 後的五係數與標籤計算 MSE。$\delta$ 是中間變數，不是第六個 GEV 係數，也不需要額外的 $\delta$ 標籤；最後第三個係數仍是 $\eta_0$，再依原流程還原資料尺度。此次只調整顯示符號與說明，不改變模型、標籤或數值；內部欄位 `eta0`、既有權重與凍結訓練來源保持不變。

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

目前主線只有四個階段；時間變動已包含在 NN 的五係數中，不另設時間選模關卡。全球五係數 GP 與完整下游流程仍待串接與驗證。

~~~mermaid
flowchart TD
    A["全球資料<br/>每格 50 年年最大值"] --> B["Cheng NN<br/>21D → 五個時間係數"]
    B --> C["GP ＋ Buffered Spatial CV<br/>內層 FFS／kernel；外層 OOF"]
    X["對齊候選空間變數<br/>建立適當公里距離"] --> C
    C --> D["指定年份 GEV<br/>重新檢查有效性；計算 RL50／RL100"]
~~~

### 1. 全球資料：逐時溫度 → 每格年最大值

~~~mermaid
flowchart LR
    A["ERA5 全球逐時溫度<br/>0.25°；1976–2025"] --> B["檢查月份與小時完整性"]
    B --> C["逐 GRID 取年最大值<br/>每格 50 筆"]
    C --> D["LSM > 0.5 篩選陸地<br/>格網中心接上 AR6 區域"]
    D --> E["對齊候選空間變數<br/>保留經緯度與 GRID ID"]
    E --> F["各區建立／驗證公里距離<br/>供 folds、buffer 與 GP"]
~~~

不先做區域平均。保留原始海陸資料，陸地分析另套遮罩；區域外島嶼另列，不自動派到最近區域。候選變數包括地形、土地覆蓋、海岸距離、降雨、風、日射與雲量。

**公里距離的處理位置**：先以原始經緯度進行年最大值整理、陸地／區域篩選及格點對齊，再於建立 spatial folds／buffer 前建立距離表示。臺灣可用 EPSG:3826 投影後除以 1,000；原始 lon/lat 不刪除、不重採樣溫度格網。海岸距離、地形鄰域等前處理若較早需要距離，當時就應使用適當 CRS／測地線，不必等到此步。

全球 buffer 可直接用 WGS84 測地線 km；平面 GP 應使用各區經誤差檢查的投影，範圍太大時需要更小的計算區域或另行驗證的球面 covariance。不可把 AR6 全區一律套臺灣 CRS、Web Mercator 或同一 UTM zone；等距方位投影也不保證任意兩點距離無失真。座標投影只換座標表示，**不會把 0.25° 原格網變成等公里的新格網**。

### 2. Cheng NN：21 維摘要 → 5 個 GEV 時間係數

~~~mermaid
flowchart TD
    A["每格 50 年的年最大值"] --> B["全段 median / IQR 標準化"]
    B --> C["21D<br/>11 分位數＋五期 median / IQR"]
    C --> D["小模型 128 → 128 → 64<br/>Adam；L2 = 10_{-4}"]
    S["四情境混合 10 萬組<br/>train / val / test = 8 / 1 / 1"] -.-> D
    D --> E["逆轉換原尺度五係數<br/>μ₀、βμ、η₀、βσ、ξ₀"]
~~~

每期 10 年；不對各期分別標準化。真實資料與模擬採相同轉換，缺年或長度不同時不直接套用這個 50 年模型。位置與 log-scale 使用線性時間項，shape 固定；所有格點保留完整五係數，不根據斜率大小自動歸零或分成四種模型。非零斜率不代表趨勢顯著，採用此表示也不表示已排除變點。

參數與觀測 support 檢查留在 NN 推論內；BOUND decoder 保留，但不因本次流程刪減而替換正式 checkpoint。MLE 僅保留作 NN 效能／精度比較，不是 NN 後逐格必跑的步驟。

### 3. GP／Spatial CV：重建係數的空間分布

~~~mermaid
flowchart LR
    A["NN 五係數＋候選變數<br/>依區域建立 GP"] --> B["外層 Spatial CV<br/>保留 test GRID 並排除周圍 buffer"]
    B --> C["內層 buffered Spatial CV<br/>只用 training GRID 選變數與 kernel"]
    C --> D["外層 OOF 預測<br/>RMSE、MAE、Bias 與殘差診斷"]
~~~

不另做時間結構選擇；GP response 為 NN 的五個係數，不是 50 筆原始溫度。AR6 區域不是 CV folds；距離採測地線或合適區域投影的 km，不能將 0.25° 當固定公里，也不能將臺灣 EPSG:3826 用於全球。`src/taiwan_simulation/cheng_spatial_cv.py` 已提供臺灣五係數介面；**全球分區版本仍待驗證，不能把臺灣 CRS 的程式直接用於全球**。

### 4. Return levels：指定年份的極端溫度水準

~~~mermaid
flowchart LR
    A["GP 係數＋指定年份 t"] --> B["重建當年 GEV<br/>μ、σ、ξ"]
    B --> C["檢查數值與 support"]
    C --> D["計算年度 RL50、RL100"]
    D --> E["模擬：比較已知真值<br/>真實：評估預測與不確定性"]
~~~

$RL_T(s,t)$ 是指定年份分布的 $1-1/T$ 分位數，不代表未來 $T$ 年必然發生一次。年最大值不需月尺度轉年尺度。真實資料的 GP-vs-NN 誤差只代表與 NN 估計的一致性，不是對未知真值的誤差。NN 的 observed-support 保證不會自動延伸到 GP 預測，GP 係數重建分布後仍需檢查。

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
uv run python .\src\parameters\download_global_predictors.py `
  --start-year 1976 --end-year 2025 `
  --components all `
  --output-directory "D:\論文資料\ERA5\1975-2025_global_hourly" `
  --dry-run
~~~

移除 `--dry-run` 才實際下載。`all` 包含逐時溫度、LSM／地表位勢、ESA CCI 2000 土地覆蓋、GSHHG 海岸線及每日風、雲、日射、降雨。CDS 認證需在執行電腦另外設定，不將金鑰寫入專案。重跑會驗證並跳過完整檔案。下載目錄名稱中的 `1975` 是歷史命名，分析期間仍為 1976–2025；換電腦時依實際磁碟改路徑。

### 2. ERA5 逐時資料 → 年最大值

以下為先處理已下載 1976–2004 的例子；延伸年份前須確認原始月份完整：

~~~powershell
uv run python .\src\global_pipeline\prepare_era5_annual_maxima.py `
  --input-directory "D:\論文資料\ERA5\1975-2025_global_hourly" `
  --output-directory "D:\論文資料\ERA5\1976-2004_global_annual_maxima" `
  --start-year 1976 --end-year 2004 --workers 2
~~~

檢查每年 12 個月、全部整點時間及格網一致性後，以 UTC 年逐格計算最大值並將 Kelvin 轉為 °C。保留原始檔、逐年 NetCDF、有效時數、最大值發生時間、LSM 及陸地寬表；任何格點缺少有效小時時，該格年度值留空。未完成年度從一月重算，已驗證年度不重跑；用 `run_status.json`、`progress_YYYY.json` 查進度。`--audit-only` 只做結構／時間檢查，不代表全檔數值檢查。

1976–2004 只有 29 年，可先整理，**不可直接套用 50 年 NN**。合併後的 `era5_t2m_annual_maxima_1976_2004_land.csv` 每格一列，以 `LSM > 0.5` 篩選；原始 longitude 為 0–360°，另有 `longitude_180`。

### 3. 建立與套用 AR6 分區

~~~powershell
uv run python .\src\global_pipeline\global_climate_regions.py prepare `
  --lsm "D:\論文資料\ERA5\1975-2025_global_hourly\static\era5_land_sea_mask_global_025.nc"

uv run python .\src\global_pipeline\global_climate_regions.py apply `
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

### 5. 全球 50 年資料套用固定 21D NN 與計時

~~~powershell
uv run python .\src\global_pipeline\infer_global_cheng_nn.py `
  --input "D:\論文資料\ERA5\1976-2004_global_annual_maxima\era5_t2m_annual_maxima_1976_2025_global.nc" `
  --output-directory "D:\論文資料\ERA5\cheng_nn_21d_1976_2025\run_01" `
  --device cuda --batch-size 2048 --threads 4
~~~

每次請用新的輸出目錄；不覆蓋舊結果。不指定輸出目錄時，自動在 `results/global/nn_inference/run_日期_時間/` 建立新 run。`--device auto` 有 CUDA 時使用 GPU，否則 CPU。

- 從合併 NetCDF 保留原 float32 溫度精度及 GRID ID，依 `LSM > 0.5` 取陸地；檢查完整 1976–2025、攝氏單位、有效小時數、缺值及正 IQR。年最大值已是 °C，不再减 273.15，也不重算最大值。
- 全段 median/IQR 標準化一次；11 分位數＋1976–1985、1986–1995、1996–2005、2006–2015、2016–2025 各 median/IQR。五期不各自標準化、不重新 fit target scaler。
- 只載入已選定且 SHA256 符合的 21D／100k／小模型，沿用權重內 training scaler，輸出原尺度五係數。沒有訓練、沒有重跑 test/MLE、沒有顯著性或 RL 判斷。
- `annual_maxima_land.npy`、`inputs_21d.npy`、`sample_location_scale.npy`、`grid_index.csv` 是依相同行順序整理好的資料；`coefficients_original.npy` 與 `coefficients_and_checks.csv` 是預測及檢查結果。`review_required.csv` 保留需檢視的格點，不截斷、不刪除、不自動重配適。
- 檢查 GEV parameter/support，以及 NN 預測係數是否超出模擬設計、特徵是否超出 training-only 21D 邊界。邊界檢查不是 OOD 假設檢定，沒有超界也不保證全球有效；真實資料沒有已知係數真值，不能計算真值 RMSE。極點 aliases 保留並標記，後續空間模型另處理。
- `timing.csv` 與 `run_report.json` 記錄讀檔／雜湊、模型載入、特徵轉換、NN、逆轉換、training 參考、診斷、存檔及總時間。NN 計時包含 batch 搬移及 training target scaler 逆轉換，CUDA 前後 synchronize；無 warm-up。`nn_pipeline_seconds` 為特徵＋推論＋樣本逆標準化，總 wall time 另含 imports/I/O/檢查；排除最後報告與狀態序列化及程序退出。這是一輪實測，不是多輪隔離效能基準。

### 臺灣 50 GRID 年度事件與每日氣象配對

獨立入口 `src/taiwan_simulation/match_annual_events.py` **不載入／修改 NN、不下載、不執行 GP**。沿用已保存的臺灣本島 50 格 `grid_index.csv`，讀取全球年度 NetCDF 的 `annual_maximum_t2m`、`maximum_time` 和有效小時數，保留 **50 × 50 = 2,500 個 GRID-year**。不能用沒有發生日期的年度溫度 CSV 代替 NetCDF。程式目前限制最多 50 格，尚未開放全球執行。

設定範本：`simulation/taiwan_validation/event_matching.example.json`。此範本對應**既有 CDS 月檔命名**，不是已核對過的 ARCO 每日檔規格。使用 ARCO 資料時，先依實際檔案設定每個變數的 `path_template`、`variable`、`day_label`、`valid_count_variable` 與來源說明；不要猜檔名或累積時間窗。

~~~powershell
# 在專案根目錄執行；D 槽必須可讀。--output 必須是尚不存在的新資料夾。
uv run python .\src\taiwan_simulation\match_annual_events.py `
  --config .\simulation\taiwan_validation\event_matching.example.json `
  --output .\results\simulation\taiwan_validation_era5_1976_2025\events_1976_2025_run01
~~~

若磁碟代號改變，可用 `--annual-netcdf`、`--daily-root` 指定實際路徑，不必修改程式。設定檔內的相對路徑以**設定檔所在資料夾**為基準；命令列的相對覆寫路徑同樣如此，建議覆寫時使用絕對路徑。

| 輸出 | 內容 |
|---|---|
| `annual_event_index.csv` | 每格每年最高溫、發生的 UTC 小時／日期、完整小時數與事件狀態 |
| `event_weather_long.csv` | 每事件 × 五變數的值、單位、來源檔、有效小時數及配對狀態 |
| `event_weather.csv` | 2,500 列事件寬表，含各變數值與狀態；另列風向量大小 |
| `missing_report.csv`、`missing_summary.csv` | 缺檔、缺日期、缺變數／值、日期／格網／單位錯誤、不完整日及未驗證涵蓋率 |
| `grid_predictors.csv` | 每格事件日平均、有效年數、缺漏年數、共同有效年數與氣象資料完整性標記 |
| `grid_registry.csv`、`effective_config.json`、`run_report.json`、`run_status.json` | 實際格點／設定、來源指紋、輸出雜湊、計時與完成狀態 |

配對與缺漏規則：

- 用原始經緯度精確對應格網中心，容許座標排列反向及 0–360°／−180–180° 表示轉換；不做最近鄰、插值或把經緯度改成公里後配對。保留原始 GRID ID。
- 天氣指**最高溫發生的 UTC 當日統計**，不是發生瞬間：`u10/v10/tcc` 為每日平均；`tp/ssrd` 為每日累積，輸出單位依序 m/s、m/s、0–1、mm、MJ/m²。若來源含 CF 時間界限，必須吻合 UTC 00:00–24:00；不接受以 W/m² 當累積日射量。
- `wind_vector_magnitude = hypot(mean(u10), mean(v10))` 是**日平均風向量的大小**，不能稱為 `mean(hypot(u10,v10))` 的日平均風速。若研究需要後者，須由逐時分量另算。
- 設定中的 `day_label="start"` 表示當日 00:00；`"end"` 表示次日 00:00 標示前一日。檔名樣板按**檔內標籤日期**展開，所以年末累積資料可路由到下一年檔案。非午夜標籤或多筆同日資料會報錯，不自行猜測。
- 有每格／每日有效小時數欄位時，必須等於 24。**CDS 的 1_hourly request 只證明請求設定，不證明每格有 24 個有限值**；範本對沒有 count 的資料保留數值並標記 `coverage_unverified`，預設不納入可用彙整。這不等於已證實缺值，也不代表必須重新下載。若檢查來源後決定接受此限制，可逐變數明確設 `allow_unverified_coverage=true`；仍保留 warning，不能宣稱涵蓋率已驗證。
- 不丟棄 GRID/year、不填 0。`*_observed_mean` 為可用年份的描述性平均；`*_event_mean` 預設需要完整 50 年才填入。`ready_for_spatial_cv` 也要求五變數有共同的完整 50 年；它只檢查**氣象配對**，不代表靜態地形等變數、buffer 或 folds 已就緒。
- 有任何缺漏／未驗證警示時仍輸出報告，但狀態為 `completed_with_issues`、CLI exit code 為 2；乾淨完成為 0。來源磁碟不存在時直接停止，不產生假全缺漏表。

這些事件日變數接後續 GP，**不加入已固定的 21D NN**。若 outer-test 格點的最高溫日期被用來產生 predictors，研究任務須明確定義為「已知該地溫度序列的空間重建」，不能宣稱該格點完全沒有溫度資訊。離線測試已涵蓋 50 格 × 50 年及缺漏防呆；實際 CDS＋ARCO 混合來源改用下方入口，審核完成後才考慮全球擴展。

### 臺灣 50 GRID：混合來源與候選變數前處理

`src/taiwan_simulation/prepare_predictors.py` 直接讀 NetCDF，以 **GRID ID／精確格網中心＋UTC 日期**配對；目前不需要 DuckDB。實際氣象變數是降雨、風分量、雲量與日射，**尚未包含濕度；雲量不等於相對濕度**。DuckDB 適合日後查詢大型已整理的表，不會取代原始格網、單位或累積時間窗檢查。此入口不載入 NN、不下載、不執行 GP／FFS；仍限制既有臺灣 50 格。

2026-10-21 已核對 D 槽檔案：1976-01 至 1977-06 使用 CDS 月檔中的每日統計（547 日），後續 ARCO 有 17,716 個日檔，合計 18,263 日。檔案數吻合不等於全部數值已通過；實際完成狀態及異常數請看輸出目錄的 `run_status.json`／`run_report.json`。

~~~powershell
# 專案根目錄；輸出必須是新目錄。中斷續跑時對同目錄加 --resume。
uv run python .\src\taiwan_simulation\prepare_predictors.py `
  --data-root "D:\論文資料\ERA5\global_predictors_1976_2025" `
  --annual-netcdf "D:\論文資料\ERA5\1976-2004_global_annual_maxima\era5_t2m_annual_maxima_1976_2025_global.nc" `
  --output .\results\simulation\taiwan_validation_era5_1976_2025\predictors_20261021_01 `
  --workers 4
~~~

| 輸出 | 定義 |
|---|---|
| `annual_event_index.csv`、`event_weather*.csv` | 2,500 個年度最高溫事件及 UTC 當日氣象；不是發生瞬間的逐時值 |
| `daily_taiwan/taiwan_daily_YYYY.csv.gz` | 每年每日 50 格數值、來源涵蓋率與品質旗標 |
| `annual_weather.csv` | 每格每年降雨總量 mm；日射的年平均每日量 MJ/m²；u/v/雲量的年平均，以及日平均風向量大小的年平均 |
| `annual_weather_summary.csv` | 各年度量的 50 年平均及每十年 OLS 趨勢；保留有效年數，不用部分年份冒充完整趨勢 |
| `event_grid_predictors.csv` | 另外保存最高溫事件日條件下的彙整，不混稱全年氣候平均 |
| `static_predictors.csv`、`static_report.json` | 高程、粗解析度地形、2000 年土地覆蓋面積比例、海岸距離及來源 |
| `grid_predictors.csv` | 全年氣象摘要＋靜態變數，按 `grid_id` 一對一合併，另列完整性旗標 |
| `daily_missing_report.csv`、`event_missing*.csv` | 缺檔、缺值、無效值等逐筆紀錄；保留負降雨原值，不自動補零或刪格 |
| `cache_months/`、`protocol.json` | 可重用的臺灣月快取及來源指紋／程式雜湊；來源改變會重抽，設定不符不盲目續跑 |
| `candidate_groups_comparison.json` | 「平均」與「平均＋時間趨勢」兩套待比較候選群組，不是已完成選模 |

- ARCO 需符合已檢查的完整來源契約：五變數各 24 小時、逐時全球有限值檢查、每日完整稽核及檔案大小一致。降雨／日射採 UTC 01:00 至次日 00:00 的小時累積，代表該日 00:00–24:00。
- CDS 接受已核對的 UTC 每日統計及有限日值，但另外標記**缺少每格有效小時數證據**；不宣稱重新驗證了每小時涵蓋率。這與上方通用入口的嚴格預設不同，政策記入 `protocol.json`。
- 負降雨、負日射及超出 0–1 的雲量會列為無效值。某變數某年有無效／缺漏日，該年度量不填；50 年摘要也不悄悄使用較短期間。極小負降雨可能需要另訂數值誤差處理規則，原始值與稽核必須保留。
- 地形由 ERA5 geopotential / 9.80665 得到，是 **0.25° 粗地形，不是高解析 DEM**；TPI／relief 使用 5×5 原始格。CCI 2000 土地分類採每格 90×90 像素面積加權，包含水域；它不表示 50 年土地覆蓋都不變。海岸距離採 GSHHG L1 海洋岸線及 EPSG:3826 公里，不包含湖岸。
- 先保留平均與趨勢，避免直接丟掉氣象時間資訊；是否有助於估計 `beta_mu`／`beta_sigma` 要由後續 training-fold 內的 FFS／buffered CV 判斷。`candidate_data_complete` 只是資料旗標，不表示 buffer、fold、選模或已知真值模擬已完成。

## 專案結構

主分支保留臺灣、全球與參數資料三類程式目錄，並已合併 NN 直接銜接空間 CV 的流程及參數定義。移除獨立時間選模入口不會刪除本機歷史資料／結果，既有權重與凍結訓練來源也保持不變。

從專案根目錄執行下列入口即可；不需要切換到子資料夾。`taiwan_simulation/` 包含既有臺灣處理與檢查工具，不表示 ERA5 臺灣五係數模擬流程已完成。

~~~text
fast_parameter_using_NN/
├── README.md
├── pyproject.toml / uv.lock / .python-version
├── conftest.py                          # 測試暫存放系統 Temp，結束後清理
├── simulation/                         # 模擬與方法驗證工作區
│   ├── nn_training/
│   │   ├── stage.json                 # 最終21D權重與共用訓練來源
│   │   └── cheng_NN_diagnostics.ipynb # 已移入；歷史共用診斷，非自動選定21D
│   └── taiwan_validation/             # ERA5臺灣50年；NN直接接空間選模
│       ├── stage.json                  # 模擬／bootstrap仍待實作
│       ├── taiwan_cheng_workflow.ipynb # 臺灣NN試跑與空間介面
│       └── spatial_cv_config.example.json # buffer需校準，不可直接當正式設定
├── global/                             # 全球真實資料工作區
│   ├── preparation/
│   │   ├── stage.json                 # 年最大值／候選變數／公里座標規格
│   │   └── global_region_partition.ipynb # 已移入；AR6／LSM工具
│   ├── nn_inference/stage.json        # 全球21D推論、有效性檢查與計時入口
│   ├── spatial_cv/stage.json          # 五係數buffered CV；待串接
│   └── return_levels/stage.json       # 指定年份RL；待串接
├── data/                               # 本機資料，不推送
│   ├── original_data/                  # 原始觀測資料保留
│   ├── processed/global_regions/       # AR6／LSM 對照
│   ├── simulated/cheng_nn_mixed_100k_20261006/
│   └── spatial_predictors/             # 地形／土地／海岸／大氣變數
├── models/                             # 既有含 t 權重保留，不推送
├── notebooks/
│   ├── cheng_NN.ipynb                  # 共用訓練定義；為保留來源hash暫不搬動
│   └── 其他 GP／分位數比較 notebook    # 歷史結果，不是全球五係數入口
├── src/
│   ├── taiwan_simulation/              # 臺灣資料處理與驗證工具
│   │   ├── prepare_taiwan_cheng.py     # 臺灣50年NN、座標與空間response
│   │   ├── match_annual_events.py      # 年度事件／每日氣象配對與缺漏報告
│   │   ├── prepare_predictors.py       # CDS＋ARCO、事件日與全年氣象摘要／趨勢
│   │   ├── prepare_static_predictors.py # ERA5粗地形、CCI土地比例、海岸距離
│   │   ├── cheng_spatial_cv.py         # 五係數FFS／nested buffered CV
│   │   ├── prepare_daily_tmax_block_maxima.py # 既有TCCIP日資料轉block maxima
│   │   ├── export_spatial_selection_figures.py
│   │   ├── plot_selected_oof_parameter_maps.py
│   │   ├── plot_variograms.py
│   │   ├── k_sensitivity_experiment.py
│   │   └── return_level_sensitivity.py
│   ├── global_pipeline/                # 全球資料處理與NN推論
│   │   ├── prepare_era5_annual_maxima.py # 逐時ERA5轉年最大值
│   │   ├── global_climate_regions.py    # AR6分區與表格接合
│   │   └── infer_global_cheng_nn.py     # 真實50年 -> 21D -> 五係數與計時
│   ├── parameters/                     # 候選變數下載與處理
│   │   ├── download_global_predictors.py
│   │   ├── atmospheric_predictors.py
│   │   ├── coast_distance_predictor.py
│   │   ├── land_cover_predictors.py
│   │   ├── rainfall_predictors.py
│   │   └── terrain_predictors.py
│   ├── research_workflows.py           # 工作區設定索引與50年輸入介面；唯讀CLI
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

- 全球年最大值與 NN 輸入可獨立於候選變數處理；完成候選變數的完整性／對齊及 NN 預測診斷後，才銜接空間模型。
- 21D 與四情境比例是經驗設計。關注非零時間斜率的偏差、分情境表現與 GEV support 相容性，不能只看混合後整體 RMSE。
- 本分支暫停時間結構選模與變點檢定，待與指導老師確認後再決定是否恢復。線性時間項不涵蓋所有變點或非線性趨勢；目前結果條件於此模型假設，不能當作已證實適用每一格。
- 將 GP／nested buffered Spatial CV 延伸到五係數；區域距離、buffer、fold 與 kernel 需另驗證，不將歷史臺灣三參數結果當成全球證據。
- 優先以 ERA5 臺灣 1976–2025 校準已知真值模擬與選模穩定性；臺灣地理遮罩、完整50年資料及事件層級空間相依尚需確認。
- AR6 分區不保證區內定常／等向性，也不自動解決洋流影響。全球土地覆蓋目前是 2000 年靜態層，仍需評估年代代表性。
- RL 與整體下游流程另行評估，不用本次 NN 係數 RMSE 代替最終 RL 的驗證。

## 參考論文

- **Rai et al. (2024).** *Fast parameter estimation of generalized extreme value distribution using neural networks.*

  用途：NN 估計 GEV、分位數輸入與神經網路架構設計的參考。

- **Hamdi, Duluc, and Rebour (2018).** *Temperature Extremes: Estimation of Non-Stationary Return Levels and Associated Uncertainties.* **Atmosphere, 9**, 129. [DOI](https://doi.org/10.3390/atmos9040129)

  用途：時間變動 GEV 與 return level 估計的背景參考；本分支不執行文中的時間模型比較。

- **Kim et al. (2017).** *Appropriate model selection methods for nonstationary generalized extreme value models.* **Journal of Hydrology, 547**, 557–574. [DOI](https://doi.org/10.1016/j.jhydrol.2017.02.005)

  用途：時間非定常 GEV 選模的延伸閱讀；目前暫停此階段，不作為現行流程的選模準則。

- **Zhou and Wu (2009).** *Local linear quantile estimation for nonstationary time series.* **The Annals of Statistics, 37**(5B), 2696–2729. [DOI](https://doi.org/10.1214/08-AOS636)、[公開全文](https://arxiv.org/abs/0908.3576)

  用途：以不同時期的分位數與 IQR 描述分布變化的設計參考。

- **Hersbach et al. (2020).** *The ERA5 global reanalysis.* **Quarterly Journal of the Royal Meteorological Society, 146**, 1999–2049. [DOI](https://doi.org/10.1002/qj.3803)

  用途：ERA5 全球再分析資料的來源與資料特性參考。

- **Iturbide et al. (2020).** *An update of IPCC climate reference regions for subcontinental analysis of climate model data: definition and aggregated datasets.* **Earth System Science Data, 12**, 2959–2970. [DOI](https://doi.org/10.5194/essd-12-2959-2020)、[官方邊界](https://github.com/SantanderMetGroup/ATLAS/tree/devel/reference-regions)

  用途：全球資料依 IPCC AR6 氣候參考區域分區的依據。

- **Roberts et al. (2017).** *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure.*

  用途：具有時間或空間相依資料的交叉驗證設計參考。

- **Brenning (2012).** *Spatial cross-validation and bootstrap for the assessment of prediction rules in remote sensing: The R package sperrorest.*

  用途：空間交叉驗證與 bootstrap 重抽樣的參考。

- **Pohjankukka et al. (2017).** *Estimating the prediction performance of spatial models via spatial k-fold cross validation.*

  用途：使用 spatial k-fold CV 評估空間模型預測表現的參考。

- **Valavi et al. (2019).** *blockCV: An R package for generating spatially or environmentally separated folds.*

  用途：空間區塊劃分與自相關距離設定的參考。

- **Meyer et al. (2019).** *Importance of spatial predictor variable selection in machine learning applications.*

  用途：透過空間交叉驗證篩選候選變數的參考。

- **Snyder (1987).** *Map Projections—A Working Manual.*

  用途：經緯度投影、公里座標與距離失真檢查的參考。

- **Tibshirani, Walther, and Hastie (2001).** *Estimating the number of clusters in a data set via the gap statistic.*

  用途：以 gap statistic 輔助判斷空間分群數的參考。

- **Hanel, Buishand, and Ferro (2009).** *A nonstationary index flood model for precipitation extremes in transient regional climate model simulations.*

  用途：空間極值模型與已知真值模擬驗證的參考。
