# 使用神經網路快速估計時間非定常 GEV 參數

## 專案目的與目前狀態

本專案以 **ERA5 全球 1976–2025 年、每個 GRID 的 50 筆年最大值**為分析目標。Cheng NN 快速估計時間變動 GEV 的五個係數，直接銜接 GP／buffered Spatial CV，最後估計指定年份的 return levels。

**本分支的暫定流程（待與指導老師確認）**：移除獨立的時間非定常診斷／選模階段，不逐格重跑四個 MLE 候選模型，也暫不檢查變點。NN 的線性時間表示、五個輸出、四情境模擬訓練、support 約束及基本有效性檢查均保留；不是改回不含時間的 NN。此變更留在審閱分支，尚未合併主分支。

目前選定 **21D 輸入、三層小模型、10 萬組四情境混合模擬資料**。train／validation／test = **80,000／10,000／10,000**。checkpoint 由 validation 選定，已在同一批 10,000 組 test 與五係數 MLE 比較。這是模擬估計能力的評估，**不代表全球真實資料的估計準確率已驗證，或五係數 Spatial CV 已完成**。

舊版「11 分位數 → 三個固定 GEV 參數」NN、兩個 `.pth` 權重及依賴它的臺灣前處理／校準模擬入口已移除。原始資料與已完成的歷史結果保留於本機。含時間的 17D／21D／50D 對照實驗保留，用於重現選型依據；它們不等於已退役的不含時間 NN。

## 兩個研究工作區

- **`simulation/`**：NN 訓練／診斷，以及 **ERA5 臺灣本島 1976–2025、50 年**校準的模擬／bootstrap 驗證。臺灣驗證不再使用舊 45 年 TCCIP 作為這個 NN 的直接輸入。
- **`global/`**：全球真實資料整理、NN 推論、buffered spatial CV 與 RL。
- **`src/`**：程式按臺灣流程、全球流程、參數資料三類分資料夾；兩邊共用的 NN／GP 核心仍只維護一份。各工作區的 `stage.json` 記錄共用入口、輸出位置和待完成事項；`available_tools` 只表示工具存在，`planned` 表示流程尚未串接。

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

目前只支援明確的 `temporal_policy="full_no_selection"`（保留五係數、直接做空間選模）。臺灣 50 GRID 候選資料已完成獨立清理與對齊，buffer／fold 設定仍須診斷及使用者確認，空間相依模擬尚待校準；尚未跑正式五係數 Spatial CV、RL 或 bootstrap，不再以時間選模作為前置條件。真實資料 OOF 誤差命名為 `RMSE_vs_NN_reference`，不能當作未知氣候真值的 RMSE。臺灣資料／已完成的歷史結果均保留。

### 成對校準模擬：BOUND NN＋GP vs MLE＋GP（2026-10-23）

新增 `tools/thesis_run_bundle/portable/comparison.py`，銜接既有五係數 MLE 與相同的 nested buffered Spatial CV。比較 notebook：`simulation/taiwan_validation/nn_mle_full_workflow_comparison.ipynb`。這是 EAS 搬機版的校準模擬擴充，沿用驗證工作區，不是先前 10,000 組 test 比較，也不是重新訓練 NN。

| 項目 | 比較規則 |
|---|---|
| 模擬資料 | 同一已校準五係數曲面；每次生成新 50 年序列，兩條路線使用完全相同的資料與 GRID |
| NN 路線 | 固定 21D BOUND NN → nested buffered GP → 外層 OOF |
| MLE 路線 | 每格五係數 MLE → 同一套 nested buffered GP → 外層 OOF |
| 空間選模 | 相同 folds、buffer、seed、候選變數及 kernels；各路線只在內層獨立選模，不強制選到一樣的變數 |
| 準確度 | 五係數的估計器與 GP OOF 分別對已知模擬真值計算 RMSE／MAE／Bias，報告 NN 相對 MLE 的改善百分比 |
| 計時 | 估計器、Spatial CV、估計器開始至完整輸出三段 wall time；不含離線訓練、前處理、校準、資料生成、執行前輸入檢查及最後 report JSON |
| 失敗處理 | MLE 失敗保留格點與診斷並停止該路線，不刪格、不改用 NN；另報邊界解、重試、support 與 GP optimizer 警示 |

MLE 初始化只用觀測，不讀 NN／真值／情境標籤；預設 `wide_domain`，因校準曲面不一定落在 NN 訓練參數盒內。它仍有數值邊界：標準化尺度的 `(mu0, beta_mu/sigma0, eta0, beta_sigma, xi0)` 分別限制於 `[-20,20]、[-3,3]、[-7,4]、[-1,1]、[-0.49,0.8]`。使用 4 個 data-only 起點、SLSQP、support margin `1e-8`，最多 600 iterations，失敗或發現更低但未收斂的可行 NLL 時重試至 1800。這不是無界或已保證全域最優的 MLE；與舊 training-domain MLE 表需分開解讀。

更新後的搬機包可執行（需先有完成且相容的 EAS 真實流程；既有校準通過檢查可沿用）：

~~~powershell
cd "F:\論文跑"
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Mode compare-mle -Replicates 1 -Jobs -2 -Threads 1 -MleJobs 1
# 確認一次成對比較後，再擴充到共 100 次；已完成路線驗證後跳過。
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Mode compare-mle -Replicates 100 -Jobs -2 -Threads 1 -MleJobs 1
~~~

輸出另存 `outputs/simulations/eas_calibrated_v1/mle_comparison_v2_crps/`：`parameter_comparison.csv`、`parameter_summary.csv`、`timing_comparison.csv`、`quality_comparison.csv`，以及每條路線的完整結果。舊 `mle_comparison_v1` 保留、不覆寫。Notebook 只讀檔畫圖，不會自行啟動昂貴流程。預設 MLE 為單 worker、GP 保留 `-Jobs -2`；改變設定須另開 `-Experiment`。記錄硬體／套件／執行緒，跨機器的成對結果不計加速倍數。

**新增年最大溫度分布 CRPS**：每個 replicate 在同一真值、同一 GRID 與 1976–2025 年份另外生成獨立資料 B；原資料 A 用於 NN／MLE＋GP，B 只評分且兩方法共用。原 A 隨機串流不變，B 使用 `SeedSequence(seed, spawn_key=(replicate, 0x43525053))`。這是同分布的獨立重複樣本，不是未來 50 年預測。由五係數重建各年 plug-in GEV，分別評估直接估計與 GP OOF，保留五係數 RMSE，不改 NN loss 或內層選模指標。

- `crps_comparison.csv`：每次、每階段的 NN／MLE／oracle 平均 CRPS（°C，越小越好）及 `100(1-CRPS_NN/CRPS_MLE)%`。
- `crps_summary.csv`：跨 replicate 的平均、標準差與平均分數改善率；只有一次時標準差留空。
- `crps_yearly_comparison.csv`：50 個年份分開評分，notebook 另列第 10／20／30／40／50 年並畫曲線。
- 每條路線保存獨立 B、逐格逐年 CRPS 陣列及 `crps_metrics.csv`／`crps_yearly_metrics.csv`；配對檢查資料、GRID／年份與 oracle 一致，計分另行計時並納入完整流程耗時，生成 A／B 不納入估計器時間。

公式採 [Friederichs and Thorarinsdottir (2012), *Forecast verification for extreme value distributions with an application to probabilistic peak wind prediction*](https://doi.org/10.1002/env.2176) 式 (9)–(10)，以 CDF 積分驗證，接近零 shape 使用數值積分避免相消。實作限定有限一階矩的 `xi < 1`；不合法參數、未支援 shape 或數值失敗保留計數，任一項失敗就不提供該總平均，不能篩掉失敗格點後比較。觀測落在預測 support 外仍可算 CRPS，照算且另列超界計數，不拿 B 修正模型。Oracle 真分布也有抽樣變異，CRPS 通常不為零；此分數未整合參數／GP 不確定性，不取代 RL 尾端與空間相依檢查。

Notebook 另提供**五張指定年份空間圖，每張 3×5**：一張一個參數，三列為模擬真值／MLE／BOUND NN，五欄為第 10／20／30／40／50 年（1985／1995／2005／2015／2025）。所有估計仍使用完整 50 年；繪製的是 `mu(t)、beta_mu、eta(t)=log sigma(t)、beta_sigma、xi0`，不是把 `mu0`／`eta0` 誤當每年的數值。兩個斜率與 shape 固定，因此三張圖在五欄重複並明確註記。同張圖的 15 格共用色階和座標範圍；各方法與模擬真值的年份 RMSE 另存表。預設 replicate 1 的直接估計，可改 `YEAR_PLOT_STAGE="gp_oof"` 看完整流程 OOF。

圖形程式為 `src/taiwan_simulation/plot_nn_mle_year_slices.py`。Notebook 執行後輸出 PNG（300 dpi）／SVG 至比較目錄的 `figures/replicate_001/estimation/`，保存來源雜湊和年份定義。沒有正式成對結果時不產生正式圖，不混入舊 test 或示範資料；不新增估計或訓練。

**解讀限制**：真值由 NN＋GP 校準，結果條件於此生成機制，不等於證明 NN 普遍優於 MLE；仍未加入校準曲面不確定性、同年跨格或年際殘差相依。一次比較包含兩條完整 GP 流程，NN 推論的加速不可直接當成完整流程加速。本次只以小資料驗證程式，未啟動正式 EAS／100 次比較，也未替換正式 NN。

## Cheng NN 模型參數

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

**海岸距離統一採投影後量距（2026-10-21）**：全球搬機包 `tools/global_preprocessing_bundle/` 與 EAS 都以「經緯度 → 適當區域投影（公尺）→ 點到海岸折線的最短距離 ÷ 1,000」產生 `coast_distance_km`。EAS 保留原 LCC 與量距算法；兩個入口皆已修正固定 15° 搜尋窗的限制，安全檢查不足時擴大（全部 EAS 格點擴大至 25° 後距離與原值相同）。其他區域採完整區域格點決定中心的 WGS84 AEQD，支援極區及跨日期變更線，不把東亞 CRS 套全球。海岸線使用 GSHHG 2.3.7 intermediate L1（南極另加 L5 冰前緣，不含湖岸）。搬機包的區域 CRS、搜尋窗與投影距離檢查寫入 `static/coast_regions/projection_report.json`；EAS 入口另在 `static_report.json` 記錄搜尋窗檢查。這是海岸特徵的計算設定，不替代 GP／folds／buffer 的區域診斷，也不宣稱精確測地線距離。舊球面最近取樣點版本不再作為搬機包輸出算法。

EAS 前處理可使用 `prepare_eas_predictors.py --cache-source <舊結果目錄> --output <新結果目錄>`：核對 registry、氣象擷取程式與來源指紋、快取 SHA-256 後，唯讀重用舊月資料；缺少或驗證失敗的月份從本機原始資料重新擷取至新目錄，保留拒用原因。此選項只重用氣象快取，不沿用舊靜態／年度彙整表。繼續新目錄時加 `--resume`，並保持相同 `--cache-source`；新結果仍引用舊快取，不可先刪除舊目錄。本步不下載、不啟動 NN／GP。

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

`src/taiwan_simulation/prepare_predictors.py` 直接讀 NetCDF，以 **GRID ID／精確格網中心＋UTC 日期**配對；目前不需要 DuckDB。DuckDB 適合日後查詢大型已整理的表，不會取代原始格網、單位或累積時間窗檢查。此入口不載入 NN、不下載、不執行 GP／FFS；仍限制既有臺灣 50 格。

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
| `daily_taiwan/taiwan_daily_YYYY.csv.gz` | 每年每日 50 格數值、來源涵蓋率與品質旗標；清理版另保留 `tp_raw`、`tp_quality_raw`、`tp_zero_clipped` |
| `annual_weather.csv` | 每格每年降雨總量 mm；日射的年平均每日量 MJ/m²；u/v/雲量的年平均，以及日平均風向量大小的年平均 |
| `annual_weather_summary.csv` | 各年度量的 50 年平均及每十年 OLS 趨勢；保留有效年數，不用部分年份冒充完整趨勢 |
| `event_grid_predictors.csv` | 另外保存最高溫事件日條件下的彙整，不混稱全年氣候平均 |
| `static_predictors.csv`、`static_report.json` | 高程、粗解析度地形、2000 年土地覆蓋面積比例、海岸距離及來源 |
| `grid_predictors.csv` | 全年氣象摘要＋靜態變數，按 `grid_id` 一對一合併，另列完整性旗標 |
| `daily_missing_report.csv`、`event_missing*.csv` | 缺檔、缺值、無效值等逐筆紀錄；保留負降雨原值，不自動補零或刪格 |
| `precipitation_corrections.csv` | 明確啟用清理規則時的逐筆原值、清理值、容差及來源；與缺漏報告分開 |
| `cache_months/`、`protocol.json` | 可重用的臺灣月快取及來源指紋／程式雜湊；來源改變會重抽，設定不符不盲目續跑 |
| `candidate_groups_comparison.json` | 「平均」與「平均＋時間趨勢」兩套待比較候選群組，不是已完成選模 |

- ARCO 需符合已檢查的完整來源契約：五變數各 24 小時、逐時全球有限值檢查、每日完整稽核及檔案大小一致。降雨／日射採 UTC 01:00 至次日 00:00 的小時累積，代表該日 00:00–24:00。
- CDS 接受已核對的 UTC 每日統計及有限日值，但另外標記**缺少每格有效小時數證據**；不宣稱重新驗證了每小時涵蓋率。這與上方通用入口的嚴格預設不同，政策記入 `protocol.json`。
- 負降雨、負日射及超出 0–1 的雲量會列為無效值。某變數某年有無效／缺漏日，該年度量不填；50 年摘要也不悄悄使用較短期間。極小負降雨可能需要另訂數值誤差處理規則，原始值與稽核必須保留。
- 地形由 ERA5 geopotential / 9.80665 得到，是 **0.25° 粗地形，不是高解析 DEM**；TPI／relief 使用 5×5 原始格。CCI 2000 土地分類採每格 90×90 像素面積加權，包含水域；它不表示 50 年土地覆蓋都不變。海岸距離採 GSHHG L1 海洋岸線及 EPSG:3826 公里，不包含湖岸。
- 先保留平均與趨勢，避免直接丟掉氣象時間資訊；是否有助於估計 `beta_mu`／`beta_sigma` 要由後續 training-fold 內的 FFS／buffered CV 判斷。`candidate_data_complete` 只是資料旗標，不表示 buffer、fold、選模或已知真值模擬已完成。

#### 負降雨資料品質附註

- **官方說明**：ECMWF 的 Kevin Marsh（2021-12-17）指出，ERA5 的 GRIB → NetCDF 轉換可能讓極小正降雨變成極小負值；對確認屬於此轉換問題的數值，可視為零。這不是所有負值都可直接補零的通則，也不代表負降雨具有物理意義。[Negative Values in ERA5 Precipitation Data](https://forum.ecmwf.int/t/negative-values-in-era5-precipitation-data/1636)
- **本專案抽查（2026-10-21）**：1977-08-14、25°N／121.25°E 的來源 NetCDF 使用打包值 `raw = -32766`、`scale_factor = 5.545988768748903e-7`、`add_offset = 0.018171986799682658`。以 float64 計算 `raw × scale_factor + add_offset` 得到約 `3.47e-18 m`（接近零）；改用 float32 乘加則得到 `-1.862645149230957e-9 m`，與 ARCO 同格逐時值一致。這重現了**低精度解包乘加的近零相消誤差**，不是單純將正數存成 float32 就會變負。重新讀取雲端該 UTC 日的 24 個小時累積並加總後，整張全球日圖與 D 槽存檔完全一致，排除了此案例由本機每日加總、存檔或臺灣格點抽取產生負號。
- **證據與處理邊界**：上述是抽查案例的數值重現，並非所有負值均已逐筆追查，也未核驗 ARCO 歷史建置程式的每一步。原始 NetCDF／月快取不修改；預設仍嚴格標記負值無效。2026-10-21 經確認後，獨立清理版採 `small_negative_to_zero_v1`：僅 `quality=5`、有限且 `-0.0001 <= tp < 0 mm/day` 的值歸零，另保留原值與修正旗標並重算年度／事件摘要。這個容差是本專案的明確處理規則，不是 ECMWF 通用門檻；較大的負值與真正缺值仍報錯，不補零。

清理版輸出為 `results/simulation/taiwan_validation_era5_1976_2025/predictors_20261021_clean_v1/`：600 個月、50 格 × 50 年已完成，6,690 筆極小負降雨被標記修正（含 31 筆事件日），最大單格單年修正量為 0.0007581 mm。50 格的全年候選變數皆完整，沒有剩餘日值錯誤；狀態 `completed_with_coverage_notes` 保留 CDS 最初 547 日缺少逐時涵蓋證據的警示，並非日資料缺漏。這是臺灣資料前處理完成，**不是全球 GP／已知真值模擬完成**。

~~~powershell
# 產生另一個新衍生目錄；只驗證並讀取既有 600 個月快取，不覆寫原始 run。
uv run python .\src\taiwan_simulation\prepare_predictors.py `
  --cache-source .\results\simulation\taiwan_validation_era5_1976_2025\predictors_20261021_01 `
  --output .\results\simulation\taiwan_validation_era5_1976_2025\predictors_clean_NEW_RUN `
  --precipitation-zero-tolerance-mm 0.0001
~~~

`--cache-source` 會核對原始 protocol、GRID、期間、來源指紋、每個 NPZ 與 metadata 雜湊；缺少或變更時停止，不偷偷重抽或改寫來源。衍生 run 另存當次程式版本與清理規則。

### 臺灣 NN → Spatial CV → RL：連續計時入口

`src/taiwan_simulation/run_timed_workflow.py` 在同一個程序重新執行固定 NN、依 GRID 合併候選變數、nested buffered Spatial CV／FFS、五係數 OOF support 診斷，以及各年 `RL50`／`RL100`。輸入採已完成 pilot 的 `annual_maxima.npy`、`years.npy`、`grid_index.csv`，逐一核對 SHA256 及原始年度檔；**不沿用舊 NN 預測值或把不同次執行時間相加**。

- 總計時從 `infer_sequences` 呼叫前開始，包含模型載入、21D 轉換、NN／decoder、GP 選模及預測、support／RL 計算、成果檔寫入與輸出雜湊；CUDA 前後同步。
- `run_report.json` 的 `timings.total_from_nn_start_until_outputs_seconds` 為連續 wall time；`timing.csv` 另存分段時間。資料下載／前處理、啟動匯入、輸入驗證及最後報告／狀態序列化不計入此總時間。
- 必須明確選擇 `--model original` 或 `--model bound`，並提供已定案的候選分組、五係數 buffers、outer／inner folds 與最少訓練格數。範例 `spatial_cv_config.example.json` 的 buffers 仍為 `null`，**不是可直接執行的正式設定**；不完整資料或未設定 buffer 會停止。
- 50 格的距離可行性檢查不是 buffer 選模結果：在目前 5 outer／3 inner、最少 20 格條件下，25 km 未額外排除相鄰格；50 km 使 15 個 inner splits 中 11 個不足 20 格。不能為了能執行就暗中縮小 buffer 或沿用舊 TCCIP 的門檻。
- BOUND 只保證 NN 對原觀測的 support，不保證 GP OOF 也合法；OOF support 只診斷，不用 held-out 觀測重新修正 GP。RL 誤差是相對 NN reference 的一致性，不是未知氣候真值誤差。
- 此入口不包含最終全資料 GP 配適、空間殘差診斷、已知真值空間模擬、bootstrap 或全球正式流程。這些待辦在執行報告明列；程式與離線測試完成不表示正式研究已執行。

~~~powershell
# 僅示意；先把 YOUR_FINAL_CONFIG.json 換成已確認的正式設定。
# prepared 只取既有資料快取；本次 NN 會以 --model 指定的權重重新推論。
uv run python .\src\taiwan_simulation\run_timed_workflow.py `
  --prepared .\results\simulation\taiwan_validation_era5_1976_2025\pilot_20261013_original `
  --predictors .\results\simulation\taiwan_validation_era5_1976_2025\predictors_20261021_clean_v1\grid_predictors.csv `
  --config .\simulation\taiwan_validation\YOUR_FINAL_CONFIG.json `
  --output .\results\simulation\taiwan_validation_era5_1976_2025\timed_NEW_RUN `
  --model bound --device cpu --threads 4
~~~

### 臺灣 50 GRID：五係數 buffer／fold 診斷

`src/taiwan_simulation/diagnose_buffer_folds.py` 將舊 TCCIP 的分折／variogram 診斷獨立移植到 ERA5 臺灣本島 50 GRID、1976–2025 年及 Cheng NN 五係數。只做開發期診斷，**不執行 GP／FFS、正式 OOF 評分、RL 或 bootstrap，不修改 NN 權重或正式設定**。

- 固定 seed `20260929`、KMeans `n_init=100`；比較 outer folds 3–10、inner folds 2–5、buffer 0／25／30／40／50／60／75／100／125／150／200 km，共 352 組；最低 training GRID 固定 20。
- 距離與正式 `cheng_spatial_cv.py` 相同：EPSG:3826 公里座標、保留距任一 test GRID **嚴格大於** buffer 的 training GRID。每次 outer buffer 後，重新在剩下的 training GRID 建 inner folds，再套相同 buffer。所有失敗 split 都保留，不自動縮 buffer 或降低最少格數。
- `summary.csv` 比較內外層保留訓練量、test 格數、失敗 splits 及實際 train–test 最近距離；`splits.csv`／`outer_assignments.csv` 保存實際 GRID membership。outer test 至少 5 格、inner test 至少 3 格只是實務診斷旗標，不是統計定理。較大 K 可能只是把留出區域縮小，不能因保留訓練量較高就自動勝出。
- 五係數各擬合 spherical／exponential／Gaussian variogram，僅用 pair-count weighted **半變異數擬合誤差**比較曲線，不是用 CV RMSE 挑 buffer。全 50 格的原始估計值及扣除 intercept＋elevation 的殘差只作 exploratory；另在每個未加 buffer 的 outer-training 子集重新配適高程線性基線並計算殘差 variogram，不使用該 outer-test 的係數。
- `variogram_summary.csv` 的 `range_supported` 是描述性篩檢：過短而小於格距／第一可用 lag、超出觀察 lag、撞最佳化界線、近純 nugget 或無 empirical sill 的 range 均標警示，不能硬套成精確 buffer。配對距離不是獨立樣本，擬合 covariance 不是正式空間信賴區間；50 格也不保證足以辨識相關距離。
- 將全資料 response 診斷用於調整設計時，這批資料屬開發資料，不能宣稱後續 OOF 是完全未看過的獨立驗證。Buffer 應對應預測距離及殘差相關的診斷，而非最低 RMSE。不同 native threading／套件版本也可能影響 tied KMeans，正式執行前須核對保存的 membership。

~~~powershell
uv run python .\src\taiwan_simulation\diagnose_buffer_folds.py `
  --prepared .\results\simulation\taiwan_validation_era5_1976_2025\pilot_20261013_bound `
  --predictors .\results\simulation\taiwan_validation_era5_1976_2025\predictors_20261021_clean_v1\grid_predictors.csv `
  --output .\results\simulation\taiwan_validation_era5_1976_2025\buffer_fold_diagnostics_NEW_RUN `
  --model bound
~~~

診斷先以指定的固定 NN 重新推論，不沿用舊三參數或將 `eta0` 誤當 `sigma0`。輸入、程式、權重與輸出雜湊均記錄；本輪診斷 wall time 不是未執行之整套 NN→GP→RL 的時間。正式流程仍須使用者確認設定才啟動。

2026-10-21 診斷已完成，輸出 `results/simulation/taiwan_validation_era5_1976_2025/buffer_fold_diagnostics_20261021_v1/`；352 組中 51 組通過最低訓練量，**不等於 51 組皆達到空間獨立**。精簡比較在 `comparison_for_review.csv`，待確認建議在 `review_proposal.json`，正式 config 仍未更動。

| Outer / inner folds | Buffer km | Outer training GRID | Inner training GRID | 內層不足 20 格 |
|---|---:|---:|---:|---:|
| 5 / 3 | 0 或 25 | 38–42 | 23–29 | 0 / 15 |
| 5 / 3 | 30 | 30–37 | 13–23 | 8 / 15 |
| 5 / 5 | 30 | 30–37 | 19–27 | 1 / 25 |
| **6 / 5** | **30** | **35–37** | **21–29** | **0 / 30** |
| 7 / 5 | 30 | 35–41 | 20–32 | 0 / 35 |
| 6 / 5 | 40 或 50 | 31–35 | 17–26 | 12 / 30 |

暫建議 **6 outer／5 inner、五係數共用 30 km、min_train=20、seed=20260929** 作局部空間重建的開發期試跑；outer test 為 7–11 格、inner validation 為 5–11 格，實際最小分隔約 37.48 km。25 km 與不加 buffer 的 membership 完全相同。6/5 比 7/5 保有較大的最小驗證樣本與訓練量餘裕；不是依 GP 的 RMSE 選出。

**不能宣稱已校準出可靠的五個相關距離**：全資料的五個高程殘差 variogram 均未通過 range 可辨識性篩檢；6 個 outer-training 子集分別只有 `mu0:1/6`、`beta_mu:0/6`、`eta0:1/6`、`beta_sigma:1/6`、`xi0:2/6` 通過描述性篩檢。原始 `mu0` 的約 67.5 km 也不等於最終 GP 殘差相關距離。30 km 是有實際剔除且保留足夠資料的幾何候選，並非空間獨立保證；需後續 GP 殘差診斷，若仍需更大隔離而格數不夠，應重新設計研究範圍／驗證目標，不能偷偷降低門檻。完整診斷耗時約 319.33 秒，非整套流程時間。

### EAS：五係數 buffer／fold 診斷與完整實資料試跑

目前開發範圍改為 IPCC AR6 **EAS、LSM > 0.5 的 7,822 個 GRID**，1976–2025 共 50 年；臺灣歷史檔案保留。`src/global_pipeline/diagnose_eas_buffer_folds.py` 使用同一個固定 BOUND 21D NN，不重訓或替換正式模型。

- 分折實際呼叫 **scikit-learn `KMeans`**：依區域 LCC 公里座標分成指定 K 群，不使用係數或氣象變數分群；不是隨機打散、也不保證等量。`n_init=100` 是找較佳群中心的 100 次初始化，**不是做 100 次 CV**。K 由研究設定及診斷比較決定，套件不自動決定 outer/inner K。
- 每次外層留一群作 test；其餘 GRID 先排除距任一 test GRID **不超過 buffer** 的部分。只用保留下來的 training GRID 重新建立內層 K 群，每次內層也套 buffer，選 predictors × kernel，再以保留的 outer training 重擬合、預測 outer test。外層 test 不參與 FFS、標準化或 kernel 調參。
- KMeans／GP／variogram 用區域 LCC 的 Euclidean km；**buffer 用 WGS84 測地線 km**，不把經緯度直接當公里，也不把地球弧長直接塞入原本平面 GP kernel。此區域投影的抽查 pair-distance 最大相對偏差約 1.67%，並不等於所有點對的誤差上界。
- 比較 outer K = 4／5／6、inner K = 3／5、buffer = 0／50／100／200／300／500／750／1000 km，共 48 組。使用原大型流程的 `min_train=100`，不是臺灣 50 GRID 試跑的 20；凡有內／外層 training 不足 100 格即不採用，不偷偷放寬。42 組通過最低訓練量，**不代表已證明空間獨立**。

2026-10-21 結果在 `results/global/eas/buffer_fold_diagnostics_20261021_v1/`：

| Outer / inner | Buffer km | Outer training GRID | Inner training GRID | 內層不足 100 格 |
|---|---:|---:|---:|---:|
| 5 / 3 | 300 | 4,976–6,220 | 2,028–4,309 | 0 / 15 |
| **5 / 3** | **500** | **4,034–6,007** | **1,357–3,780** | **0 / 15** |
| 5 / 5 | 500 | 4,034–6,007 | 1,673–3,855 | 0 / 25 |
| 6 / 3 | 500 | 4,324–6,729 | 1,574–5,181 | 0 / 18 |
| 5 / 3 | 750 | 2,749–5,393 | 775–2,991 | 0 / 15 |
| 5 / 3 | 1000 | 1,524–4,642 | 9–2,127 | 2 / 15 |

第一輪採 **5 outer／3 inner、共用 500 km buffer、seed=20260929**。5 個 outer test 各 1,291–1,761 格，比 6 folds 的 644–1,600 格均衡；3 個 inner folds 減少完整 FFS 的運算量。500 km 是區域留出距離的**開發期選擇**，不是依最低 RMSE 或五個已確定相關距離選出；300／750 km 可作後續距離敏感度分析。高程去均值殘差的部分 variogram range 不可辨識，仍須正式 GP 殘差診斷，不能宣稱 500 km 已消除全部空間相關。

使用者已選擇 **每折使用全部保留 training GRID**，不採舊的 800 格上限；`max_train=7822` 只是不得截斷本區資料的技術設定。精確 GP 加上 FFS 可能耗時很長，不能拿約 126 秒的幾何／variogram 診斷時間當作整套流程時間。

EAS 的 GP length-scale 搜尋範圍明確設為 **1–6,000 km、初始 300 km**，不沿用臺灣核心的 1–500 km／初始 50 km；四個 kernel 與所有內外層擬合使用相同設定。這是涵蓋區域尺度的開發期最佳化範圍，不是固定相關距離；仍需檢查最佳化是否收斂／撞邊界。舊臺灣呼叫未指定此選項時，保持原設定。

EAS 候選資料由 `src/global_pipeline/prepare_eas_predictors.py` 從已下載檔案獨立擷取至 `D:/論文資料/ERA5/eas_predictors_1976_2025/run_20261021_v1/`。保留原始逐日月快取及品質旗標，僅將已核准的 `−0.0001 <= tp < 0 mm/day` 異常負降雨轉為 0，其他缺漏不補值。候選為年度氣象摘要的 50 年平均＋每十年斜率、高程、海岸距離、2000 年土地覆蓋、TPI、局部 relief；這一輪未以事件當日變數替代年度摘要，也未宣稱涵蓋所有可能候選。

`src/global_pipeline/run_eas_workflow.py` 在候選資料完整且分折 membership 與診斷一致後，重用共用執行器跑 **新一次 NN → 內層 FFS／kernel → 外層 OOF → support／RL50、RL100**。從 NN 載入開始到輸出寫入結束連續計時，前處理／等待另列；每完成 outer fold × coefficient 就保存 checkpoint。這是 EAS 真實資料流程，不是已知真值模擬；OOF RMSE 是與固定 NN 估計的一致性，不能當作真實 GEV 參數誤差。未包含 bootstrap、全資料最終 GP 重擬合或其他 AR6 區域。`launch_status.json`／`run/run_status.json`／最終 `run/run_report.json` 才是執行狀態依據，程式存在不表示已跑完。

~~~powershell
uv run python .\src\global_pipeline\run_eas_workflow.py `
  --diagnostics .\results\global\eas\buffer_fold_diagnostics_20261021_v1 `
  --predictors-dir "D:\論文資料\ERA5\eas_predictors_1976_2025\run_20261021_v1" `
  --output .\results\global\eas\workflow_NEW_RUN --wait
~~~

巢狀 CV 的依據見文末 **Varma and Simon (2006)**、**Schratz et al. (2019)**：支持「內層選模、外層評估」的分工，不指定本研究必須使用 5／3 folds 或 500 km。

## 專案結構

### src 程式分類

| 資料夾 | 用途 |
|---|---|
| `src/taiwan_simulation/` | 臺灣 ERA5／Cheng NN 準備、五係數 spatial CV，以及保留的臺灣歷史分析／繪圖工具 |
| `src/global_pipeline/` | 全球年最大值整理、AR6 分區、NN 推論與 BOUND 對照 |
| `src/parameters/` | 候選空間變數的來源下載與處理：地形、土地覆蓋、海岸距離、降雨、風、雲與日射；不是 NN 權重 |
| `src/` 根目錄 | 共用 NN 訓練／評估／MLE 比較、GP、距離、研究設定與路徑工具 |

這次只整理程式，不搬動 `data/`、`results/`、權重或外接磁碟上的下載包。共用 NN 訓練來源保持原路徑與內容，保留既有 manifest 的雜湊校驗。歷史臺灣工具移入分類資料夾不代表它們已改成全球五係數流程。

舊的平面路徑已搬移，請改用下列新路徑；`--help` 只查看選項，不執行推論或下載：

~~~powershell
uv run python .\src\taiwan_simulation\prepare_taiwan_cheng.py --help
uv run python .\src\global_pipeline\infer_global_cheng_nn.py --help
uv run python .\src\parameters\download_global_predictors.py --help
~~~

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
│   ├── taiwan_simulation/
│   │   ├── prepare_taiwan_cheng.py     # 臺灣50年NN、座標與空間response
│   │   ├── match_annual_events.py      # 臺灣年度事件／每日氣象配對、缺漏與每格彙整；不跑NN
│   │   ├── prepare_predictors.py       # CDS＋ARCO、事件日與全年氣象摘要／趨勢；50格試跑
│   │   ├── prepare_static_predictors.py # ERA5粗地形、CCI土地比例、海岸距離
│   │   ├── cheng_spatial_cv.py         # 五係數FFS／nested buffered CV
│   │   ├── prepare_daily_tmax_block_maxima.py # 歷史TCCIP日資料處理
│   │   ├── export_spatial_selection_figures.py # 歷史OOF圖表
│   │   ├── plot_selected_oof_parameter_maps.py
│   │   ├── plot_variograms.py
│   │   ├── k_sensitivity_experiment.py
│   │   └── return_level_sensitivity.py
│   ├── global_pipeline/
│   │   ├── prepare_era5_annual_maxima.py # 逐時ERA5轉年最大值
│   │   ├── global_climate_regions.py   # AR6分區與表格接合
│   │   ├── infer_global_cheng_nn.py    # 全球五係數推論與計時
│   │   └── compare_global_cheng_bound.py # 實驗BOUND對照
│   ├── parameters/
│   │   ├── download_global_predictors.py # 全球來源下載／續傳
│   │   ├── terrain_predictors.py
│   │   ├── land_cover_predictors.py
│   │   ├── coast_distance_predictor.py
│   │   ├── rainfall_predictors.py
│   │   └── atmospheric_predictors.py
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

- **Varma and Simon (2006).** *Bias in error estimation when using cross-validation for model selection.* **BMC Bioinformatics, 7**, 91. [DOI](https://doi.org/10.1186/1471-2105-7-91)

  用途：巢狀交叉驗證的選模偏差依據；內層選設定、外層評估整套選模程序，避免直接把選模時最低的 CV 誤差當成最終泛化誤差。支持方法原則，不指定本研究折數或 buffer。

- **Schratz et al. (2019).** *Hyperparameter tuning and performance assessment of statistical and machine-learning algorithms using spatial data.* **Ecological Modelling, 406**, 109–120. [DOI](https://doi.org/10.1016/j.ecolmodel.2019.06.002)

  用途：巢狀空間交叉驗證的直接應用參考，將內層調參與外層空間預測評估分開。本文的 FFS／kernel 選擇沿用此分離原則；不是由該文推導出 6 outer／5 inner 或 30 km buffer。

- **Snyder (1987).** *Map Projections—A Working Manual.*

  用途：經緯度投影、公里座標與距離失真檢查的參考。

- **Tibshirani, Walther, and Hastie (2001).** *Estimating the number of clusters in a data set via the gap statistic.*

  用途：以 gap statistic 輔助判斷空間分群數的參考。

- **Hanel, Buishand, and Ferro (2009).** *A nonstationary index flood model for precipitation extremes in transient regional climate model simulations.*

  用途：空間極值模型與已知真值模擬驗證的參考。
