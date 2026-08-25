# 專案進度與下次執行清單

更新日期：2026-08-21

## 2026-08-21：整合單一 simulated notebook

- 已將 calibrated simulation 的完整流程集中到 `notebooks/simulated.ipynb`，並移除先前未完成的 `notebooks/simulation.ipynb`。
- Notebook 順序固定為：生成已知真值、生成 annual maxima、Frozen NN、模擬曲面診斷、buffered spatial CV／FFS／kernel selection、OOF 參數恢復、$RL_{50}$ 與 $RL_{100}$ 恢復。
- 底層運算保留在 `src` 模組；Notebook 只負責設定、執行、載入結果與解讀，避免複製演算法。
- 已新增 `src/calibrated_simulation_spatial_cv.py`，使用 outer test／inner selection 的 nested buffered Spatial CV，避免現有真實資料 CLI 誤讀資料，也避免用同一批 folds 同時選模與報告最終誤差。
- Notebook 的輕量 smoke test 已通過；完整 nested CV 因運算昂貴未自動執行。
- `tests/test_calibrated_simulation_spatial_cv.py` 共 2 項測試通過。
- 正在診斷 `simulated.ipynb` 第一個設定 cell 啟動時間過久；該 cell 尚未執行 generation、diagnostics 或 nested CV，先檢查 kernel 啟動、套件 import 與殘留多核心程序。
- 診斷結果：各 import 單獨僅需約 0.06--0.71 秒；VS Code 同時存在 `langchain` 與 `31011` kernels。已將 notebook kernelspec 明確指定為 `Python (31011)`（kernel name `31011`）。

## 2026-08-23：補充模擬臺灣溫度與簡報內容

- 已於 `notebooks/simulated.ipynb` 加入模擬 annual-maximum temperature 的臺灣 GRID 地圖，不再只展示 GEV 參數曲面。
- 預計展示第 1、23、45 個模擬年度，三張圖使用共同攝氏溫標。
- 圖片已輸出至 `results/figures/calibrated_simulation_annual_maxima_examples.png` 與桌面 `picture` 資料夾。
- `calibrated_simulation_diagnostics` 相關測試共 3 項通過。
- 將提供精簡英文 Beamer 頁面，依序說明生成模型、實際溫度資料、Frozen NN、nested buffered Spatial CV 與驗證指標。
- 正在重新核對真實 TCCIP 主流程的時間尺度；需區分原始逐月資料、月最大值序列、年最大值序列，以及 NN/GEV/RL 實際使用的輸入，避免將 calibrated annual simulation 誤當成真實資料主分析。

## 2026-08-20：建立情境一（真實最終模型校準模擬）

- 已確認 `src/simulate_spatial_gev.py` 是舊的 23 測站／annual-monthly NN 與簡單 GP 實驗，且仍被其他程式引用，不應覆蓋。
- 已確認 `src/downstream_spatial_simulation.py` 是獨立的 downstream 已知真值模擬，但目前只使用人工矩形 GRID 與簡化 predictor 候選。
- 正在新增獨立的 calibrated parametric simulation：使用 1,385 個臺灣本島 GRID、真實 predictors，以及 `spatial_ffs_selected_models.csv` 的最終 predictor/kernel 結構生成已知 GEV 真值。
- 新模組將先驗證資料生成與真值輸出，不改寫既有兩套模擬結果。
- 已新增 `src/calibrated_parametric_simulation.py` 與對應單元測試；真實輸入檢查確認為 1,385 個 GRID。
- 初次 pilot 使用過窄的 $\xi\in[-0.2,0.2]$，造成 592/1,385 格被裁切，因此已改為與既有 quantile-ratio estimator 一致的 $[-0.5,0.5]$，避免人為點質量。
- 修正後 pilot 僅 4/1,385 格（0.29%）碰到 $\xi$ 邊界；單元測試 3 項皆通過。
- 情境一已輸出校準模型、45 年 annual maxima、已知 GEV/RL 真值、NN 估計與 NN recovery metrics。
- 目前完成的是資料生成與 frozen-NN 恢復檢查；下一步才是對每個 replicate 執行 nested buffered Spatial CV、FFS、kernel recovery 與 GP 後的 RL recovery，兩者不可混稱。
- 已依研究方向淘汰人工 $20\times20$ 矩形 GRID 的三情境 downstream 模擬；刪除其程式、notebook、專屬結果表與圖片，舊 Beamer 章節亦已停用。

## 2026-08-20：Downstream simulation 設計檢查

- 已檢查 `src/downstream_spatial_simulation.py`，本次未修改程式。
- 現行模擬包含 RBF、Matérn 與「高程＋方向性 Matérn」三個情境。
- 現行流程已涵蓋已知參數曲面、45 年 annual maxima、frozen NN、nested buffered Spatial CV、GP 與 return levels。
- 現行候選 mean structures 只有 `T0` 與單一高程 `T1`，尚未對應最新的多變數最終模型。
- 下一步建議以真實臺灣 GRID、真實 predictor matrix 與最終模型估計值建立 calibrated parametric simulation，並保留至少一個模型錯置情境，避免只驗證生成模型本身。
- 待決定後再修改模擬程式；正式評估應報告參數與 RL 誤差、模型選回率及不同 replicate 的穩定性。

## 目前進度

- 已建立 TCCIP 逐日最高溫、月最高溫發生日與年最大值流程。
- 已將座標由 WGS84 經緯度投影至 TWD97 / TM2（EPSG:3826），距離單位為 km。
- 已整理地形、土地覆蓋、海岸、降雨與 AgERA5 大氣候選變數。
- 現行選模使用 buffered Spatial CV，並在相同 folds 與 buffer 下比較 predictors 和 GP kernels。
- 已加入 $K=3,4,5,6,7$ 敏感度實驗及 $RL_{50}$、$RL_{100}$ 評估。
- 已加入 return-level parameter sensitivity 分析。

## 下次工作開始時先做

先切換到專案目錄並啟用環境：

```powershell
Set-Location "C:\Users\User.DESKTOP-4RV84M1\Desktop\論文\fast parameter estimate\fast_parameter_using_NN"
conda activate gev-nn
```

確認工作目錄狀態；既有原始資料、圖片與使用者修改不得直接刪除或回復：

```powershell
git status --short
```

執行核心測試：

```powershell
python -m pytest `
  .\tests\test_prepare_daily_tmax_block_maxima.py `
  .\tests\test_spatial_predictor_alignment.py `
  .\tests\test_return_level_sensitivity.py -q
```

檢查真實資料與大氣資料是否完整：

```powershell
python .\src\real_grid_modeling_pipeline.py --check-only
```

只有上述檢查全部通過後，才重新建立 model-ready GRID：

```powershell
python .\src\real_grid_modeling_pipeline.py --prepare-only --n-jobs -2
```

## 下一次正式執行

### 已完成：校準模擬曲面診斷

- 新增 `notebooks/simulation.ipynb`。
- 比較真實 NN 參數曲面與 calibrated simulation 真值曲面。
- 使用共同色階、標準化 variogram 與最近鄰粗糙度比值，檢查模擬資料是否過度平滑或過度粗糙。
- 本段只診斷生成資料，不重新訓練 NN，也不覆蓋既有模擬輸出。
- README 已將 Hanel et al. (2009) 列為本段空間 GEV 模擬的主要參考；未將一般背景文獻誤列為直接實作來源。
- 已輸出共同色階地圖、邊際分布、標準化 variogram 與粗糙度摘要。
- 最近鄰粗糙度比值：mu = 0.979、log_sigma = 0.593、xi = 0.871；log_sigma 的模擬真值在最短距離較平滑，需與 frozen-NN 復原曲面分開解讀。
- 新增 2 個診斷測試；連同 calibrated simulation 測試共 5 個通過。

執行完整 buffered Spatial-CV、FFS 與 GP kernel 選擇：

```powershell
python .\src\real_grid_modeling_pipeline.py --n-jobs -2
```

若需要檢查 $K$ 的敏感度，再執行：

```powershell
python .\src\k_sensitivity_experiment.py `
  --k-values 3 4 5 6 7 --n-jobs -2
```

若需要更新簡報圖片，再執行：

```powershell
python .\src\compare_four_stage_oof.py --n-jobs -2 `
  --output-directory "C:\Users\User.DESKTOP-4RV84M1\Desktop\picture"
```

## 執行後檢查

- 確認每個 fold 的 training retention 與 test 樣本數。
- 比較 OOF RMSE、MAE、Bias 與 fold-level stability。
- 檢查 OOF residual Moran's $I$ 與 residual variogram。
- 重新計算並比較 $RL_{50}$ 與 $RL_{100}$。
- 確認輸出圖與 CSV 的模型名稱、predictors、kernel 和執行設定一致。

## 修改原則

- 未經明確要求，不修改 README 的其他章節。
- 未經明確要求，不修改 `.gitignore`。
- 不刪除或覆蓋使用者既有資料、模型權重與圖片。
- 程式路徑優先使用專案相對路徑。
- 大型原始資料不加入 Git；程式、測試與必要摘要表才列入版本控制。
## 2026-08-23 時間尺度查核

- `grid_station_gev_params_with_loc.csv`：1,412 格，`n_obs` 中位數為 45。
- `model_ready_grid_parameters.csv`：1,385 格，`n_obs` 全部為 45。
- 因此目前已執行的真實資料 GP／選模流程實際使用 45 個年最大值；並非 540 個月最大值。
- `tccip_grid_preprocessing.py` 仍留有「540 個月最大值」的舊版說明，與目前 canonical pipeline 不一致，後續須統一時間尺度與 return-level 定義。
## 2026-08-23 月最大值模擬修正

- 目標：將 calibrated simulation 從每格 45 個年最大值改為 45 年乘 12 月，共 540 個月最大值。
- 同步修正年尺度 `RL50`、`RL100` 的月分布換算。
- `notebooks/simulated.ipynb` 產生的圖片將統一另存至 `C:/Users/User.DESKTOP-4RV84M1/Desktop/picture`。
- 已完成：每格生成 1980-01 至 2024-12 的 540 個月最大值。
- 已完成：Frozen NN 使用 540 筆樣本計算 11 個 quantiles。
- 已完成：`RL50`、`RL100` 改用月 GEV 轉換成年尺度 return level。
- 已完成：`simulated.ipynb` 所有新圖統一輸出至桌面 `picture`。
- 已完成：重新生成月模擬資料與非 OOF 診斷圖；9 項相關測試通過。
- 待執行：重跑完整 nested buffered Spatial CV，才能產生與月版本一致的 OOF 參數圖及 OOF return-level 圖。
## 2026-08-23 月版本 nested Spatial CV 重跑

- 使用獨立輸出資料夾 `nested_spatial_cv_monthly`，不讀取舊年度 OOF CSV。
- 執行前驗證 `block_scale=monthly`、`n_months=540`、`months_per_year=12`。
- 記錄輸入檔 SHA-256 與修改時間，確保 OOF 結果來自最新月模擬資料。
- 已完成完整 5 outer folds × 3 responses 的 nested buffered Spatial CV。
- 新輸出位於 `data/simulated/calibrated_final_model/nested_spatial_cv_monthly/`。
- metadata 驗證：`block_scale=monthly`、`n_months=540`、`months_per_year=12`、`n_grid=1385`。
- 輸入 SHA-256 與目前 `replicate_000_model_ready.csv` 完全相符。
- 已只使用新 CSV 產生 OOF parameter 與 annual RL50/RL100 圖至桌面 `picture`。
- Nested OOF GP RMSE：mu 0.651011、log_sigma 0.187659、xi 0.118331、RL50 1.511151、RL100 1.739570。
## 2026-08-23 清理舊年度模擬與敏感度規劃

- 已刪除 45 annual-maxima CSV、舊 nested OOF、早期 spatial_gev 三情境資料與對應舊圖。
- 保留 540 monthly-maxima、`nested_spatial_cv_monthly`、新 OOF 圖與所有真實資料。
- 清理驗證：舊 annual／stationary-rbf／stationary-matern／anisotropic 檔案計數為 0。
- 新月模擬的 exact one-at-a-time RL 誤差：RL50 的 mu/log_sigma/xi 貢獻 RMSE 為 0.0360/0.2028/0.3670；RL100 為 0.0360/0.2237/0.4560。
- 建議先輸出 RL 誤差貢獻表，再以標準化 RL-aware auxiliary loss 調整；尚未變更 NN loss 權重。

## 2026-08-23 舊真實資料 OOF 保留核對

- 已確認未刪除「無 predictors」與「僅高程」的真實 GRID OOF 結果。
- `results/tables/elevation_gp_oof_predictions.csv` 仍有 33,888 列，包含 T0、T1 與三個 GEV 參數。
- 桌面 `picture/four_stage_oof_predictions.csv` 仍有 16,620 列，包含 No predictors、Elevation only、Previous selection、Current selection。
- 對應的四階段 OOF 圖、metrics、residual variogram 與 elevation 圖仍存在。
- 本次僅唯讀核對，未再刪除任何檔案。

## 2026-08-24 投影設定核對

- 實際程式以 `src/spatial_coordinates.py` 為統一座標入口。
- 臺灣原始經緯度為 WGS84（EPSG:4326），轉為 TWD97 / TM2 zone 121（EPSG:3826）。
- EPSG:3826 採 Transverse Mercator；原生單位為公尺，程式再除以 1,000 產生 `x_km`、`y_km`。
- GP 計算僅減去 training-set 座標平均，不分軸除以標準差，因此距離與 kernel length scale 仍為 km。
- K-means folds、buffer、variogram、Moran's I、地形與海岸距離均使用相同的 TWD97 公里座標。

## 2026-08-25 模擬 recovery 圖版面調整

- 僅從簡報 recovery 圖移除 Frozen NN 欄位，未移除 NN 流程、NN 數據或模型權重。
- `calibrated_simulation_oof_parameter_recovery.png` 已改為 3×2：Truth 與 Nested OOF GP。
- `calibrated_simulation_oof_return_level_recovery.png` 已改為 2×2：RL50/RL100 的 Truth 與 Nested OOF GP。
- 已使用最新 540 monthly-maxima 且 SHA-256 相符的 OOF 輸出重新繪圖，並覆蓋桌面 `picture` 同名檔案。

## 2026-08-25 Nested OOF GP 的 RL 誤差來源

- 將 one-at-a-time return-level decomposition 套到最新 Nested OOF GP，而非只套 NN。
- RL50 的 mu/log_sigma/xi-only RMSE 為 0.6510/0.6845/1.1982；全部參數為 1.5112。
- RL100 的 mu/log_sigma/xi-only RMSE 為 0.6510/0.7541/1.4415；全部參數為 1.7396。
- 結果顯示最終空間流程的長 return-period 誤差主要受 xi 空間預測限制；各 contribution 因非線性與交互作用不可相加。

## 2026-08-25 模擬 replicate 與 OOF 對齊核對

- 目前 calibrated simulation 只有一個獨立 replicate：`replicate_000`，共 1,385 個 GRID。
- `replicate_000_monthly_maxima.csv` 與 `replicate_000_model_ready.csv` 是同一 replicate 的原始月最大值與整理後分析表，不是兩次模擬。
- Nested OOF metadata 的 input 明確指向 `replicate_000_model_ready.csv`，SHA-256 相符。
- one-at-a-time RL 分解在相同 station、相同 replicate 內組合 truth 與 OOF prediction，未跨不同模擬資料。

## 2026-08-25 模擬方法文獻核對

- 核對 Hanel, Buishand, and Ferro (2009) 的 Appendix C（段落 78）。
- 該文先設定具有已知參數的非平穩 GEV 模型，再生成具空間相關性的樣本、重新估計參數並與已知設定比較。
- 本專案沿用「已知 GEV 真值生成資料後檢查恢復能力」的 parametric simulation 原則。
- 臺灣 GRID、真實 predictors、NN 與 nested buffered Spatial CV 屬本研究延伸，並非該文原樣流程。
- 簡報應稱 real-data-calibrated simulated truth，不應稱為真實世界中已知的 GEV 參數。

## 2026-08-25 現行 calibrated simulation 程式核對

- 實際生成程式為 `src/calibrated_parametric_simulation.py`，而非舊的 `simulate_spatial_gev.py`。
- 使用 1,385 個臺灣本島 GRID；最終選模結構為 mu: elevation+TPI+wind/RBF、log_sigma: elevation+cloud/Matérn 1.5、xi: elevation/Matérn 0.5。
- 先以最多 800 個 GRID 校準三個 real-data GP，再由 fitted linear mean 加一個 fitted covariance 的 GP latent draw 建立新的已知真值曲面。
- latent covariance 排除 WhiteKernel nugget；三個參數各自獨立抽取 spatial effect；xi 裁切至 [-0.5, 0.5]，sigma 由 exp(log_sigma) 得到。
- 每個 GRID 依其固定真值參數獨立抽取 540 筆 monthly GEV maxima；目前未加入月份季節性，也未加入同一月份跨 GRID 的事件層級相依。
- 每組 540 筆資料以 median/IQR robust standardization 後取 11 個 empirical quantiles，送入 frozen baseline NN，再進 nested buffered Spatial CV。

## 2026-08-25 Calibrated simulation 推送前驗證

- notebook 名稱統一為 `notebooks/simulation.ipynb`，並同步 README 與 workflow 文件。
- 保留原本 `.gitignore` 規則；原始資料與未指定圖片不納入本次提交。
- 停用環境中無關的第三方 pytest 外掛後，四組模擬與 sensitivity 測試共 11 項全部通過。
