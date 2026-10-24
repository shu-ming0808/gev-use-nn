# 論文跑：獨立 CPU 搬機包

使用方式見 `PowerShell指令.txt` 與 `WSL_PowerShell指令.txt`。Windows 10/11 x64／WSL2 Ubuntu CPU。預設 Jobs=-2 作為候選工作上限，ParallelMode=staged 依工作數分配矩陣執行緒；folds、五係數及模擬次數仍依序執行。
依使用者要求，記憶體預估門檻已改為僅提示，不再因估計不足而阻擋執行。估計仍依有效工作數放大單工作保守估計並預留 15%，不是實測峰值。`check` 的 `memory_screen_enforced=false`，即使 `parallel_memory_screen.passes=false` 也不會被此預檢擋住。約 10 GB 可用 RAM 不代表能同時配適 19 個大型 GP；實際記憶體不足、子程序退出等錯誤仍照常失敗並回報，不隱藏，不默默減工作數或格點。
若要改為一個 GP、最多 20 條矩陣執行緒，明確加 `-ParallelMode fixed -Jobs 1 -Threads 20`。`Workers` 只控制全球資料前處理，與 GP 的 `Jobs` 不同。請於同一 EAS／校準模擬實驗沿用相同 Jobs；改變會被視為不同設定，不混用舊結果。
科學核心來源逐位元複製，模型 checkpoint SHA256 固定；舊絕對路徑只保留在來源報告中作追溯，執行不依賴它們。
`portable/inference.py` 將 checkpoint 的來源檢查映射到包內同名且同雜湊檔案，不略過來源檢查。

## 資料與環境

- 包內：EAS 7,822 格 × 50 年原始年最大值、完整候選變數表、診斷、全球陸地年最大值快取、AR6 格點登錄、BOUND 權重、訓練特徵範圍參考。
- 包外唯讀：同一硬碟的 `論文資料/ERA5/global_predictors_1976_2025`，含每日風、雲、雨、日射及靜態資料。不重複複製，不需 API 金鑰。
- 不含任何舊 `.venv`、CUDA 環境或帳號金鑰。安裝在另一台的 LOCALAPPDATA；由包位置推算資料根目錄，D/F 換槽不影響。
- 內附 uv；首次需連網安裝固定 Python 3.10.20 與 CPU PyTorch。其他數值套件版本和來源環境一致；不同 CPU 仍可能有浮點差異，不保證逐位元相同。
- 官方安裝依據：[uv 管理 Python](https://docs.astral.sh/uv/guides/install-python/)、[uv 的 PyTorch CPU 索引設定](https://docs.astral.sh/uv/guides/integration/pytorch/)。
- 新 Windows 若缺少 DLL，另安裝 [Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)；這個系統元件可能需要管理員。指令檔提供官方下載位置。

## 校準模擬設計（不是重跑同一份真實溫度）

1. 先完成或匯入 EAS 真實資料的 BOUND NN／nested spatial CV 結果。
2. 為產生模擬真值，對真實 NN 五係數逐一在全 EAS 做 FFS/kernel 選模，再用全部格點配適 GP。全資料校準不是外層驗證，不報作泛化誤差。
3. 以五張 GP posterior-mean 曲面作固定模擬真值，不以原始 NN 估計當未知氣候真值，也不每次重新抽 GP 曲面。模型收斂警告另外保存。
4. 每格每年：mu(t)=mu0+beta_mu*t，sigma(t)=exp(eta0+beta_sigma*t)，xi(t)=xi0；t=(year-2000.5)/10。
5. 用新的 Uniform(0,1) 亂數經 GEV 反分布生成 50 年資料；每次重新 median/IQR 標準化、算21D，再進凍結 BOUND NN。
6. 每次重做內層 FFS/kernel 選模與外層 buffered CV，分別評估 NN、GP OOF 對已知模擬真值的五係數 RMSE/MAE/Bias、條件 RL50/RL100 與超界率。
7. 總 seed=20260929；第 r 次用獨立 SeedSequence 子串流。地理 folds seed 固定，避免把換 folds 與資料抽樣變異混在一起。

**限制：**給定固定曲面後，各格各年條件獨立。沒有再現跨格同年天氣事件相依、年際殘差相依、校準曲面的估計不確定性；不是用來宣稱現實氣候參數的信賴區間覆蓋率。真實資料決定的候選氣象摘要視為固定設計，不從每次模擬溫度重建降雨等資料。

## EAS 設定與計時

21D BOUND NN：128→128→64，五係數，固定權重，不重訓。外層5、內層3、WGS84 500 km buffer、min_train=100、全部保留訓練格點；公里投影為既有 EAS LCC。FFS 相對改善門檻0.01，VIF5；RBF、Matérn0.5/1.5/2.5；沒有時間結構選模。

計時從每次新的估計器開始，包括 NN 載入／特徵轉換／decoder 或逐格 MLE、GP/FFS、OOF support/RL、真值評估、結果輸出與雜湊。校準、模擬生成、前處理另計，不摻入估計器加速比。NN 單步加速不等於整套 GP 流程加速。

## NN 與 MLE 完整流程比較

使用 `run.ps1 -Mode compare-mle -Replicates 1`，確認結果後再改為 `-Replicates 100`。兩路線使用相同校準真值、同一批新生成的 50 年序列、相同格點／folds／buffer／候選變數與 kernel；各自在內層選模，外層評估。正式 NN 權重不變，沒有以真值或 NN 輸出初始化 MLE。

- MLE 預設為 `wide_domain` 有界五係數 MLE，逐格四起點配適；失敗重試後仍未通過即停止該路線，不刪格或以 NN 補上。這不是舊 10,000 組測試資料的 training-domain MLE 比較。
- 分別報告估計器與 GP OOF 的五係數 RMSE/MAE/Bias、觀測 support 超界率，以及估計器、Spatial CV、完整流程耗時。不能將對 NN 參考值的真實資料誤差當成對模擬真值的誤差。
- CRPS 使用同一組真值另外生成的獨立 50 年評分序列，兩方法共用，不參與配適或選模。計算逐年年最大溫度的 plug-in GEV CRPS，不積分係數或 GP 不確定性；保留失效計數，不以刪除無效格點的平均掩蓋失敗。
- 已完成且雜湊／設定／模型／格點一致的舊 EAS 真實結果，可直接作為校準來源，無須為新版比較重跑 EAS。其舊計時不冒充本次模擬計時。
- 結果在 `outputs/simulations/eas_calibrated_v1/mle_comparison_v2_crps`。`notebooks/nn_mle_full_workflow_comparison.ipynb` 整理準確度、計時、CRPS、品質警示及五張指定年份參數圖。
- 首次比較另需建立全 EAS 校準曲面，再跑兩條完整 GP 流程；不是只有跑一次很快的 NN。請先做一次，不會自動開始 100 次。

校準真值來自 NN/GP，結論只適用於此校準設計，不能單憑它證明 NN 普遍優於 MLE。BOUND 的保證只屬於 NN decoder 使用的觀測序列；後續獨立預測五係數的 GP 沒有相同保證。GP 超界會如實列入品質表，不用 outer-test 溫度修補預測。

## 全球範圍與安全停止點

46 個有陸地且 spatial_cv_eligible 的 AR6 區域；已完成 EAS 表沿用，其他45區先轉換氣象/地形/海岸/土地覆蓋資料。`LSM>0.5` 不等於所有格點都有有效候選資料；缺漏會阻擋該區，不能默默刪除。

全球指令先完成資料轉換，再生成各區投影誤差、五係數 variogram 與 buffer/folds 可行性表。未確認的區域停在 `review_required`；不擅用 EAS 500 km。診斷不是獨立性證明，不按最低 RMSE 選 buffer。使用者確認設定檔後才跑各區 GP。投影未過5%門檻或 RAM 不足要另處理，不能只改 approved。

## 續跑與驗證

每月前處理快取、每係數校準、已完成的模擬次數可續跑；尚未完成的單次 GP 流程保留舊 attempt 並整次重跑。輸出與來源均有 SHA256 檢查；同一搬機包設系統檔案鎖。

校準與所有模擬依賴同一份已完成真實資料結果。`simulate -Replicates 1` 再改100，總共100次，不重做已完成第1次。不會在打包時自動啟動正式 EAS、全球或100次長任務。
# 2026-10-24：分階段核心與 WSL

`run-three.ps1 -Experiment three_staged_v1` 會依序跑 Windows NN→GP、Windows MLE→GP、WSL NN→GP 各一次。先確認兩平台 CPU 環境，以共同 CPU 預算執行。Windows 先生成一組校準模擬；WSL 驗證並使用相同觀測、真值與獨立 CRPS 評分資料，不重新抽樣或校準。三方法表格在 `outputs/three_methods/<experiment>`。`-CheckOnly` 不啟動長流程；WSL 首次安裝/重開機仍須依說明完成。一次順序計時不是顯著性證據，初始化與校準時間不含於方法計時。

預設 `run.ps1`／`run-wsl.ps1` 使用 staged GP 排程。Jobs=-2 是候選工作上限；CPU預算預設可用核心減一。工作多時優先多工作、少矩陣執行緒，工作少時每工作最多4條；最後重新配適4條。可設 CpuBudget、WorkerThreadsMax、RefitThreads，或以 ParallelMode=fixed 回到原排程。NN、MLE、資料、folds、buffer、seed、選模目標不變；浮點計算順序可有微小差異，不能保證逐位一致。

每趟 `execution_phases.json` 記錄階段耗時與記憶體樣本；RSS加總可能重複計入共用頁面，系統使用量也包含其他程式。記憶體預檢仍僅警告，不恢復阻擋、不縮減GRID。

WSL用獨立CPU lock及Linux環境；程式與已準備輸入在Linux磁碟，原始每日檔留外接硬碟，成果另存 `outputs_wsl`。參見 `WSL_PowerShell指令.txt`。舊Windows EAS成果只作校準來源，不當Linux耗時結果；`-Mode eas -NewRun` 才啟動新的完整真實資料試跑。跨平台先驗證原fold membership，若不一致就停止，不偷偷重切。

全球前處理的重複 `n_grids` 進度欄位已修正；以精確新舊程式hash白名單辨識此純進度修正，相同原protocol/來源/GRID/快取SHA256仍須逐一通過，才允許Windows舊600月快取續用，留下runtime_compatibility.json。沒有宣稱已彙整完成。WSL不自動借用Windows月快取，不在本次更新啟動長任務。
