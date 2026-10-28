# 論文跑：獨立 CPU 搬機包

## 現行緯度候選／16GB主機／8GiB程序預算（2026-10-29）

主入口 `run-eas-16gb-linux.sh compare --experiment latitude_v3_ram16_mem8_cpu20_v1 --replicates 1` 採用新 `expanded_latitude_v3`：在 expanded_v2 的14組／24欄上增加一組 `latitude`（`latitude_deg`，北緯度數），成為15組／25欄。緯度由 GRID 座標逐列取得，不使用留出溫度，交給內層 FFS 選擇，不強制保留；標準化沿用各訓練折的程序。沒有加入多項式／交互作用或改 NN。

順序是新真實 EAS → 全格校準 → MLE＋GP → NN＋GP。結果隔離至 `eas_terrain_latitude_v3_ram16_mem8_v1/attempt_*` 與 `simulations/latitude_v3_ram16_mem8_cpu20_v1`；不能重用下方舊 expanded_v2 結果當成新模型。既有全部7,822格、五係數、NN權重、folds、500km buffer及seed保留。尚未完整跑新模型，不宣稱RMSE／CRPS改善。

記憶體排程預算由5提高至8GiB，保留Linux可用RAM1.5GiB及動態工作數控制。GP最多2個工作、通常2×10，RAM只許單工時新排程可用1×20；refit20、MLE最多20×1仍按RAM減少、NN4。不是8GiB硬限制，沿用不足時單工警告而不刪GRID的規則；不保證不會OOM或全程CPU滿載。

`configure-wsl-16gb.ps1 -Processors 20 -Apply` 現設WSL上限10GB、swap4GB、CPU20，讓8GiB程序預算之外仍有Linux空間；16GB主機名目剩約6GB給Windows，但背景程式會爭用RAM。設定工具備份並保留其他key，不自動停WSL。必須先停止所有WSL／Docker工作，再手動shutdown重啟；未在打包時替本機或遠端修改設定。可貼指令見 `PowerShell指令.txt`。

合成benchmark預算同改8GiB，reserve仍2GiB；只在正式流程停下時執行，不自動套用結果。真實EAS第一次必須重跑新增候選的選模，可能很久；相同命令可重用已完整且協定一致的階段，未完成GP仍需整個attempt重跑。

## 歷史16GB／20核心／5GiB入口（2026-10-29前一版）

已確認 `outputs_wsl/eas_terrain_v2_ram16_v1/attempt_001` 完成真實 EAS（7,822格、25個外折係數模型、13個輸出hash通過），總耗時93,789.139秒，NN0.905秒。`completed_with_warnings`仍有GP超界1,973格，不代表科學品質問題解決。舊校準只完成1／5係數，尚無完整MLE／NN成對模擬。

前一版主命令為 `run-eas-16gb-linux.sh compare-existing --experiment terrain_v2_ram16_cpu20_v1 --replicates 1`。這是未加入緯度的歷史指令，請勿用現行入口直接執行：現行預設profile與記憶體已變更。當時只驗證並重用已完成真實EAS作校準來源，不再跑整趟真實資料。來源/GRID/權重/科學設定/地形hash仍嚴格驗證。

新預算20，GP最多2個工作；通常2×10，若RAM在建立新排程時只允許1個則1×20。refit最多20、MLE最多20個單執行緒工作且仍按RAM減少、NN4條。保持5GiB排程預算與1.5GiB reserve，不增加GP工作數、不砍GRID、不改科學模型；既有池內的native thread數固定，尾端/循序部分不保證滿載。這是解開已確認CPU8／每工作4的上限，不是已在目標CPU量測到加速。

前一版WSL設定為CPU20/RAM8GB/swap4GB，現已提升至本頁開頭10GB；省略Processors仍保留原值。必須在所有WSL/Docker工作停止後手動shutdown才生效，腳本不替你停程式。Microsoft說明見 [WSL資源設定](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)。

`status.ps1`只讀保存狀態與日誌；`-LiveWSL`額外讀Linux CPU/RAM/程序，不重新stage或啟動研究流程。`benchmark-16gb-linux.sh`比較2×4、2×10、1×20的2000格合成GP，留存時間/取樣RAM/預測差異，不自動套用配置；它不是完整EAS或校準7822格效能保證。只在正式流程沒跑時測。

## 歷史16GB入口（2026-10-27，8核心）

主指令改為 `run-eas-16gb-linux.sh compare`：`expanded_v2`的全部7,822格、五係數、候選、folds/buffer、seed和NN權重不變。GP候選最多2×4、單一refit4、MLE最多4×1、NN4，CPU預算8、初始化4條；程序群組排程預算5GiB、保留Linux可用RAM1.5GiB。校準階段可能因記憶體餘裕自然降為1×4。原先單工警告而不砍GRID的規則保留，不是硬上限或OOM保證；單一全格校準仍可能大於排程預算。

當時 `configure-wsl-16gb.ps1` 預設只顯示計畫；加 `-Apply` 才備份並改成memory=8GB、swap=4GB（現行值見本頁開頭），保留其他設定並拒絕重複section/key。它不停止WSL或啟動訓練；需要停止所有WSL/Docker工作後手動shutdown才生效。swap是磁碟緩衝，可能很慢。

入口提供`check`（驗證及8筆NN，不跑GP）、`prepare`、`eas`與`compare`；`--plan-only`完全不跑安裝或計算。`--run-label ram16_v1`讓真實結果放到`eas_terrain_v2_ram16_v1/attempt_*`，避免舊32GB的execution protocol／計時混用；同來源地形快取仍可核對重用。新模擬前綴`terrain_v2_16gb_`，先MLE＋GP再NN＋GP，預設一次。不能沿用舊32GB實驗名稱接新設定，未完成的GP需重新跑完整attempt。

GP內折、外折及校準在預測／必要診斷完成後釋放已用完的fit，避免前後兩個Cholesky矩陣不必要重疊；不改likelihood、kernel或候選選模。這是記憶體生命週期改善，不是改成近似GP，也未宣稱全EAS實測峰值／加速倍數。主指令詳見`PowerShell指令.txt`；下段32GB配置保留作歷史參考，16GB不要直接使用舊入口。

## 歷史32GB配置

使用方式見 `PowerShell指令.txt`。目前 Linux 入口為 Ultra 7 265／32 GB 配置：CPU 排程預算18、GP最多4 workers×4條、最後refit8條、MLE最多16 workers×1條、NN4條，函式庫從8條初始化。`adaptive` 在同輪候選工作完成後依RAM餘裕補工作，結果按原順序回收；folds、五係數與模擬次數仍依序執行。這是已做相容性測試的起點，不是同款CPU已實測最佳值或全程100% CPU保證。
程式RAM排程預算20 GiB、保留Linux當下可用RAM2 GiB；建議WSL上限22 GB，設定需由使用者在停止相關工作後依說明操作。排程使用實際可用容量及保守工作集估計，不是OS硬上限或OOM保證。沿用使用者關閉硬性預檢的決定：估計不足且沒有其他工作執行時，仍允許一個worker繼續並警告，不刪GRID。真實記憶體不足、子程序退出仍回報失敗。
`Workers` 只控制全球資料前處理，與GP的`Jobs`不同。每次重新計時用新實驗名稱；同名續跑嚴格保留相同執行設定，不混用舊耗時。歷史EAS結果僅作校準來源時允許`n_jobs`不同，其餘科學設定、權重、格點順序與來源檔案雜湊仍須吻合。
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

### 新地形 profile：expanded_v2

`run-eas-terrain-linux.sh prepare` 只讀同碟 `論文資料/ERA5/global_predictors_1976_2025/static/era5_surface_geopotential_global_025.nc`，補坡度、true-north northness/eastness、八鄰格高差 RMS。原20欄完整保留，新增4欄／3組候選，合計14組／24欄。原始檔、NN、凍結inputs和舊結果不修改；沒有重新下載或重新處理每日氣象。

這是 ERA5 0.25° 粗地形。坡度以當地 WGS84 東／北向公尺差計算，坡向是下坡方向，平地為(0,0)。不套臺灣固定投影到EAS；既有LCC海岸距離、TPI、relief不改。任何halo缺漏、非有限值或GRID對不齊直接報錯，不補值／刪格。

`portable/terrain.py` 將完整來源和程式hash的overlay放在 `outputs/prepared_inputs/eas_terrain_v2/<hash>`；WSL在`outputs_wsl`。新profile真實EAS用`eas_terrain_v2/attempt_*`，`run-eas-terrain-linux.sh eas`執行；`compare`會先完成或驗證重用新profile真實EAS，之後才校準並依序跑MLE＋GP、NN＋GP。舊20欄EAS不能直接當新profile的相同校準來源。首次仍需要長時間選模，新增候選不保證提速或改善準確度。

原`run-linux.sh`仍用legacy_v1。Windows可用`run.ps1 -Mode prepare-terrain -TerrainProfile expanded_v2`，WSL wrapper有相同選項。全球目前維持舊候選／月快取協定，本次僅EAS擴充。`benchmark-linux.sh`是獨立合成GP硬體配置短測，報告在`outputs_wsl/benchmarks`；不自動套用最快配置或宣稱全EAS加速。

GP likelihood 已重用同一 Cholesky 下的多右側求解，通常將三次 `cho_solve` 減為一次；接近數值消去時保留直接解 residual 的退路。kernel、likelihood、GRID 不變，舊新數值對照已測試；尚未量到整趟 EAS 的速度提升。新增候選會增加選模工作量，CPU 或 RAM 未滿本身不代表沒有平行。

21D BOUND NN：128→128→64，五係數，固定權重，不重訓。外層5、內層3、WGS84 500 km buffer、min_train=100、全部保留訓練格點；公里投影為既有 EAS LCC。FFS 相對改善門檻0.01，VIF5；RBF、Matérn0.5/1.5/2.5；沒有時間結構選模。

計時從每次新的估計器開始，包括 NN 載入／特徵轉換／decoder 或逐格 MLE、GP/FFS、OOF support/RL、真值評估、結果輸出與雜湊。校準、模擬生成、前處理另計，不摻入估計器加速比。NN 單步加速不等於整套 GP 流程加速。

## NN 與 MLE 完整流程比較

目前用 `PowerShell指令.txt` 的 Linux 入口跑一次，確認後再沿用實驗名稱改為 `--replicates 100`。兩路線使用相同校準真值、同一批新生成的50年序列、相同格點／folds／buffer／候選與kernel；各自在內層選模、外層評估。正式NN權重不變，沒有以真值或NN輸出初始化MLE。Windows舊入口`run.ps1 -Mode compare-mle`仍保留，但不混用兩平台耗時。

- MLE 預設為 `wide_domain` 有界五係數 MLE，逐格四起點配適；失敗重試後仍未通過即停止該路線，不刪格或以 NN 補上。這不是舊 10,000 組測試資料的 training-domain MLE 比較。
- 分別報告估計器與 GP OOF 的五係數 RMSE/MAE/Bias、觀測 support 超界率，以及估計器、Spatial CV、完整流程耗時。不能將對 NN 參考值的真實資料誤差當成對模擬真值的誤差。
- CRPS 使用同一組真值另外生成的獨立 50 年評分序列，兩方法共用，不參與配適或選模。計算逐年年最大溫度的 plug-in GEV CRPS，不積分係數或 GP 不確定性；保留失效計數，不以刪除無效格點的平均掩蓋失敗。
- 已完成且雜湊／設定／模型／格點一致、候選profile相同的 EAS 真實結果，可直接作為校準來源。從legacy_v1改expanded_v2必須先完成新profile的EAS。舊計時不冒充本次模擬計時。
- Linux結果在 `outputs_wsl/simulations/<本次實驗名稱>/mle_comparison_v2_crps`，Windows結果仍在`outputs`。`notebooks/nn_mle_full_workflow_comparison.ipynb`整理準確度、計時、CRPS、品質警示及五張指定年份參數圖。
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

此段是2026-10-24歷史設計：Windows `run.ps1`保留staged設定，現行WSL入口已改為本頁開頭的Ultra 7 265 adaptive配置。可明確設定CpuBudget、WorkerThreadsMax、RefitThreads，或用ParallelMode=fixed做另案對照；不要在同一實驗變更設定。NN、資料、folds、buffer、seed、選模目標不變，浮點順序仍可能造成微小差異，不能保證逐位一致。

每趟 `execution_phases.json` 記錄階段耗時與記憶體樣本；RSS加總可能重複計入共用頁面，系統使用量也包含其他程式。記憶體預檢仍僅警告，不恢復阻擋、不縮減GRID。

WSL用獨立CPU lock及Linux環境；程式與已準備輸入在Linux磁碟，原始每日檔留外接硬碟，成果另存 `outputs_wsl`。參見 `WSL_PowerShell指令.txt`。舊Windows EAS成果只作校準來源，不當Linux耗時結果；`-Mode eas -NewRun` 才啟動新的完整真實資料試跑。跨平台先驗證原fold membership，若不一致就停止，不偷偷重切。

全球前處理的重複 `n_grids` 進度欄位已修正；以精確新舊程式hash白名單辨識此純進度修正，相同原protocol/來源/GRID/快取SHA256仍須逐一通過，才允許Windows舊600月快取續用，留下runtime_compatibility.json。沒有宣稱已彙整完成。WSL不自動借用Windows月快取，不在本次更新啟動長任務。
