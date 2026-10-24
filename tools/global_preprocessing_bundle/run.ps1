param(
    [string]$DataRoot = "",
    [string]$OutputRoot = "",
    [string]$Regions = "all",
    [string]$Exclude = "EAS",
    [ValidateRange(1,4)][int]$Workers = 2
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location -LiteralPath $PSScriptRoot
$drive = [System.IO.Path]::GetPathRoot($PSScriptRoot)
if (!$DataRoot) { $DataRoot = Join-Path $drive '論文資料\ERA5\global_predictors_1976_2025' }
if (!$OutputRoot) { $OutputRoot = Join-Path $drive '論文資料\ERA5\global_preprocessed_1976_2025\run_v1' }
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    throw '找不到 uv。請先依 https://docs.astral.sh/uv/getting-started/installation/ 安裝，再重新開 PowerShell。'
}
uv sync --locked --no-dev --python 3.10
if ($LASTEXITCODE -ne 0) { throw '環境建立失敗；尚未開始前處理。' }
uv run --locked --no-dev python -B preprocess.py check --data-root $DataRoot --output-root $OutputRoot --regions $Regions --exclude $Exclude
if ($LASTEXITCODE -ne 0) { throw '來源檢查未通過，請查看 reports\preflight.json；未開始前處理。' }
uv run --locked --no-dev python -B -u preprocess.py run --data-root $DataRoot --output-root $OutputRoot --regions $Regions --exclude $Exclude --workers $Workers --resume
if ($LASTEXITCODE -ne 0) { throw '前處理停止；查看輸出目錄 run_status.json。修正後用相同指令續跑。' }
