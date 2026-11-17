param(
    [ValidateSet('check','run','status')][string]$Mode = 'check',
    [string]$DataRoot = '',
    [ValidateRange(1,4)][int]$Workers = 4,
    [ValidateRange(1,32)][int]$Threads = 4,
    [ValidateScript({ $_ -ne 0 })][int]$Jobs = -2,
    [ValidateRange(0,128)][int]$CpuBudget = 0,
    [ValidateRange(0,32)][int]$WorkerThreadsMax = 0,
    [ValidateRange(0,32)][int]$RefitThreads = 0,
    [ValidateRange(0,1024)][double]$MemoryBudgetGiB = 0,
    [ValidateRange(0,1024)][double]$MemoryReserveGiB = 2,
    [switch]$SkipPreprocess
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Entry = Join-Path $Root 'run-global-auto.py'
if (-not (Test-Path -LiteralPath $Entry)) { throw "Missing auto-global entry: $Entry" }
if ($Mode -eq 'status') {
    $Status = Join-Path $Root 'outputs\global\run_status.json'
    $Regions = Join-Path $Root 'outputs\global\region_status.csv'
    if (Test-Path -LiteralPath $Status) { Get-Content -LiteralPath $Status -Raw }
    else { Write-Host 'Global run not started.' }
    Write-Host "Region table: $Regions"
    exit 0
}
. (Join-Path $Root 'environment.ps1')
if (-not (Test-Path -LiteralPath $Python)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'setup.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Windows CPU environment setup failed.' }
    . (Join-Path $Root 'environment.ps1')
}
if (-not $DataRoot) { $DataRoot = Join-Path (Split-Path $Root -Parent) '論文資料\ERA5\global_predictors_1976_2025' }
$Arguments = @('-B','-u',$Entry,'--mode',$Mode,'--data-root',$DataRoot,
    '--workers',"$Workers",'--threads',"$Threads",'--jobs',"$Jobs",
    '--memory-reserve-gib',"$MemoryReserveGiB")
if ($CpuBudget -gt 0) { $Arguments += @('--cpu-budget',"$CpuBudget") }
if ($WorkerThreadsMax -gt 0) { $Arguments += @('--worker-threads-max',"$WorkerThreadsMax") }
if ($RefitThreads -gt 0) { $Arguments += @('--refit-threads',"$RefitThreads") }
if ($MemoryBudgetGiB -gt 0) { $Arguments += @('--memory-budget-gib',"$MemoryBudgetGiB") }
if ($SkipPreprocess) { $Arguments += '--skip-preprocess' }
if ($Mode -eq 'check') {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Auto-global check failed (exit $LASTEXITCODE)." }
    exit 0
}
$LogDir = Join-Path $Root 'outputs\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_global_auto.log')
Start-Transcript -LiteralPath $Log | Out-Null
try {
    Write-Host "Global real-data run; log: $Log"
    Write-Host 'EAS formal 500-km result stays separate. Other regions receive independent diagnostics.'
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Auto-global runner stopped (exit $LASTEXITCODE). See $Log" }
} finally {
    Stop-Transcript | Out-Null
}
