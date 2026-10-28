param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Experiment = ('u265_rolling_' + (Get-Date -Format 'yyyyMMdd_HHmmssfff')),
    [ValidateSet(1,100)][int]$Replicates = 1,
    [string]$Distro = 'Ubuntu-24.04',
    [ValidateRange(1,1024)][double]$MemoryBudgetGiB = 20,
    [ValidateRange(0,128)][double]$MemoryReserveGiB = 2,
    [ValidateRange(0,256)][int]$CpuBudget = 18,
    [switch]$CheckOnly,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$Summary = Join-Path $PSScriptRoot "outputs_wsl\simulations\$Experiment\mle_comparison_v2_crps"
if ($PlanOnly) {
    @{ platform='Linux/WSL CPU'; order=@('MLE + GP','NN + GP'); sequential=$true;
       replicates=$Replicates; memory_budget_gib=$MemoryBudgetGiB; memory_reserve_gib=$MemoryReserveGiB;
       jobs=4; mle_jobs=16; cpu_budget=$CpuBudget; threads=8; worker_threads_max=4; refit_threads=8;
       distro=$Distro; experiment=$Experiment; parallel_mode='adaptive'; summary=$Summary; starts_workflow=$false } |
        ConvertTo-Json -Depth 4
    exit 0
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'First install Ubuntu: wsl --install -d Ubuntu-24.04; reboot and create the local Linux account.'
}
function Invoke-Entry([string]$Script, [object[]]$EntryArguments) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot $Script) @EntryArguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped at $Script. No later method started." }
}
# Installs/validates only the Linux CPU runtime; never shuts down WSL or edits .wslconfig.
Invoke-Entry 'setup-wsl.ps1' @('-Distro',$Distro,'-SkipCheck')
$Common = @('-Distro',$Distro,'-ParallelMode','adaptive','-Jobs','4','-MleJobs','16',
    '-MemoryBudgetGiB',"$MemoryBudgetGiB",'-MemoryReserveGiB',"$MemoryReserveGiB",'-CpuBudget',"$CpuBudget",
    '-WorkerThreadsMax','4','-RefitThreads','8','-Threads','8','-MethodOrder','mle-first','-Experiment',$Experiment)
Invoke-Entry 'run-wsl.ps1' (@('-Mode','check') + $Common)
$Check = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'outputs_wsl\environment_check.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host ('Linux visible RAM: {0:N1} GiB; currently available: {1:N1} GiB' -f
    ($Check.memory_snapshot.visible_total_bytes / 1GB), ($Check.memory_snapshot.available_bytes / 1GB))
if ($Check.memory_snapshot.visible_total_bytes -lt 22GB) {
    Write-Warning 'WSL currently exposes less than 22 GiB. Adaptive mode will use fewer workers. See the WSL RAM instructions beside the PowerShell instructions if the host has 32 GB RAM.'
}
if ($CheckOnly) { Write-Host 'Linux CPU and data checks passed. No full workflow started.'; exit 0 }
Write-Host 'Calibrate/generate once, then MLE + GP FIRST, followed by NN + GP, sequentially on the same simulated data.'
Invoke-Entry 'run-wsl.ps1' (@('-Mode','compare-mle','-Replicates',"$Replicates") + $Common)
Write-Host "Completed Linux MLE then NN. Results: $Summary"
