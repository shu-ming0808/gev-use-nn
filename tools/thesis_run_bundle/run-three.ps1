param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Experiment = ('three_staged_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Distro = 'Ubuntu',
    [ValidateScript({ $_ -ne 0 })][int]$Jobs = -2,
    [ValidateRange(0,256)][int]$CpuBudget = 0,
    [ValidateRange(1,32)][int]$WorkerThreadsMax = 4,
    [ValidateRange(1,32)][int]$RefitThreads = 4,
    [switch]$CheckOnly,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Pair = Join-Path $Root "outputs\simulations\$Experiment\mle_comparison_v2_crps"
$Replay = Join-Path $Root "outputs_wsl\simulations\$Experiment\platform_nn_replay"
$Summary = Join-Path $Root "outputs\three_methods\$Experiment"
$Plan = @(
    @{ order=1; method='Windows NN + GP'; data='one new calibrated simulation'; },
    @{ order=2; method='Windows MLE + GP'; data='same fit observations, truth and CRPS observations'; },
    @{ order=3; method='WSL NN + GP'; data='exact saved Windows arrays; no new calibration or sampling'; }
)
if ($PlanOnly) {
    @{experiment=$Experiment; sequential=$true; workflows=$Plan; summary=$Summary; starts_workflow=$false} | ConvertTo-Json -Depth 5
    exit 0
}
function Invoke-Entry([string]$Script, [object[]]$EntryArguments) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root $Script) @EntryArguments
    if ($LASTEXITCODE -ne 0) { throw "Stopped at $Script (exit $LASTEXITCODE). No later workflow started." }
}
# Verify WSL availability BEFORE starting a many-hour Windows benchmark.
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'Install WSL first: admin PowerShell -> wsl --install -d Ubuntu; reboot and create the Linux user.'
}
& wsl.exe -d $Distro --exec true
if ($LASTEXITCODE -ne 0) { throw 'Open Ubuntu and finish its first-time setup. See WSL_PowerShell instructions.' }
. (Join-Path $Root 'environment.ps1')
if (-not (Test-Path -LiteralPath $Python)) { Invoke-Entry 'setup.ps1' @() }
Invoke-Entry 'run.ps1' @('-Mode','check','-Jobs',"$Jobs")
Invoke-Entry 'setup-wsl.ps1' @('-Distro',$Distro)
$Win = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $Root 'outputs\environment_check.json') | ConvertFrom-Json
$Linux = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $Root 'outputs_wsl\environment_check.json') | ConvertFrom-Json
# Use the same effective budget even if WSL has fewer assigned vCPUs.
$Budget = [Math]::Min([int]$Win.gp_execution_policy.cpu_budget,[int]$Linux.gp_execution_policy.cpu_budget)
if ($CpuBudget -gt 0) { $Budget = [Math]::Min($Budget,$CpuBudget) }
Write-Host "Three sequential workflows; experiment=$Experiment; shared CPU budget=$Budget; jobs=$Jobs"
if ($CheckOnly) { Write-Host 'Both CPU environments passed. No full workflow started.'; exit 0 }
$LockDir = Join-Path $Root 'outputs\three_methods'
New-Item -ItemType Directory -Force -Path $LockDir | Out-Null
$Lock = [System.IO.File]::Open((Join-Path $LockDir 'run.lock'),'OpenOrCreate','ReadWrite','None')
try {
    New-Item -ItemType Directory -Force -Path $Summary | Out-Null
    $StatusPath = Join-Path $Summary 'run_status.json'
    $Common = @('-Experiment',$Experiment,'-Jobs',"$Jobs",'-CpuBudget',"$Budget",'-ParallelMode','staged',
        '-Threads','1','-WorkerThreadsMax',"$WorkerThreadsMax",'-RefitThreads',"$RefitThreads")
    @{status='running';stage='windows_nn_mle';experiment=$Experiment;cpu_budget=$Budget} |
        ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Invoke-Entry 'run.ps1' (@('-Mode','compare-mle','-Replicates','1','-MleJobs','1') + $Common)
    @{status='running';stage='wsl_nn';experiment=$Experiment;cpu_budget=$Budget} |
        ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Invoke-Entry 'run-wsl.ps1' (@('-Mode','replay-nn','-Distro',$Distro,'-Source',$Pair) + $Common)
    Push-Location -LiteralPath $Root
    try {
        & $Python -B -m portable.platform_replay --windows $Pair --linux $Replay --output $Summary --jobs $Jobs
        if ($LASTEXITCODE -ne 0) { throw 'Three-method summary verification failed.' }
    } finally { Pop-Location }
    @{status='completed';experiment=$Experiment;summary=$Summary} |
        ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    Write-Host "Completed. Timing/support and five-coefficient RMSE: $Summary"
} catch {
    if ($StatusPath) {
        @{status='failed';experiment=$Experiment;error_message=$_.Exception.Message} |
            ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    }
    throw
} finally { $Lock.Dispose() }
