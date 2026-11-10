param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare','nn','mle','status')]
    [string]$Method,
    [ValidateRange(0,64)][int]$Threads = 0,
    [ValidateScript({ $_ -in @(-1,-2) -or ($_ -ge 1 -and $_ -le 64) })][int]$Jobs = -1,
    [ValidateRange(0,64)][int]$MleJobs = 0,
    [ValidateRange(0,64)][int]$CpuBudget = 0,
    [ValidateRange(0,64)][int]$WorkerThreadsMax = 0,
    [ValidateRange(0,64)][int]$RefitThreads = 0
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$DetectedCpus = [Environment]::ProcessorCount
if ($CpuBudget -eq 0) { $CpuBudget = $DetectedCpus }
if ($CpuBudget -gt $DetectedCpus) { throw "Requested $CpuBudget CPU slots, but Windows exposes only $DetectedCpus." }
if ($Threads -eq 0) { $Threads = $CpuBudget }
if ($MleJobs -eq 0) { $MleJobs = $CpuBudget }
if ($WorkerThreadsMax -eq 0) { $WorkerThreadsMax = $CpuBudget }
if ($RefitThreads -eq 0) { $RefitThreads = $CpuBudget }
$Experiment = Join-Path $Root 'outputs_wsl\simulations\eas_four_pair_rl_v1\split_comparison_v1'
if ($Method -eq 'status') {
    foreach ($name in @('nn','mle')) {
        $path = Join-Path $Experiment ("arms\$name\extension_status.json")
        if (Test-Path -LiteralPath $path) {
            Write-Host "${name}:"
            Get-Content -LiteralPath $path -Raw
        } else {
            Write-Host "${name}: extension not started"
        }
    }
    exit 0
}

. (Join-Path $Root 'environment.ps1')
if (-not (Test-Path -LiteralPath $Python)) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'setup.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Windows CPU environment setup failed.' }
    . (Join-Path $Root 'environment.ps1')
}
$LogDir = Join-Path $Root 'outputs_wsl\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_windows_extend_' + $Method + '.log')
$Arguments = @('-B','-u',(Join-Path $Root 'extend-six.py'),$Method,
    '--threads',"$Threads",'--jobs',"$Jobs",'--mle-jobs',"$MleJobs",
    '--cpu-budget',"$CpuBudget",'--worker-threads-max',"$WorkerThreadsMax",
    '--refit-threads',"$RefitThreads")

Start-Transcript -LiteralPath $Log | Out-Null
try {
    Write-Host "Windows CPU mode: $Method"
    Write-Host "Detected CPUs: $DetectedCpus; CPU slots: $CpuBudget; GP n_jobs: $Jobs; MLE jobs: $MleJobs"
    Write-Warning 'No RAM-based worker throttling. Exact GP can exhaust physical RAM and terminate the run; checkpoints remain reusable.'
    Write-Host "Log: $Log"
    Write-Host "Existing results and original four-run manifest will not be overwritten."
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Windows continuation stopped (exit $LASTEXITCODE). Check $Log and keep all attempt folders."
    }
    Write-Host "Result directory: $Experiment"
} finally {
    Stop-Transcript | Out-Null
}
