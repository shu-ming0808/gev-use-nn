param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('prepare','nn','mle','status')]
    [string]$Method,
    [ValidateRange(0,64)][int]$Threads = 0,
    [ValidateScript({ $_ -in @(-1,-2) -or ($_ -ge 1 -and $_ -le 64) })][int]$Jobs = -1,
    [ValidateScript({ $_ -in @(0,-1,-2) -or ($_ -ge 1 -and $_ -le 64) })][int]$MleJobs = 0,
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
if ($MleJobs -eq 0) { $MleJobs = -2 }
if ($WorkerThreadsMax -eq 0) { $WorkerThreadsMax = $CpuBudget }
if ($RefitThreads -eq 0) { $RefitThreads = $CpuBudget }
$Experiment = Join-Path $Root 'outputs_wsl\simulations\eas_four_pair_rl_v1\split_comparison_v1'
if ($Method -eq 'status') {
    foreach ($name in @('nn','mle')) {
        $path = Join-Path $Experiment ("arms\$name\extension_status_011_020.json")
        if ($name -eq 'mle' -and -not (Test-Path -LiteralPath $path)) {
            $path = Join-Path $Experiment 'arms\mle\mle\extension_status_011_020.json'
        }
        if (Test-Path -LiteralPath $path) {
            Write-Host "${name}:"
            Get-Content -LiteralPath $path -Raw
        } else {
            Write-Host "${name}: 11--20 not started"
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
$Log = Join-Path $LogDir ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_windows_extend_11_20_' + $Method + '.log')
$Arguments = @('-B','-u',(Join-Path $Root 'extend-11-20.py'),$Method,
    '--threads',"$Threads",'--jobs',"$Jobs",'--mle-jobs',"$MleJobs",
    '--cpu-budget',"$CpuBudget",'--worker-threads-max',"$WorkerThreadsMax",
    '--refit-threads',"$RefitThreads")
Start-Transcript -LiteralPath $Log | Out-Null
try {
    Write-Host "Windows CPU mode: $Method; formal buffer: 500 km"
    Write-Host "Detected CPUs: $DetectedCpus; GP n_jobs: $Jobs; MLE jobs: $MleJobs"
    Write-Warning 'No RAM-based worker throttle. Exact GP may exhaust memory; verified checkpoints are reusable.'
    Write-Host "Log: $Log"
    Write-Host 'Replicates 001--010 and their manifests will not be overwritten.'
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Windows 11--20 continuation stopped (exit $LASTEXITCODE). Check $Log."
    }
    Write-Host "Result directory: $Experiment"
} finally {
    Stop-Transcript | Out-Null
}
