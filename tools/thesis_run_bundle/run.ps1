param(
    [ValidateSet('check','prepare-terrain','eas','simulate','compare-mle','global','status','import-eas')][string]$Mode = 'check',
    [ValidateSet('legacy_v1','expanded_v2','expanded_latitude_v3')][string]$TerrainProfile = 'legacy_v1',
    [ValidateSet(1,100)][int]$Replicates = 1,
    [ValidateRange(1,32)][int]$Threads = 1,
    [ValidateScript({ $_ -ne 0 })][int]$Jobs = -2,
    [ValidateSet('staged','fixed','adaptive')][string]$ParallelMode = 'staged',
    [ValidateRange(1,1024)][double]$MemoryBudgetGiB = 20,
    [ValidateRange(0,128)][double]$MemoryReserveGiB = 2,
    [ValidateRange(0,256)][int]$CpuBudget = 0,
    [ValidateRange(1,32)][int]$WorkerThreadsMax = 4,
    [ValidateRange(1,32)][int]$RefitThreads = 4,
    [switch]$NewRun,
    [ValidateScript({ $_ -ne 0 })][int]$MleJobs = 1,
    [ValidateSet('wide_domain','training_domain')][string]$MleDomain = 'wide_domain',
    [ValidateRange(1,4)][int]$Workers = 2,
    [string]$Experiment = 'eas_calibrated_v1',
    [string]$DataRoot = '',
    [string]$Settings = '',
    [string]$Source = '',
    [switch]$SkipPreprocess
)
. (Join-Path $PSScriptRoot 'environment.ps1')
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run setup.ps1 on this computer first.' }
$Arguments = @('-B','-u',(Join-Path $BundleRoot 'run.py'),$Mode,'--threads',"$Threads",
    '--jobs',"$Jobs",'--mle-jobs',"$MleJobs",'--mle-domain',$MleDomain,
    '--replicates',"$Replicates",'--workers',"$Workers",'--experiment',$Experiment)
$Arguments += @('--parallel-mode',$ParallelMode,'--worker-threads-max',"$WorkerThreadsMax",'--refit-threads',"$RefitThreads")
$Arguments += @('--terrain-profile',$TerrainProfile,
    '--memory-budget-gib',$MemoryBudgetGiB.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--memory-reserve-gib',$MemoryReserveGiB.ToString([System.Globalization.CultureInfo]::InvariantCulture))
if ($CpuBudget -gt 0) { $Arguments += @('--cpu-budget',"$CpuBudget") }
if ($NewRun) { $Arguments += '--new-run' }
if ($DataRoot) { $Arguments += @('--data-root',$DataRoot) }
if ($Settings) { $Arguments += @('--settings',$Settings) }
if ($Source) { $Arguments += @('--source',$Source) }
if ($SkipPreprocess) { $Arguments += '--skip-preprocess' }
$LogDir = Join-Path $BundleRoot 'outputs\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ((Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '_' + $Mode + '.log')
Start-Transcript -LiteralPath $Log | Out-Null
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Runner stopped (exit $LASTEXITCODE). See $Log" }
} finally { Stop-Transcript | Out-Null }
