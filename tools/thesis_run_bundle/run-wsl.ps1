param(
    [ValidateSet('check','eas','simulate','compare-mle','replay-nn','global','status')][string]$Mode = 'check',
    [ValidateSet(1,100)][int]$Replicates = 1,
    [string]$Distro = 'Ubuntu',
    [int]$Jobs = -2,
    [ValidateSet('staged','fixed')][string]$ParallelMode = 'staged',
    [ValidateRange(1,32)][int]$Threads = 1,
    [ValidateRange(0,256)][int]$CpuBudget = 0,
    [ValidateRange(1,32)][int]$WorkerThreadsMax = 4,
    [ValidateRange(1,32)][int]$RefitThreads = 4,
    [int]$MleJobs = 1,
    [ValidateSet('wide_domain','training_domain')][string]$MleDomain = 'wide_domain',
    [ValidateRange(1,4)][int]$Workers = 2,
    [string]$Experiment = 'eas_wsl_staged_v1',
    [string]$Settings = '',
    [string]$Source = '',
    [switch]$SkipPreprocess,
    [switch]$NewRun
)
$ErrorActionPreference = 'Stop'
$LinuxRoot = (& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxRoot) { throw 'Cannot locate bundle in WSL.' }
$Arguments = @($Mode,'--replicates',"$Replicates",'--jobs',"$Jobs",'--threads',"$Threads",
    '--parallel-mode',$ParallelMode,'--worker-threads-max',"$WorkerThreadsMax",'--refit-threads',"$RefitThreads",
    '--mle-jobs',"$MleJobs",'--mle-domain',$MleDomain,'--workers',"$Workers",'--experiment',$Experiment)
if ($CpuBudget -gt 0) { $Arguments += @('--cpu-budget',"$CpuBudget") }
if ($NewRun) { $Arguments += '--new-run' }
if ($SkipPreprocess) { $Arguments += '--skip-preprocess' }
if ($Settings) {
    $LinuxSettings = (& wsl.exe -d $Distro --exec wslpath -a -u (Resolve-Path -LiteralPath $Settings).Path).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot convert settings path.' }
    $Arguments += @('--settings',$LinuxSettings)
}
if ($Source) {
    $LinuxSource = (& wsl.exe -d $Distro --exec wslpath -a -u (Resolve-Path -LiteralPath $Source).Path).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot convert source path.' }
    $Arguments += @('--source',$LinuxSource)
}
& wsl.exe -d $Distro --exec bash "$LinuxRoot/wsl/run.sh" $LinuxRoot @Arguments
if ($LASTEXITCODE -ne 0) { throw 'WSL workflow failed. Check outputs_wsl/logs; keep caches.' }
