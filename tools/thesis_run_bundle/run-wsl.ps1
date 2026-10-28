param(
    [ValidateSet('check','prepare-terrain','eas','simulate','compare-mle','replay-nn','global','status')][string]$Mode = 'check',
    [ValidateSet('legacy_v1','expanded_v2','expanded_latitude_v3')][string]$TerrainProfile = 'legacy_v1',
    [ValidateSet(1,100)][int]$Replicates = 1,
    [string]$Distro = 'Ubuntu-24.04',
    [int]$Jobs = 4,
    [ValidateSet('staged','fixed','adaptive')][string]$ParallelMode = 'adaptive',
    [ValidateRange(1,1024)][double]$MemoryBudgetGiB = 20,
    [ValidateRange(0,128)][double]$MemoryReserveGiB = 2,
    [ValidateSet('alternating','mle-first','nn-first')][string]$MethodOrder = 'alternating',
    [ValidateRange(1,32)][int]$Threads = 8,
    [ValidateRange(0,256)][int]$CpuBudget = 18,
    [ValidateRange(1,32)][int]$WorkerThreadsMax = 4,
    [ValidateRange(1,32)][int]$RefitThreads = 8,
    [int]$MleJobs = 16,
    [ValidateSet('wide_domain','training_domain')][string]$MleDomain = 'wide_domain',
    [ValidateRange(1,4)][int]$Workers = 2,
    [string]$Experiment = 'eas_wsl_u265_v1',
    [string]$Settings = '',
    [string]$Source = '',
    [switch]$SkipPreprocess,
    [switch]$NewRun
)
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5 otherwise decodes UTF-8 wslpath output as the local codepage.
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$LinuxRoot = ((& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot) -join "`n").Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxRoot) { throw 'Cannot locate bundle in WSL.' }
$Arguments = @($Mode,'--replicates',"$Replicates",'--jobs',"$Jobs",'--threads',"$Threads",
    '--parallel-mode',$ParallelMode,'--worker-threads-max',"$WorkerThreadsMax",'--refit-threads',"$RefitThreads",
    '--mle-jobs',"$MleJobs",'--mle-domain',$MleDomain,'--workers',"$Workers",'--experiment',$Experiment,
    '--memory-budget-gib',$MemoryBudgetGiB.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--memory-reserve-gib',$MemoryReserveGiB.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--method-order',$MethodOrder,'--terrain-profile',$TerrainProfile)
if ($CpuBudget -gt 0) { $Arguments += @('--cpu-budget',"$CpuBudget") }
if ($NewRun) { $Arguments += '--new-run' }
if ($SkipPreprocess) { $Arguments += '--skip-preprocess' }
if ($Settings) {
    $LinuxSettings = ((& wsl.exe -d $Distro --exec wslpath -a -u (Resolve-Path -LiteralPath $Settings).Path) -join "`n").Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot convert settings path.' }
    $Arguments += @('--settings',$LinuxSettings)
}
if ($Source) {
    $LinuxSource = ((& wsl.exe -d $Distro --exec wslpath -a -u (Resolve-Path -LiteralPath $Source).Path) -join "`n").Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot convert source path.' }
    $Arguments += @('--source',$LinuxSource)
}
& wsl.exe -d $Distro --exec bash "$LinuxRoot/wsl/run.sh" $LinuxRoot @Arguments
if ($LASTEXITCODE -ne 0) { throw 'WSL workflow failed. Check outputs_wsl/logs; keep caches.' }
