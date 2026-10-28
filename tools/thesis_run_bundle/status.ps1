param(
    [string]$Root = $PSScriptRoot,
    [string]$Distro = 'Ubuntu-24.04',
    [ValidateRange(1,100)][int]$Latest = 4,
    [switch]$LiveWSL,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

# Read-only: do not stage code, install packages, acquire locks or start workflows.
function Read-JsonFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return [pscustomobject]@{ data=$null; error='File not found' }
    }
    try {
        $data = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
        return [pscustomobject]@{ data=$data; error=$null }
    } catch {
        return [pscustomobject]@{ data=$null; error=('Cannot read JSON: ' + $_.Exception.Message) }
    }
}

function Find-RunFiles([string]$Directory, [string]$Name) {
    # Visit directories, but do not enumerate the potentially large data files.
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) { return }
    $candidate = Join-Path $Directory $Name
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        Get-Item -LiteralPath $candidate
    }
    foreach ($child in @(Get-ChildItem -LiteralPath $Directory -Directory -ErrorAction SilentlyContinue)) {
        if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            Find-RunFiles $child.FullName $Name
        }
    }
}

function Test-CompletedReport($Report) {
    return ($null -ne $Report -and $Report.workflow_completed -eq $true -and
        $Report.status -in @('completed', 'completed_with_warnings'))
}

function Read-LogTail([string]$Path) {
    # Stream a bounded snapshot. Windows PowerShell 5.1 Get-Content -Tail can
    # stall on some short UTF-8 logs without a final newline.
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8, $true)
    $tail = New-Object 'System.Collections.Generic.Queue[string]'
    try {
        while (-not $reader.EndOfStream) {
            $tail.Enqueue($reader.ReadLine())
            if ($tail.Count -gt 25) { $null = $tail.Dequeue() }
        }
        $tail.ToArray()
    } finally { $reader.Dispose() }
}

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$notice = 'Saved statuses are NOT live process checks. A saved running status may be stale after interruption or disk transfer. A directory or checkpoint count does not prove a completed workflow.'
$surfaces = @()
foreach ($name in @('outputs_wsl', 'outputs')) {
    $directory = Join-Path $resolvedRoot $name
    $runs = @()
    foreach ($file in @(Find-RunFiles $directory 'run_status.json' | Sort-Object LastWriteTime -Descending | Select-Object -First $Latest)) {
        $loaded = Read-JsonFile $file.FullName
        $state = $loaded.data
        $reportPath = Join-Path $file.DirectoryName 'run_report.json'
        $report = Read-JsonFile $reportPath
        $elapsed = $state.elapsed_from_estimator_start_seconds
        if ($null -eq $elapsed) { $elapsed = $state.elapsed_from_nn_start_seconds }
        if ($null -eq $elapsed) { $elapsed = $state.elapsed_seconds }
        if ($null -eq $elapsed -and $null -ne $report.data) {
            $elapsed = $report.data.timings.total_from_estimator_start_until_outputs_seconds
            if ($null -eq $elapsed) { $elapsed = $report.data.timings.total_from_nn_start_until_outputs_seconds }
        }
        $runs += [pscustomobject]@{
            path=$file.FullName; last_write_time=$file.LastWriteTime.ToString('o')
            saved_status=$state.status; stage=$state.stage; coefficient=$state.coefficient
            outer_fold=$state.outer_fold; completed_cases=$state.completed_cases; total_cases=$state.total_cases
            completed_coefficients=$state.completed_coefficients; total_coefficients=$state.total_coefficients
            completed_pairs=$state.completed_pairs; replicate=$state.replicate; estimator=$state.estimator
            elapsed_seconds=$elapsed; progress_file=$state.progress_file
            workflow_completed=$state.workflow_completed
            completion_confirmed_by_report=(Test-CompletedReport $report.data)
            error=$loaded.error
        }
    }
    $completed = @()
    foreach ($file in @(Find-RunFiles $directory 'run_report.json' | Sort-Object LastWriteTime -Descending)) {
        $loaded = Read-JsonFile $file.FullName
        if (Test-CompletedReport $loaded.data) {
            $elapsed = $loaded.data.timings.total_from_estimator_start_until_outputs_seconds
            if ($null -eq $elapsed) { $elapsed = $loaded.data.timings.total_from_nn_start_until_outputs_seconds }
            $completed += [pscustomobject]@{
                path=$file.FullName; last_write_time=$file.LastWriteTime.ToString('o')
                status=$loaded.data.status; workflow_completed=$true; elapsed_seconds=$elapsed
                estimator=$loaded.data.estimator; n_grids=$loaded.data.n_grids
            }
            if ($completed.Count -ge $Latest) { break }
        }
    }
    $log = $null
    $logDirectory = Join-Path $directory 'logs'
    if (Test-Path -LiteralPath $logDirectory -PathType Container) {
        $logFile = Get-ChildItem -LiteralPath $logDirectory -File -Filter '*.log' |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -ne $logFile) {
            $log = [pscustomobject]@{
                path=$logFile.FullName; last_write_time=$logFile.LastWriteTime.ToString('o')
                tail=@(Read-LogTail $logFile.FullName)
            }
        }
    }
    $surfaces += [pscustomobject]@{ name=$name; exists=(Test-Path -LiteralPath $directory -PathType Container); runs=$runs; completed_reports=$completed; latest_log=$log }
}

$live = $null
if ($LiveWSL) {
    $liveNotice = 'Explicit LiveWSL check may start the selected distro; no workflow is launched. Linux ps %CPU is a lifetime average and can exceed 100 per process; it is not instantaneous whole-machine usage.'
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        $live = [pscustomobject]@{ notice=$liveNotice; error='wsl.exe not found'; output=@() }
    } else {
        # Arguments are passed directly; distro names are not interpolated into shell code.
        $command = "printf 'Visible CPUs: '; nproc; free -h; ps -eo pid,ppid,pcpu,pmem,etime,comm --sort=-pcpu | head -n 16"
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $lines = @(& wsl.exe -d $Distro --exec sh -c $command 2>&1 | ForEach-Object { "$_" })
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $oldPreference
        $live = [pscustomobject]@{ notice=$liveNotice; distro=$Distro; exit_code=$exitCode; output=$lines }
    }
}

$result = [pscustomobject]@{ root=$resolvedRoot; notice=$notice; live_processes_checked=[bool]$LiveWSL; surfaces=$surfaces; live_wsl=$live }
if ($Json) { $result | ConvertTo-Json -Depth 10; exit 0 }
Write-Output $notice
foreach ($surface in $surfaces) {
    Write-Output ("`n=== " + $surface.name + ' ===')
    if (-not $surface.exists) { Write-Output 'No output directory found.'; continue }
    if ($surface.runs.Count -eq 0) { Write-Output 'No saved run_status.json found.' }
    foreach ($run in $surface.runs) { $run | Format-List }
    Write-Output 'Completed workflows confirmed by run_report.json (copies may duplicate a run):'
    if ($surface.completed_reports.Count -eq 0) { Write-Output 'None found.' }
    foreach ($report in $surface.completed_reports) { $report | Format-List }
    if ($null -ne $surface.latest_log) {
        Write-Output ('Latest saved log: ' + $surface.latest_log.path)
        Write-Output ('Last write: ' + $surface.latest_log.last_write_time)
        $surface.latest_log.tail | Write-Output
    }
}
if ($null -ne $live) { $live.notice | Write-Output; $live.error | Write-Output; $live.output | Write-Output }
