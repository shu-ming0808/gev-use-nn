param([string]$Distro = 'Ubuntu-24.04', [switch]$SkipCheck)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$LinuxRoot = ((& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot) -join "`n").Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxRoot) { throw 'Install WSL Ubuntu and finish first-user setup first.' }
$Arguments = @($LinuxRoot)
if ($SkipCheck) { $Arguments += '--skip-check' }
& wsl.exe -d $Distro --exec bash "$LinuxRoot/wsl/setup.sh" @Arguments
if ($LASTEXITCODE -ne 0) { throw 'WSL CPU setup failed; no formal workflow started.' }
