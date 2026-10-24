param([string]$Distro = 'Ubuntu')
$ErrorActionPreference = 'Stop'
$LinuxRoot = (& wsl.exe -d $Distro --exec wslpath -a -u $PSScriptRoot).Trim()
if ($LASTEXITCODE -ne 0 -or -not $LinuxRoot) { throw 'Install WSL Ubuntu and finish first-user setup first.' }
& wsl.exe -d $Distro --exec bash "$LinuxRoot/wsl/setup.sh" $LinuxRoot
if ($LASTEXITCODE -ne 0) { throw 'WSL CPU setup failed; no formal workflow started.' }
