param()
. (Join-Path $PSScriptRoot 'environment.ps1')
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$ManagedPython = Join-Path $env:UV_PYTHON_INSTALL_DIR 'cpython-3.10.20-windows-x86_64-none\python.exe'
$InstallExit = 0
if (-not (Test-Path -LiteralPath $ManagedPython)) {
    & $Uv python install 3.10.20 --no-bin --no-registry
    $InstallExit = $LASTEXITCODE
}
# Some Windows uv builds fail creating the optional minor-version junction
# after successful extraction. Verify the actual interpreter, then use it
# directly; do not depend on a junction, PATH, or registry registration.
if (-not (Test-Path -LiteralPath $ManagedPython)) { throw 'Python download/extraction failed. Check network and certificates.' }
& $ManagedPython -c "import sys,ssl; assert sys.version_info[:3] == (3,10,20); print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw 'Downloaded Python did not pass executable/version validation.' }
if ($InstallExit -ne 0) { Write-Warning 'Optional Python alias was not created; verified direct interpreter is used.' }
& $Uv sync --project $BundleRoot --locked --no-dev --python $ManagedPython --no-python-downloads
if ($LASTEXITCODE -ne 0) { throw 'CPU environment installation failed.' }
& $Python -B (Join-Path $BundleRoot 'run.py') check
if ($LASTEXITCODE -ne 0) { throw 'Bundle validation failed. For missing DLL/WinError 126, see the Microsoft Visual C++ runtime step in the instructions.' }
Write-Host 'Ready. Python and CPU packages installed in this computer LocalAppData.'
