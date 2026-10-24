$ErrorActionPreference = 'Stop'
$BundleRoot = $PSScriptRoot
$Uv = Join-Path $BundleRoot 'bin\uv.exe'
if (-not (Test-Path -LiteralPath $Uv)) { throw 'Missing bundled bin\uv.exe.' }
$Manifest = Get-Content -Raw -LiteralPath (Join-Path $BundleRoot 'manifest.json') | ConvertFrom-Json
$ExpectedUv = $Manifest.files.'bin/uv.exe'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Uv).Hash.ToLower() -ne $ExpectedUv) {
    throw 'Bundled uv checksum mismatch.'
}
# Machine-local paths avoid copying a venv with invalid D:/F: interpreter links.
$Runtime = Join-Path $env:LOCALAPPDATA 'ThesisRunCPU\v1'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $Runtime 'env'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime 'python'
$env:UV_CACHE_DIR = Join-Path $Runtime 'cache'
$env:UV_LINK_MODE = 'copy'
$env:UV_SYSTEM_CERTS = 'true'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$Python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
