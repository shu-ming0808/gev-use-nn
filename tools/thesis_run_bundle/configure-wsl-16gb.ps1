[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $env:USERPROFILE '.wslconfig'),
    [ValidateRange(1,256)][int]$Processors,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
$Path = [IO.Path]::GetFullPath($ConfigPath)
$Old = if (Test-Path -LiteralPath $Path) { [IO.File]::ReadAllText($Path) } else { '' }
$Lines = [Collections.Generic.List[string]]::new()
if ($Old.Length) { foreach ($Line in ($Old -split '\r?\n')) { $Lines.Add($Line) } }
$Sections = @(for ($i=0; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i] -match '^\s*\[wsl2\]\s*(?:[#;].*)?$') { $i }
})
if ($Sections.Count -gt 1) { throw 'Multiple [wsl2] sections found. No changes made; review the config first.' }
if ($Sections.Count -eq 0) {
    if ($Lines.Count -gt 0 -and $Lines[$Lines.Count-1] -ne '') { $Lines.Add('') }
    $Start = $Lines.Count
    $Lines.Add('[wsl2]')
} else { $Start = $Sections[0] }
$End = $Lines.Count
for ($i=$Start+1; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i] -match '^\s*\[') { $End=$i; break }
}
$Settings = @(@('memory','10GB'), @('swap','4GB'))
if ($PSBoundParameters.ContainsKey('Processors')) {
    $Settings += ,@('processors', [string]$Processors)
}
foreach ($Setting in $Settings) {
    $Key = $Setting[0]
    $Value = $Setting[1]
    $MatchesAt = @(for ($i=$Start+1; $i -lt $End; $i++) {
        if ($Lines[$i] -match ('^\s*' + $Key + '\s*=')) { $i }
    })
    if ($MatchesAt.Count -gt 1) { throw ('Duplicate ' + $Key + ' keys found. No changes made.') }
    if ($MatchesAt.Count -eq 1) { $Lines[$MatchesAt[0]] = $Key + '=' + $Value }
    else { $Lines.Insert($End, $Key + '=' + $Value); $End++ }
}
$New = ($Lines -join "`r`n").TrimEnd("`r","`n") + "`r`n"
$Changed = $New -cne $Old
$Backup = $null
if ($Apply -and $Changed) {
    if (-not (Test-Path -LiteralPath ([IO.Path]::GetDirectoryName($Path)))) { throw 'Config parent directory is missing.' }
    if (Test-Path -LiteralPath $Path) {
        if ([IO.File]::ReadAllText($Path) -cne $Old) { throw 'Config changed during inspection. No update made.' }
        $Backup = $Path + '.backup-' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
        Copy-Item -LiteralPath $Path -Destination $Backup -ErrorAction Stop
    }
    [IO.File]::WriteAllText($Path, $New, [Text.UTF8Encoding]::new($false))
    if ([IO.File]::ReadAllText($Path) -cne $New) { throw 'Written config did not verify. Keep the backup.' }
}
[pscustomobject]@{
    config_file=$Path; applied=[bool]$Apply; changed=$Changed
    memory='10GB'; swap='4GB'; backup=$Backup
    processors=$(if ($PSBoundParameters.ContainsKey('Processors')) { $Processors } else { 'preserved' })
    wsl_stopped=$false; workflow_started=$false
    note='Global WSL2 setting. Only memory, swap and explicitly supplied processors are updated. Stop all WSL/Docker jobs before manually running wsl --shutdown, then restart. Swap uses disk and is not extra physical RAM.'
} | ConvertTo-Json -Depth 3
