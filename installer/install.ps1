# Open Mocap installer launcher (Windows PowerShell).
# Fetches uv (a standalone Python manager) into this folder if needed, then
# runs install.py on a managed Python 3.12. Arguments pass through, e.g.
#   powershell -ExecutionPolicy Bypass -File install.ps1 --profile both
$ErrorActionPreference = "Stop"
$Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Dir

$UvLocal = Join-Path $Dir ".uv\uv.exe"
if (Test-Path $UvLocal) {
    $Uv = $UvLocal
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    $Uv = (Get-Command uv).Source
} else {
    Write-Host "[Open Mocap] Fetching uv (Python environment manager)..."
    $env:UV_INSTALL_DIR = Join-Path $Dir ".uv"
    $env:UV_NO_MODIFY_PATH = "1"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $Uv = $UvLocal
}

$env:OPENMOCAP_UV = $Uv
& $Uv run --no-project --python 3.12 (Join-Path $Dir "install.py") @args 2>&1 | Tee-Object -FilePath (Join-Path $Dir "install.log")
exit $LASTEXITCODE
