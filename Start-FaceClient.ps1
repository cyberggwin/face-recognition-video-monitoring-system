$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = ".\venv311\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Nu exista $pythonExe. Ruleaza setup-ul de mediu din Comenzi.md."
}

& $pythonExe .\src\face_client_windows.py
exit $LASTEXITCODE
