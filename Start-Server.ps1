param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8000,
    [string]$Source = "0",
    [string]$Encodings = ".\\models\\known_faces.pkl",
    [string]$AuthToken = "",
    [string]$FcmServiceAccount = "",
    [string]$DeviceRegistry = ".\\models\\device_tokens.json",
    [string]$PushTitlePrefix = "FaceClient",
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = ".\\venv311\\Scripts\\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Nu exista $pythonExe. Ruleaza setup-ul de mediu din Comenzi.md."
}

if ($FcmServiceAccount -eq "") {
    $defaultFcm = ".\\credentials\\fcm-service-account.json"
    if (Test-Path $defaultFcm) {
        $FcmServiceAccount = $defaultFcm
    }
}

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    if (-not $ForceRestart) {
        Write-Host "Portul $Port este deja ocupat de PID: $($pids -join ', ')." -ForegroundColor Yellow
        Write-Host "Ruleaza: .\\Start-Server.ps1 -ForceRestart" -ForegroundColor Yellow
        exit 1
    }

    foreach ($procId in $pids) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "Am oprit procesul PID $procId de pe portul $Port." -ForegroundColor Cyan
        }
        catch {
            Write-Host "Nu am putut opri PID ${procId}: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    }

    Start-Sleep -Milliseconds 400
}

$args = @(
    ".\\src\\live_api_server.py",
    "--host", $BindHost,
    "--port", "$Port",
    "--source", $Source,
    "--encodings", $Encodings
)

if ($AuthToken -ne "") {
    $args += @("--auth-token", $AuthToken)
}

if ($FcmServiceAccount -ne "") {
    $args += @("--fcm-service-account", $FcmServiceAccount)
}

if ($DeviceRegistry -ne "") {
    $args += @("--device-registry", $DeviceRegistry)
}

if ($PushTitlePrefix -ne "") {
    $args += @("--push-title-prefix", $PushTitlePrefix)
}

Write-Host "Pornesc serverul pe http://${BindHost}:$Port ..." -ForegroundColor Green
if ($FcmServiceAccount -ne "") {
    Write-Host "Push mode: ENABLED (service account: $FcmServiceAccount)" -ForegroundColor Green
}
else {
    Write-Host "Push mode: DISABLED (fara FCM service account)" -ForegroundColor Yellow
}
& $pythonExe @args
exit $LASTEXITCODE
