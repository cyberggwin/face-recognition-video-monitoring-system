param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listeners) {
    Write-Host "Nu exista niciun proces care asculta pe portul $Port." -ForegroundColor Yellow
    exit 0
}

$pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($procId in $pids) {
    try {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $procId"
        Write-Host "Oprire PID $procId ($($proc.Name))" -ForegroundColor Cyan
        Stop-Process -Id $procId -Force -ErrorAction Stop
    }
    catch {
        Write-Host "Nu am putut opri PID ${procId}: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Server oprit pe portul $Port." -ForegroundColor Green
