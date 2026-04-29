param(
  [switch]$SkipRebuild,
  [switch]$NoBrowser,
  [switch]$LiveData,
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [string]$PythonPath = "C:\Users\Administrator\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DemoDb = Join-Path $RepoRoot "data\flood_warning_system_demo.db"
$FrontendDir = Join-Path $RepoRoot "frontend"
$FrontendDemoMode = if ($LiveData) { "false" } else { "true" }

if (-not (Test-Path $PythonPath)) {
  $PythonPath = "python"
}

Write-Host "[AgentTwin Demo] Repo: $RepoRoot" -ForegroundColor Cyan
Write-Host "[AgentTwin Demo] Demo DB: $DemoDb" -ForegroundColor Cyan

Push-Location $RepoRoot
try {
  if (-not $SkipRebuild) {
    Write-Host "[AgentTwin Demo] Rebuilding demo database..." -ForegroundColor Yellow
    & $PythonPath "scripts\rebuild_demo_db.py" --force
  }

  Write-Host "[AgentTwin Demo] Inspecting demo database..." -ForegroundColor Yellow
  & $PythonPath "scripts\inspect_demo_db.py"
}
finally {
  Pop-Location
}

$BackendCommand = @"
`$env:FLOOD_DB_PATH = '$DemoDb'
Set-Location '$RepoRoot'
& '$PythonPath' -m uvicorn flood_system.api:app --host 127.0.0.1 --port $BackendPort
"@

$FrontendCommand = @"
`$env:VITE_DEMO_MODE = '$FrontendDemoMode'
Set-Location '$FrontendDir'
npm.cmd run dev -- --host 127.0.0.1 --port $FrontendPort
"@

Write-Host "[AgentTwin Demo] Starting backend on http://127.0.0.1:$BackendPort" -ForegroundColor Green
Start-Process powershell.exe -ArgumentList @("-NoExit", "-NoProfile", "-Command", $BackendCommand) -WorkingDirectory $RepoRoot

Write-Host "[AgentTwin Demo] Starting frontend on http://127.0.0.1:$FrontendPort" -ForegroundColor Green
Start-Process powershell.exe -ArgumentList @("-NoExit", "-NoProfile", "-Command", $FrontendCommand) -WorkingDirectory $FrontendDir

Write-Host ""
Write-Host "[AgentTwin Demo] Open: http://127.0.0.1:$FrontendPort" -ForegroundColor Cyan
Write-Host "[AgentTwin Demo] Event ID: event_demo_beilin_primary" -ForegroundColor Cyan
Write-Host "[AgentTwin Demo] Backend env FLOOD_DB_PATH is pinned to the demo database." -ForegroundColor Cyan
Write-Host "[AgentTwin Demo] Frontend VITE_DEMO_MODE=$FrontendDemoMode. Use -LiveData to consume live platform data only." -ForegroundColor Cyan

if (-not $NoBrowser) {
  Start-Process "http://127.0.0.1:$FrontendPort"
}
