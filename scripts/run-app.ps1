$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve-venv.ps1")

$backendScript = Join-Path $PSScriptRoot "run-backend.ps1"
$frontendScript = Join-Path $PSScriptRoot "run-frontend.ps1"
$venv = Resolve-WorkbenchVenv -Root $root
$venvPython = Join-Path $venv "Scripts\python.exe"
$frontendModules = Join-Path $root "frontend\node_modules"
$frontendUrl = "http://localhost:5173"

function Quote-PowerShellString {
  param([Parameter(Mandatory = $true)][string]$Value)
  return "'" + $Value.Replace("'", "''") + "'"
}

function Start-AppWindow {
  param(
    [Parameter(Mandatory = $true)][string]$Title,
    [Parameter(Mandatory = $true)][string]$ScriptPath
  )

  $command = "`$Host.UI.RawUI.WindowTitle = $(Quote-PowerShellString $Title); & $(Quote-PowerShellString $ScriptPath)"
  Start-Process powershell.exe -WorkingDirectory $root -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    $command
  )
}

function Wait-ForHttp {
  param(
    [Parameter(Mandatory = $true)][string]$Url,
    [int]$TimeoutSeconds = 30
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method Get -TimeoutSec 2
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 750
    }
  }

  return $false
}

if (-not (Test-Path $venvPython)) {
  throw "Virtual environment missing. Run scripts/install.ps1 first."
}

if (-not (Test-Path $frontendModules)) {
  throw "Frontend dependencies missing. Run scripts/install.ps1 first."
}

Write-Host "Starting Subtitle Workbench..."
Start-AppWindow -Title "Subtitle Workbench Backend" -ScriptPath $backendScript
Start-AppWindow -Title "Subtitle Workbench Frontend" -ScriptPath $frontendScript

Write-Host "Waiting for the frontend at $frontendUrl ..."
if (-not (Wait-ForHttp -Url $frontendUrl -TimeoutSeconds 45)) {
  Write-Warning "The frontend did not respond yet. Opening the app URL anyway; check the backend/frontend windows for errors."
}

Start-Process $frontendUrl
Write-Host "Opened $frontendUrl"
