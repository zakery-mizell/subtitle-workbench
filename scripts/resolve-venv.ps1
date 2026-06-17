function Resolve-WorkbenchVenv {
  param([Parameter(Mandatory = $true)][string]$Root)

  $override = $env:SUBTITLE_WORKBENCH_VENV
  if ($null -eq $override) {
    $override = ""
  }
  $override = $override.Trim()
  $pathFile = Join-Path $Root ".venv-path"

  if (-not $override -and (Test-Path $pathFile)) {
    $override = (Get-Content $pathFile -Raw).Trim()
  }

  if ($override) {
    return $override
  }

  return Join-Path $Root ".venv"
}
