$root = Split-Path -Parent $PSScriptRoot

Push-Location (Join-Path $root "frontend")
try {
  npm run dev
} finally {
  Pop-Location
}
