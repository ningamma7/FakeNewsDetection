$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "Setting up VeriLens Research Platform at $Root" -ForegroundColor Cyan
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.12 from python.org and retry."
}
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    $Token = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
    $Content = Get-Content '.env' -Raw
    $Content = $Content -replace 'REVIEWER_TOKEN=.*', "REVIEWER_TOKEN=$Token"
    Set-Content '.env' $Content -Encoding utf8
    Write-Host "Created .env and generated a reviewer token. Keep .env private." -ForegroundColor Green
}
if (Get-Command npm -ErrorAction SilentlyContinue) {
    Push-Location frontend
    if (-not (Test-Path 'node_modules')) { npm install }
    Pop-Location
} else {
    Write-Warning "Node.js/npm not detected. Install Node.js 20+ then run 'npm install' inside frontend."
}
Write-Host "Setup complete. Start the backend with .\run_backend.ps1 and the frontend with .\run_frontend.ps1" -ForegroundColor Green
