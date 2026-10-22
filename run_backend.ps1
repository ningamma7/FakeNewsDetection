$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
if (-not (Test-Path '.venv\Scripts\python.exe')) { throw "Virtual environment missing. Run .\setup_windows.ps1 first." }
if (-not (Test-Path '.env')) { throw ".env missing. Run .\setup_windows.ps1 first and set your private REVIEWER_TOKEN." }
& .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
