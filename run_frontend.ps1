$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $Root 'frontend')
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm not found. Install Node.js 20+ and restart the terminal." }
if (-not (Test-Path 'node_modules')) { npm install }
npm run dev
