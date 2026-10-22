param([switch]$FullBaselines)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "Python environment missing. Run .\setup_windows.ps1 first." }

function Run-PythonStep([string[]]$ArgsForPython) {
    Write-Host ("Running: python " + ($ArgsForPython -join ' ')) -ForegroundColor Cyan
    & $Python @ArgsForPython
    if ($LASTEXITCODE -ne 0) { throw "Step failed: python $($ArgsForPython -join ' ')" }
}

Run-PythonStep @('-m','backend.training.prepare_dataset')
Run-PythonStep @('-m','backend.training.create_research_splits')
Run-PythonStep @('-m','backend.training.validate_splits')
Run-PythonStep @('-m','backend.training.train_tfidf_baseline','--protocol','random')
Run-PythonStep @('-m','backend.training.evaluate_model','--model-type','tfidf','--model-path','backend/models/tfidf_random.joblib','--test-csv','data/processed/research_splits/random/test.csv')

if ($FullBaselines) {
    Run-PythonStep @('-m','backend.training.train_tfidf_baseline','--protocol','temporal')
    Run-PythonStep @('-m','backend.training.evaluate_model','--model-type','tfidf','--model-path','backend/models/tfidf_temporal.joblib','--test-csv','data/processed/research_splits/temporal/test.csv')
}
Write-Host "Dataset and split preparation complete. Review outputs\research and data\processed\research_splits." -ForegroundColor Green
