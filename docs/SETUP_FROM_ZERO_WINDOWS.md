# Windows setup from zero

This guide assumes you extracted the ZIP to `D:\FakeNewsResearchPlatform`. It does not modify the separate `D:\FakeNewsDetection` project.

## 1. Confirm the included dataset

The project ZIP includes `data\raw\Fake.csv` and `data\raw\True.csv`. Confirm both files exist after extraction. If they are missing, copy your own source files. For example, if the originals are in `D:\FakeNewsDetection\backend\data`:

```powershell
New-Item -ItemType Directory -Force D:\FakeNewsResearchPlatform\data\raw | Out-Null
Copy-Item D:\FakeNewsDetection\backend\data\Fake.csv D:\FakeNewsResearchPlatform\data\raw\Fake.csv
Copy-Item D:\FakeNewsDetection\backend\data\True.csv D:\FakeNewsResearchPlatform\data\raw\True.csv
```

If copying is needed, change only the source paths. The destination filenames must be exactly `Fake.csv` and `True.csv`; extract ZIP files first if needed.

## 2. Setup

Open VS Code at `D:\FakeNewsResearchPlatform`. Open PowerShell and run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup_windows.ps1
```

The script creates a Python 3.12 virtual environment, installs backend dependencies, creates `.env` with a private reviewer token, and installs frontend npm packages if Node.js is available. Keep `.env` private.

## 3. Build the dataset and baseline

Run from the project root. The scripted sequence is:

```powershell
.\prepare_research_data.ps1
```

For both TF-IDF baseline protocols (random and temporal), use:

```powershell
.\prepare_research_data.ps1 -FullBaselines
```

Or run the steps individually:

```powershell
.\.venv\Scripts\python.exe -m backend.training.prepare_dataset
.\.venv\Scripts\python.exe -m backend.training.create_research_splits
.\.venv\Scripts\python.exe -m backend.training.validate_splits
.\.venv\Scripts\python.exe -m backend.training.train_tfidf_baseline --protocol random
.\.venv\Scripts\python.exe -m backend.training.evaluate_model --model-type tfidf --model-path backend\models\tfidf_random.joblib --test-csv data\processed\research_splits\random\test.csv
```

Review `outputs\research\dataset_preparation_report.json`, `data\processed\research_splits\split_report.json`, and `outputs\research\split_validation.json`. Report only metrics from runs that completed on your machine.

## 4. Start the website

Open two terminals. In terminal 1:

```powershell
.\run_backend.ps1
```

In terminal 2:

```powershell
.\run_frontend.ps1
```

Open `http://localhost:5173`; API documentation is at `http://127.0.0.1:8000/docs`. The app uses the TF-IDF baseline if no DistilBERT model is configured.

## 5. Live news and fact-check sources

GDELT coverage search does not require an API key (subject to provider availability and use policy). Google Fact Check Tools is optional; configure `GOOGLE_FACTCHECK_API_KEY` in `.env` after enabling the API in Google Cloud. Never put an API key in a frontend file. Related coverage does not itself establish truth. No fact-check match means **unverified**, not fake or true.

## 6. Train DistilBERT via Colab GPU in VS Code

The Colab runtime is remote and cannot automatically read your D: drive. Copy the full new project and both CSVs into Google Drive at `MyDrive/FakeNewsResearchPlatform`. Open `notebooks\colab_distilbert_research.ipynb` in VS Code, select the Colab GPU kernel, and run cells in order. Model checkpoints are saved under the Drive project path, with research reports in `outputs\research`.

After training, copy `backend\models\research_distilbert_random` from Drive to the same path under `D:\FakeNewsResearchPlatform`. In `.env`, set:

```text
MODEL_DIR=backend/models/research_distilbert_random
```

Restart FastAPI and refresh `/api/health`. Do not omit `config.json`, tokenizer files, or model weights. Use the matching locked test split for every model/protocol comparison.

## Troubleshooting

- `No model is available`: train the TF-IDF baseline or copy a complete DistilBERT checkpoint into the configured `MODEL_DIR`.
- `No GPU visible`: select a Colab GPU runtime/kernel and rerun the notebook's GPU check. Availability is controlled by Colab.
- Google Fact Check API errors: leave the key blank until the API is configured; classification and GDELT search can still be tested.
- Reviewer token errors: use the generated `REVIEWER_TOKEN` from `.env` in the Expert Review page. Do not share it publicly.
