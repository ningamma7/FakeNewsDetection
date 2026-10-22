# FakeNews Detection Platform — complete local handover

A local React + FastAPI project for news-text classification, model explanation, published fact-check lookup, reviewer feedback, and a reviewer-approved article library.

## Included in this handover

- **User Analysis** page: article headline/text, Fake News vs True News scores, confidence, explanation tokens, published ClaimReview results, related news links, and feedback submission.
- **Expert Review** page: reviewer-token-protected pending queue; approve or reject submissions; approval requires a final label and an HTTP(S) evidence URL.
- **Verified Articles** page: only reviewer-approved articles appear, with evidence links and reviewer notes.
- **SQLite database** at `backend/data/verilens.sqlite3`: prediction history, fact-check lookup records, feedback, approved articles, and reviewer audit events.
- **Trained TF-IDF + Logistic Regression baseline** at `backend/models/tfidf_random.joblib` so the local application can classify articles immediately after setup.
- **DistilBERT training code and Colab notebook** at `notebooks/Colab_Train_DistilBERT.ipynb`. Train it on a Colab GPU, download its model ZIP, and place the folder at `backend/models/research_distilbert_random/`.
- Dataset preparation, reproducible random/temporal splits, evaluation utilities, test suite, and raw ISOT dataset files.

## Important model status

The included TF-IDF baseline was trained on the cleaned ISOT dataset and evaluated on the held-out random test split. The recorded test results are in `outputs/research/evaluation_tfidf_random.json`. Accuracy on this benchmark is **99.00%**, macro-F1 **98.995%**, and MCC **0.9799**. These results can be optimistic for real-world news because benchmark datasets have source, time-period, and writing-style biases; they do not mean that the platform can independently verify arbitrary claims.

The DistilBERT notebook is prepared but **the trained DistilBERT checkpoint is not bundled** because it must be trained in a real Colab GPU runtime. Until you run the notebook and copy its downloaded model folder into the specified location, the backend automatically uses the included TF-IDF baseline. The explanation UI then displays TF-IDF feature weights; after DistilBERT is installed it displays Captum Saliency when available.

## Windows requirements

- Windows 10/11
- Python 3.12 (the `py -3.12` launcher should work)
- Node.js 20+
- Internet connection for installing Python/npm packages and for optional live evidence lookups

## 1. Set up the project

Open PowerShell in the folder containing `setup_windows.ps1`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

This creates `.venv`, installs backend dependencies, installs frontend packages, and creates a private `.env` with a generated reviewer token if `.env` does not already exist. Do not share `.env` or commit it to a public repository.

If setup has already been run, verify the project-root `.env` contains these settings:

```dotenv
DATABASE_PATH=backend/data/verilens.sqlite3
MODEL_DIR=backend/models/research_distilbert_random
BASELINE_MODEL_PATH=backend/models/tfidf_random.joblib
REVIEWER_TOKEN=your-private-generated-token
GOOGLE_FACTCHECK_API_KEY=
```

`MODEL_DIR` can point to a not-yet-existing directory: the API will fall back to the baseline. Keep the `REVIEWER_TOKEN` private. For Google Fact Check results, create a key in Google Cloud, enable the Fact Check Tools API, and put the key in `.env`. Without a key, the app clearly displays that Google Fact Check lookup is not configured. GDELT related-news lookup is separate and can fail when internet access is unavailable.

## 2. Start the application

Leave the terminals to you, as requested.

**Backend — first PowerShell terminal:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_backend.ps1
```

**Frontend — second PowerShell terminal:**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_frontend.ps1
```

Open the Vite URL printed by the frontend, normally `http://localhost:5173`. API health is `http://127.0.0.1:8000/api/health`; API docs are at `http://127.0.0.1:8000/docs`.

## 3. Train DistilBERT in Colab and install it locally

1. Open `notebooks/Colab_Train_DistilBERT.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → GPU**.
3. Upload this complete project ZIP when the first notebook cell asks for it.
4. Run the notebook cells in order. It prepares the dataset, trains DistilBERT, evaluates on the matching held-out test split, and downloads `research_distilbert_random.zip` plus an evaluation JSON.
5. Extract the model ZIP into the project’s `backend/models/` folder. The resulting path must be `backend/models/research_distilbert_random/config.json`.
6. Keep `MODEL_DIR=backend/models/research_distilbert_random` in `.env` and restart the backend.
7. Check `/api/health` to confirm the loaded model type is `transformer`. When it is loaded, the UI should use **Captum Saliency** for the “Why this result?” section.

Do not report DistilBERT metrics until the notebook actually finishes. Record the actual Colab runtime, GPU, training report, and evaluation JSON for your project report.

## 4. Reviewer workflow

1. A user analyzes an article and can submit feedback.
2. Open **Expert Review** and enter the private `REVIEWER_TOKEN` value from `.env`.
3. Inspect the article and user comment, choose a final label, provide a publicly accessible evidence URL, and add notes.
4. Approve & publish or reject. Approvals require an evidence URL and final label.
5. Only approved articles appear in **Verified Articles**. Approved feedback is not automatically used to retrain the model; any future training-data merge is a deliberate developer action.

## Database

SQLite is created automatically on first backend start. Tables include `predictions`, `fact_check_logs`, `feedback`, `verified_articles`, and `audit_logs`. Back up `backend/data/verilens.sqlite3` before making schema changes or deleting project data.

## Tests

From the project root, with the virtual environment active:

```powershell
.\.venv\Scripts\python.exe -m pytest .\backend\tests -v
```

The current backend test suite passes in the build environment. The frontend production build could not be verified in this packaging environment because npm package installation timed out; run `npm install` and `npm run build` in `frontend` on your Windows machine if you want a separate production-build check.

## Interpretation and limitations

- A classifier label/confidence is not a factual verdict. Dataset leakage and publisher/style correlations can inflate benchmark scores.
- A Google fact-check match is a published third-party review; read its context and date. No match means only that no indexed match was returned.
- Captum Saliency indicates model sensitivity, not proof that a token or article is true/false.
- Reviewer access uses a shared token suitable for local demonstration; for production, replace it with proper user authentication, roles, HTTPS, rate limiting, and secure secret management.
