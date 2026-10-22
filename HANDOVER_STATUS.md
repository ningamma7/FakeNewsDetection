# Handover verification status

Verified in the build environment on 2026-10-20:

- TF-IDF + Logistic Regression model artifact created: `backend/models/tfidf_random.joblib`.
- Held-out random test evaluation saved: `outputs/research/evaluation_tfidf_random.json`.
- Benchmark test set: 7,820 rows.
- Accuracy: 0.9900; macro-F1: 0.9900; MCC: 0.9799; ROC-AUC for true class: 0.9993.
- Backend tests: 12 passed.
- Model service smoke test: loaded the included baseline and returned a prediction with TF-IDF feature explanation.
- Colab DistilBERT training notebook: JSON structure validated and included at `notebooks/Colab_Train_DistilBERT.ipynb`.

Not completed in this environment:

- A DistilBERT checkpoint was not trained because that step requires a real GPU-enabled Colab session; the notebook performs that training after the user uploads the project ZIP and runs the cells.
- A production frontend build was not verified because npm package installation timed out in the build environment. Run `npm install` and `npm run build` locally in `frontend` if you need that check.
- Google Fact Check Tools results require the user's own Google Cloud API key; none is embedded in this package.

Benchmark metrics are not a guarantee of performance on unseen real-world news. Prediction scores are not independent fact verification.
