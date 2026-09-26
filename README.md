---
# 📰 Fake News Detection -

> Final Year Project - End-to-end fake news detection with AI explainability, expert review, and verified articles.

🔗 **Repo:** https://github.com/ningamma7/FakeNewsDetection

### 🚀 Live Features
- **ML Detection:** Detects FAKE / REAL news with confidence score
- **Explainability:** `explainability.py` shows WHY it's fake (LIME keywords)
- **Expert Review:** `ExpertReview.jsx` panel for human verification
- **Verified Articles:** `VerifiedArticles.jsx` shows trusted news database
- **FastAPI Docs:** Auto-generated Swagger at `/docs`

### 🛠️ Tech Stack
**Backend:** Python, FastAPI, Scikit-learn, TF-IDF, LIME/SHAP
**Frontend:** React.js, Vite, JavaScript, Tailwind CSS
**File Structure from your repo:**
FakeNewsDetection/
├── backend/
│   ├── app/
│   │   ├── api/               # FastAPI routes (predict, review)
│   │   ├── models/            # ML model loading
│   │   ├── services/          # Prediction logic
│   │   ├── explainability.py  # LIME/SHAP explainability
│   │   └── main.py            # FastAPI entry point
│   └── training/              # (local only) training scripts
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── ExpertReview.jsx
│   │   ├── VerifiedArticles.jsx
│   │   ├── storage.js
│   │   └── assets/
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── requirements.txt
└── .gitignore                 # Ignores venv/, node_modules/, processed/

### ⚙️ How to Run Locally

**1. Clone**
```bash
git clone https://github.com/ningamma7/FakeNewsDetection.git
cd FakeNewsDetection
*2. Backend (Port 8000)*
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r ../requirements.txt
uvicorn app.main:app --reload --port 8000
Backend: http://localhost:8000
API Docs: http://localhost:8000/docs

*3. Frontend (Port 5173)*
cd ../frontend
npm install
npm run dev
Frontend: http://localhost:5173

### 🔍 API Endpoints
Method | Endpoint | Description
POST | `/predict` | Predict fake/real news
GET | `/verified` | Get verified articles
POST | `/expert-review` | Submit expert review
### 🧠 Model Workflow
1. User pastes news in React UI (`App.jsx`)
2. API call to FastAPI `/predict`
3. Model vectorizes text (TF-IDF)
4. Prediction + Confidence + Explainability keywords from `explainability.py`
5. Result saved via `storage.js`

### 🙈 Note on .gitignore
Your repo correctly ignores heavy folders:
`backend/processed/`, `notebooks/`, `venv/`, `node_modules/`, `*.sqlite3` - keeps GitHub clean and fast.

### 📈 Future Improvements
- Deploy backend on Render, frontend on Vercel
- Upgrade to BERT model
- Add authentication for expert reviewers

### 👨‍💻  Final Year Project 2025-26
Team Members:
*Pragnya*-Team leader
*Harshitha P*
*Ningamma Mariyajjanavara*


---

After you paste, click **Commit changes** > **Commit directly to main**.

Send me screenshot once done - I will check!
