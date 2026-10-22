from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()

APP_NAME = os.getenv("APP_NAME", "VeriLens Research Platform")
DATABASE_PATH = project_path(os.getenv("DATABASE_PATH", "backend/data/verilens.sqlite3"))
MODEL_DIR = project_path(os.getenv("MODEL_DIR", "backend/models/research_distilbert_random"))
BASELINE_MODEL_PATH = project_path(os.getenv("BASELINE_MODEL_PATH", "backend/models/tfidf_random.joblib"))
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()
REVIEWER_TOKEN = os.getenv("REVIEWER_TOKEN", "").strip()
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.65"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
