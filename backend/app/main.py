from __future__ import annotations
import csv
import io
import re
from typing import Literal
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .config import APP_NAME, FRONTEND_ORIGIN, REVIEWER_TOKEN, GOOGLE_FACTCHECK_API_KEY
from .database import init_db, create_feedback, pending_feedback, review_feedback, get_verified, research_stats, connect, log_prediction, log_fact_check, list_prediction_history
from .evidence import search_live_news, search_fact_checks
from .model_service import ModelService

app = FastAPI(title=APP_NAME, version="0.1.0", description="Fake news article prediction platform with token-level explanations and optional human feedback.")
app.add_middleware(CORSMiddleware, allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()
model_service = ModelService()


class ArticleRequest(BaseModel):
    article: str = Field(min_length=20, max_length=100000)
    title: str | None = Field(default=None, max_length=500)

    @property
    def full_text(self) -> str:
        title = (self.title or "").strip()
        body = self.article.strip()
        return f"{title}\n\n{body}" if title and title.lower() not in body[:len(title) + 5].lower() else body


class AnalyzeRequest(ArticleRequest):
    search_query: str | None = Field(default=None, max_length=500)


class FeedbackRequest(BaseModel):
    article: str = Field(min_length=20, max_length=100000)
    prediction: str = Field(pattern=r"^(Fake News|True News)$")
    confidence: float = Field(ge=0, le=1)
    suggested_label: str | None = Field(default=None, pattern=r"^(Fake News|True News)$")
    comment: str | None = Field(default=None, max_length=4000)


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    final_label: Literal["Fake News", "True News"] | None = None
    evidence_url: str | None = Field(default=None, max_length=2048)
    notes: str | None = Field(default=None, max_length=4000)
    publication_date: str | None = Field(default=None, max_length=60)


def require_reviewer(token: str | None) -> None:
    if not REVIEWER_TOKEN or REVIEWER_TOKEN == "change-this-to-a-long-random-secret":
        raise HTTPException(status_code=503, detail="Set a private REVIEWER_TOKEN in the backend .env before using review endpoints.")
    if token != REVIEWER_TOKEN:
        raise HTTPException(status_code=403, detail="Reviewer token is missing or invalid.")


def get_prediction_text(payload: ArticleRequest) -> str:
    return payload.full_text


def build_evidence_query(payload: AnalyzeRequest) -> str:
    """Prefer a concise user query/headline; otherwise use the first sentence."""
    explicit = (payload.search_query or '').strip()
    if explicit:
        return re.sub(r'\s+', ' ', explicit)[:220]
    title = (payload.title or '').strip()
    if title:
        return re.sub(r'\s+', ' ', title)[:220]
    body = re.sub(r'\s+', ' ', payload.article).strip()
    first_sentence = re.split(r'(?<=[.!?])\s+', body, maxsplit=1)[0]
    return first_sentence[:220] or body[:220]


@app.get("/")
def root():
    return {"name": APP_NAME, "docs": "/docs", "health": "/api/health", "warning": "Classifier output is not an authoritative fact-check."}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": model_service.status(), "integrations": {
        "gdelt_news_search": "configured",
        "google_fact_check": "configured" if GOOGLE_FACTCHECK_API_KEY else "api_key_required",
    }}


@app.post("/api/predict")
def predict(payload: ArticleRequest):
    try:
        return model_service.predict(get_prediction_text(payload))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    """Classify the article and return external fact-check/news matches separately."""
    try:
        full_text = get_prediction_text(payload)
        prediction = model_service.predict(full_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    query = build_evidence_query(payload)
    fact_checks = search_fact_checks(query)
    live_news = search_live_news(query)
    record_id = log_prediction(full_text, prediction)
    log_fact_check(record_id, query, "Google Fact Check Tools", fact_checks)
    review_count = int(fact_checks.get("review_count", 0) or 0)
    evidence_status = "published_fact_checks_found" if review_count > 0 else (
        "api_key_required" if fact_checks.get("status") == "api_key_not_configured" else
        "provider_error" if fact_checks.get("status") == "provider_error" else "no_matching_fact_checks_found")
    return {
        "analysis_record_id": record_id,
        "prediction": prediction,
        "evidence_query": query,
        "evidence_status": evidence_status,
        "fact_check_review_count": review_count,
        "fact_checks": fact_checks,
        "related_news": live_news,
        "disclaimer": "Classifier predictions and external ClaimReview results are separate. No matching fact-check is not evidence that a claim is true.",
    }


@app.get("/api/news/search")
def news_search(q: str = Query(min_length=2, max_length=500), limit: int = Query(default=8, ge=1, le=25)):
    return search_live_news(q, limit)


@app.get("/api/fact-check/search")
def fact_check_search(q: str = Query(min_length=2, max_length=500), limit: int = Query(default=8, ge=1, le=20)):
    return search_fact_checks(q, limit)


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackRequest):
    row = create_feedback(payload.article, payload.prediction, payload.confidence, payload.suggested_label, payload.comment)
    return {"message": "Feedback submitted for expert review.", "feedback_id": row["id"], "status": row["status"]}


@app.get("/api/review/pending")
def list_pending(x_reviewer_token: str | None = Header(default=None)):
    require_reviewer(x_reviewer_token)
    return {"items": pending_feedback()}


@app.post("/api/review/{feedback_id}")
def apply_review(feedback_id: int, payload: ReviewRequest, x_reviewer_token: str | None = Header(default=None)):
    require_reviewer(x_reviewer_token)
    final_label = None if payload.final_label is None else (0 if payload.final_label == "Fake News" else 1)
    try:
        row = review_feedback(feedback_id, payload.action, final_label, payload.evidence_url, payload.notes,
                              payload.publication_date, model_service.model_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback item not found.")
    return {"message": "Review saved.", "item": row}


@app.get("/api/verified")
def verified_articles():
    return {"items": get_verified()}


@app.get("/api/history")
def prediction_history(limit: int = Query(default=25, ge=1, le=100)):
    return {"items": list_prediction_history(limit)}


@app.get("/api/research/stats")
def get_research_stats(x_reviewer_token: str | None = Header(default=None)):
    require_reviewer(x_reviewer_token)
    return {"model": model_service.status(), **research_stats()}


@app.post("/api/model/reload")
def reload_model(x_reviewer_token: str | None = Header(default=None)):
    require_reviewer(x_reviewer_token)
    return {"model": model_service.reload()}


@app.post("/api/research/export-approved")
def export_approved(x_reviewer_token: str | None = Header(default=None)):
    require_reviewer(x_reviewer_token)
    with connect() as con:
        rows = con.execute("SELECT article, label, evidence_url, reviewer_notes, publication_date, label_available_at, article_hash, model_version FROM verified_articles ORDER BY created_at ASC").fetchall()
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["text", "label", "evidence_url", "reviewer_notes", "publication_date", "label_available_at", "article_hash", "model_version"])
    writer.writerows([tuple(r) for r in rows])
    stream.seek(0)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=approved_feedback.csv"})
