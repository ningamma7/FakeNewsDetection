from fastapi.testclient import TestClient
from backend.app import database


def test_api_root_feedback_and_review(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "api.sqlite3")
    # Import is delayed so the app's startup database initialization uses the temporary path.
    from backend.app import main
    monkeypatch.setattr(main, "REVIEWER_TOKEN", "test-review-token")
    client = TestClient(main.app)

    assert client.get("/").status_code == 200
    assert client.get("/api/health").status_code == 200

    submitted = client.post("/api/feedback", json={
        "article": "A sufficiently long test article for checking the feedback review API.",
        "prediction": "Fake News",
        "confidence": 0.72,
        "suggested_label": "True News",
        "comment": "Please investigate this test record.",
    })
    assert submitted.status_code == 200
    feedback_id = submitted.json()["feedback_id"]

    pending = client.get("/api/review/pending", headers={"X-Reviewer-Token": "test-review-token"})
    assert pending.status_code == 200
    assert pending.json()["items"][0]["id"] == feedback_id

    reviewed = client.post(f"/api/review/{feedback_id}", headers={"X-Reviewer-Token": "test-review-token"}, json={
        "action": "approve",
        "final_label": "True News",
        "evidence_url": "https://example.org/fact-check",
        "notes": "Test approval only; not a factual claim.",
        "publication_date": "2017-01-01",
    })
    assert reviewed.status_code == 200
    assert reviewed.json()["item"]["status"] == "approved"
    assert client.get("/api/verified").json()["items"][0]["label"] == 1


def test_analyze_returns_google_claim_reviews_separately(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "analyze.sqlite3")
    from backend.app import main
    monkeypatch.setattr(main.model_service, "predict", lambda text: {
        "prediction": "Fake News", "prediction_id": 0, "confidence": 0.91,
        "probabilities": {"Fake News": 0.91, "True News": 0.09},
    })
    monkeypatch.setattr(main, "search_live_news", lambda query: {"provider": "GDELT DOC 2.0", "results": []})
    monkeypatch.setattr(main, "search_fact_checks", lambda query: {
        "provider": "Google Fact Check Tools", "status": "matches_found",
        "results": [{"claim": "Example claim", "reviews": [{"rating": "False", "url": "https://example.org/review"}]}],
        "review_count": 1,
    })
    client = TestClient(main.app)
    response = client.post("/api/analyze", json={
        "title": "Example headline",
        "article": "A sufficiently long article text for testing the analyze endpoint safely.",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["prediction"] == "Fake News"
    assert data["evidence_status"] == "published_fact_checks_found"
    assert data["fact_check_review_count"] == 1
    assert data["fact_checks"]["results"][0]["reviews"][0]["rating"] == "False"


def test_build_evidence_query_prefers_concise_claim():
    from backend.app.main import AnalyzeRequest, build_evidence_query
    payload = AnalyzeRequest(
        title="Headline",
        article="The full article begins here. More details follow in a second sentence.",
        search_query="Concise claim to search",
    )
    assert build_evidence_query(payload) == "Concise claim to search"

def test_build_evidence_query_uses_first_sentence_when_title_absent():
    from backend.app.main import AnalyzeRequest, build_evidence_query
    payload = AnalyzeRequest(
        article="An opening statement worth checking. The remainder contains other details."
    )
    assert build_evidence_query(payload) == "An opening statement worth checking."
