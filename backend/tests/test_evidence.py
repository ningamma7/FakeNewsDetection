from types import SimpleNamespace
from backend.app import evidence


def test_live_news_result_mapping(monkeypatch):
    payload = {"articles": [{"title": "Example current report", "url": "https://example.org/news", "domain": "example.org", "seendate": "20261019T010000Z", "language": "English"}]}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(evidence.requests, "get", lambda *args, **kwargs: response)
    result = evidence.search_live_news("example current report")
    assert result["provider"] == "GDELT DOC 2.0"
    assert result["results"][0]["url"] == "https://example.org/news"
    assert "do not establish" in result["note"]


def test_factcheck_no_key_is_explicit(monkeypatch):
    monkeypatch.setattr(evidence, "GOOGLE_FACTCHECK_API_KEY", "")
    result = evidence.search_fact_checks("some public claim")
    assert result["status"] == "api_key_not_configured"
    assert result["results"] == []


def test_factcheck_results_include_review_count(monkeypatch):
    monkeypatch.setattr(evidence, "GOOGLE_FACTCHECK_API_KEY", "test-key")
    payload = {"claims": [{
        "text": "Test claim",
        "claimant": "Example",
        "claimReview": [{
            "publisher": {"name": "Example Fact Check", "site": "example.org"},
            "title": "Claim review",
            "url": "https://example.org/review",
            "textualRating": "False",
            "reviewDate": "2026-01-01",
        }],
    }]}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(evidence.requests, "get", lambda *args, **kwargs: response)
    result = evidence.search_fact_checks("specific example claim")
    assert result["status"] == "matches_found"
    assert result["review_count"] == 1
    assert result["results"][0]["reviews"][0]["rating"] == "False"
