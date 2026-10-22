import pytest
from backend.app import database


def test_feedback_approval_and_rejection(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "test.sqlite3")
    database.init_db()
    pending = database.create_feedback(
        "This is a sufficiently long example news article for testing purposes.",
        "Fake News", 0.83, "True News", "Please independently review this label."
    )
    assert pending["status"] == "pending"
    assert len(database.pending_feedback()) == 1
    approved = database.review_feedback(
        pending["id"], "approve", 1, "https://example.org/fact-check", "Checked against primary evidence.",
        "2017-02-01", "model-test"
    )
    assert approved["status"] == "approved"
    assert database.get_verified()[0]["label"] == 1
    assert database.research_stats()["verified_articles"] == 1
    with pytest.raises(ValueError):
        database.review_feedback(pending["id"], "approve", 1, "https://example.org", "duplicate review")

    second = database.create_feedback(
        "Another sufficiently long example article for rejection testing in this suite.",
        "True News", 0.61, None, None
    )
    rejected = database.review_feedback(second["id"], "reject", None, None, "No reliable supporting evidence.")
    assert rejected["status"] == "rejected"
    assert database.research_stats()["verified_articles"] == 1
