from __future__ import annotations
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .config import DATABASE_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATABASE_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article TEXT NOT NULL,
            article_hash TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            user_suggested_label TEXT,
            user_comment TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            reviewer_action TEXT,
            final_label TEXT,
            evidence_url TEXT,
            review_notes TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS verified_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER UNIQUE NOT NULL REFERENCES feedback(id),
            article TEXT NOT NULL,
            article_hash TEXT NOT NULL,
            label INTEGER NOT NULL CHECK(label IN (0, 1)),
            evidence_url TEXT NOT NULL,
            reviewer_notes TEXT,
            publication_date TEXT,
            label_available_at TEXT NOT NULL,
            model_version TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article TEXT NOT NULL,
            article_hash TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_type TEXT,
            explanation_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_check_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_record_id INTEGER REFERENCES predictions(id),
            query_text TEXT NOT NULL,
            provider TEXT NOT NULL,
            status TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id INTEGER,
            details_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);
        CREATE INDEX IF NOT EXISTS idx_verified_hash ON verified_articles(article_hash);
        CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
        """)


def log_prediction(article: str, prediction: dict[str, Any]) -> int:
    init_db()
    created_at = utc_now()
    with connect() as con:
        cur = con.execute(
            "INSERT INTO predictions(article, article_hash, prediction_label, confidence, model_type, explanation_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (article, text_hash(article), str(prediction.get("prediction", "Unknown")),
             float(prediction.get("confidence", 0.0)), str(prediction.get("model_type", "unknown")),
             json.dumps(prediction.get("explanation", {}), ensure_ascii=False), created_at),
        )
        return int(cur.lastrowid)


def log_fact_check(prediction_record_id: int, query_text: str, provider: str, result: dict[str, Any]) -> None:
    init_db()
    with connect() as con:
        con.execute(
            "INSERT INTO fact_check_logs(prediction_record_id, query_text, provider, status, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (prediction_record_id, query_text, provider, str(result.get("status", "unknown")),
             json.dumps(result, ensure_ascii=False), utc_now()),
        )


def list_prediction_history(limit: int = 25) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT id, article, prediction_label, confidence, model_type, created_at FROM predictions ORDER BY id DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
        return [dict(row) for row in rows]


def create_feedback(article: str, prediction_label: str, confidence: float,
                    suggested_label: str | None, comment: str | None) -> dict[str, Any]:
    created_at = utc_now()
    with connect() as con:
        cur = con.execute(
            "INSERT INTO feedback(article, article_hash, prediction_label, confidence, user_suggested_label, user_comment, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (article, text_hash(article), prediction_label, confidence, suggested_label, comment, created_at),
        )
        row = con.execute("SELECT * FROM feedback WHERE id=?", (cur.lastrowid,)).fetchone()
        return dict(row)


def pending_feedback() -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM feedback WHERE status='pending' ORDER BY created_at ASC").fetchall()
        return [dict(r) for r in rows]


def review_feedback(feedback_id: int, action: str, final_label: int | None,
                    evidence_url: str | None, notes: str | None,
                    publication_date: str | None = None, model_version: str | None = None) -> dict[str, Any] | None:
    reviewed_at = utc_now()
    with connect() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        if row is None:
            return None
        if row["status"] != "pending":
            raise ValueError(f"Feedback item {feedback_id} has already been reviewed.")
        if action == "approve":
            if final_label not in (0, 1):
                raise ValueError("Approving an article requires final_label 0 (Fake) or 1 (True).")
            if not evidence_url or not evidence_url.startswith(("https://", "http://")):
                raise ValueError("Approving an article requires a publicly accessible evidence URL.")
            con.execute(
                "INSERT INTO verified_articles(feedback_id, article, article_hash, label, evidence_url, reviewer_notes, publication_date, label_available_at, model_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (feedback_id, row["article"], row["article_hash"], final_label, evidence_url, notes, publication_date, reviewed_at, model_version, reviewed_at),
            )
            status = "approved"
        else:
            status = "rejected"
            final_label = None
            evidence_url = evidence_url or None
        con.execute(
            "UPDATE feedback SET status=?, reviewer_action=?, final_label=?, evidence_url=?, review_notes=?, reviewed_at=? WHERE id=?",
            (status, action, final_label, evidence_url, notes, reviewed_at, feedback_id),
        )
        con.execute(
            "INSERT INTO audit_logs(actor, action, entity_type, entity_id, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("reviewer", action, "feedback", feedback_id, json.dumps({"final_label": final_label, "evidence_url": evidence_url, "notes": notes}, ensure_ascii=False), reviewed_at),
        )
        updated = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        return dict(updated)


def get_verified() -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM verified_articles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def research_stats() -> dict[str, Any]:
    with connect() as con:
        feedback_rows = con.execute("SELECT status, COUNT(*) n FROM feedback GROUP BY status").fetchall()
        verified = con.execute("SELECT COUNT(*) n FROM verified_articles").fetchone()["n"]
        return {"feedback_by_status": {r["status"]: r["n"] for r in feedback_rows}, "verified_articles": verified}
