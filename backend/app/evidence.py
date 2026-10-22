from __future__ import annotations
from typing import Any
import re
import requests
from .config import GOOGLE_FACTCHECK_API_KEY

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _clean_query(query: str, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", (query or "")).strip()[:limit]


def search_live_news(query: str, limit: int = 8) -> dict[str, Any]:
    q = _clean_query(query)
    if not q:
        return {"provider": "GDELT DOC 2.0", "results": [], "error": "Enter a search query."}
    params = {"query": q, "mode": "ArtList", "format": "json", "sort": "DateDesc", "maxrecords": max(1, min(limit, 25))}
    try:
        response = requests.get(GDELT_URL, params=params, timeout=15, headers={"User-Agent": "VeriLensResearch/0.1"})
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("articles", []) or []:
            url = item.get("url") or item.get("url_mobile")
            if not url:
                continue
            results.append({
                "title": item.get("title") or "Untitled article",
                "url": url,
                "domain": item.get("domain") or "",
                "published_at": item.get("seendate") or item.get("date") or None,
                "language": item.get("language") or None,
                "source": "GDELT DOC 2.0",
            })
        return {"provider": "GDELT DOC 2.0", "query": q, "results": results, "note": "Related news coverage only; results do not establish whether a claim is true."}
    except (requests.RequestException, ValueError) as exc:
        return {"provider": "GDELT DOC 2.0", "query": q, "results": [], "error": f"Live news search failed: {exc}"}


def search_fact_checks(query: str, limit: int = 8) -> dict[str, Any]:
    q = _clean_query(query)
    if not q:
        return {"provider": "Google Fact Check Tools", "results": [], "status": "query_required"}
    if not GOOGLE_FACTCHECK_API_KEY:
        return {"provider": "Google Fact Check Tools", "query": q, "results": [], "status": "api_key_not_configured", "note": "Set GOOGLE_FACTCHECK_API_KEY in the backend .env to search indexed ClaimReview fact checks."}
    try:
        response = requests.get(FACTCHECK_URL, params={"query": q, "key": GOOGLE_FACTCHECK_API_KEY, "pageSize": max(1, min(limit, 20))}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        results = []
        for claim in payload.get("claims", []) or []:
            reviews = claim.get("claimReview", []) or []
            if not reviews:
                results.append({"claim": claim.get("text", ""), "claimant": claim.get("claimant", ""), "reviews": []})
                continue
            results.append({
                "claim": claim.get("text", ""),
                "claimant": claim.get("claimant", ""),
                "claim_date": claim.get("claimDate"),
                "reviews": [{
                    "publisher": r.get("publisher", {}).get("name", ""),
                    "publisher_site": r.get("publisher", {}).get("site", ""),
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "rating": r.get("textualRating", ""),
                    "review_date": r.get("reviewDate"),
                } for r in reviews],
            })
        review_count = sum(len(item.get("reviews", [])) for item in results)
        return {"provider": "Google Fact Check Tools", "query": q, "results": results, "status": "matches_found" if review_count else "no_matches", "review_count": review_count, "note": "These are indexed third-party claim reviews. Check the review context and date; they do not automatically determine whether the submitted article is true or false."}
    except (requests.RequestException, ValueError) as exc:
        return {"provider": "Google Fact Check Tools", "query": q, "results": [], "status": "provider_error", "error": f"Fact-check search failed: {exc}"}
