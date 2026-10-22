# Keys and endpoints

## Google Fact Check Tools API
The backend calls `GET https://factchecktools.googleapis.com/v1alpha1/claims:search` using a short query and an API key from server-side `.env`. Enable the Fact Check Tools API in Google Cloud, create an API key and restrict it to this API where possible. Store it only as `GOOGLE_FACTCHECK_API_KEY` in the project-root `.env`. API responses may contain claim text, ClaimReview title, publisher, textual rating, review date and source URL. These are published third-party fact checks, not an automatic authoritative verdict for every news article. The backend keeps these reviews separate from the DistilBERT/TF-IDF prediction and does not automatically override the model label.

## GDELT DOC 2.0 search
The backend calls the GDELT DOC 2.0 article list endpoint and normalizes headline, URL, publisher/domain and publication date. Treat these results only as related news coverage. Search availability, indexed sources and quotas/usage policies are controlled by the provider.

## Reviewer token
`REVIEWER_TOKEN` is an application-level gate for the local research prototype, not full production authentication. Use a long random value. Do not expose it to ordinary users or commit `.env`. For a public deployment, replace this with proper user authentication, role-based authorization, CSRF controls where relevant, rate limiting, HTTPS, secure secret storage and audit logs.

## Local API
- `GET /api/health`
- `POST /api/predict`
- `POST /api/analyze` (returns model prediction plus Google ClaimReview search and GDELT context)
- `GET /api/news/search?q=...`
- `GET /api/fact-check/search?q=...`
- `POST /api/feedback`
- `GET /api/review/pending` (reviewer token required)
- `POST /api/review/{feedback_id}` (reviewer token required)
- `GET /api/verified`
- `GET /api/research/stats` (reviewer token required)
- `POST /api/research/export-approved` (reviewer token required)

## Quick verification test
1. Put your API key in `.env`; restart FastAPI.
2. Confirm `GET /api/health` reports `google_fact_check: configured`.
3. Query `GET /api/fact-check/search?q=...` with a specific public claim.
4. Submit the headline, article text and a concise Evidence search query through Article Analysis (or call `POST /api/analyze`).
5. Inspect the publisher, textual rating, review date and URL for each returned ClaimReview. A missing match means no indexed review was found; it is not a truth/falsity verdict.
