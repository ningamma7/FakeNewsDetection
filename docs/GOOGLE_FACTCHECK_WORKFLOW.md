# Google Fact Check Tools API workflow

This project uses the Google Fact Check Tools API to search indexed, third-party `ClaimReview` records that may be relevant to a submitted headline or claim. It does not replace the text classifier and does not automatically assign a true/false verdict to the complete article.

## 1. Configure the API

1. Open the Google Cloud Console and select or create a project.
2. Enable **Fact Check Tools API** for that project.
3. Create an API key. Restrict it to the Fact Check Tools API where the console supports that restriction.
4. In `D:\FakeNewsResearchPlatform\.env`, set `GOOGLE_FACTCHECK_API_KEY=YOUR_KEY`. Do not put the key in React code or commit `.env`.
5. Restart the backend from the project root in PowerShell:

   ```powershell
   .\run_backend.ps1
   ```

Official method: https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search

## 2. Test the API

1. Open `http://127.0.0.1:8000/docs`.
2. Call `GET /api/health`; `integrations.google_fact_check` should be `configured`.
3. Call `GET /api/fact-check/search` with a specific, concise public claim in `q`.
4. In the React app, open **Article analysis**, enter a title and article text, and put the concise claim/headline in **Evidence search query**. Click **Analyze article**.
5. The `POST /api/analyze` response includes the model prediction, `fact_checks`, `fact_check_review_count`, `evidence_status`, a human-readable summary, and separate GDELT related-news results.

## 3. Interpret results safely

- `published_fact_checks_found`: at least one published ClaimReview was returned. Read the review's claim wording, publisher, textual rating, date and linked article. A related claim may not be identical to the submitted article.
- `unverified_no_matching_fact_check`: no indexed ClaimReview matched the query. This is not evidence that the article is true or false.
- `google_api_key_required`: the backend key is not configured.
- `google_api_error`: the request failed; check the key, enabled API, quota, network and provider response.
- GDELT results are related news coverage, not factual verification.
- The app does not automatically override the model prediction with a Google result. An authorized human reviewer should examine the evidence and record a final label plus an evidence URL before a record is eligible for controlled training export.

## 4. Research audit

For each test query, keep a log with a non-sensitive case ID, submitted concise query, request date/time, Google API status, result count, review publisher, textual rating, review date, URL, and human-assessed relevance (`relevant`, `partially relevant`, `not relevant`, or `uncertain`). Report no-match and API-error rates separately. Do not convert missing results into labels, treat ratings as gold labels without adjudication, or claim this measures overall factual accuracy.
