# Research protocol: VeriLens

## Research question
Does combining a text classifier with uncertainty-aware presentation, token-level explanations, live evidence retrieval and expert-reviewed feedback improve the reliability and auditability of news assessment when evaluated beyond a single random split?

## Core comparisons
1. TF-IDF + Logistic Regression versus DistilBERT.
2. Stratified random split versus chronological split.
3. Original model versus a separate model retrained with expert-approved records that are eligible at the historical training cutoff.
4. Raw confidence versus calibrated confidence (calibration is a separate experiment; do not call softmax values calibrated by default).
5. Captum Saliency versus Integrated Gradients if implemented and tested; assess explanation faithfulness with perturbation tests rather than treating saliency as proof of truth.
6. Google Fact Check API evidence coverage and relevance: record how often a query returns published ClaimReview items and have a reviewer assess whether each result actually addresses the submitted claim.

## Dataset protocol
- Input source dataset files are `data/raw/Fake.csv` and `data/raw/True.csv`.
- Labels: 0 = Fake, 1 = True. Document this mapping throughout.
- Model input text combines the title and body; subject and publication date are metadata, not text features.
- Exact normalized duplicate text is removed before splitting. Rows with identical text but conflicting source labels are excluded and counted in the report. Near-duplicates still require a separate audit.
- Random split proportions are 70% train, 10% validation, 20% test, stratified by label with fixed seed.
- Temporal split uses dated records sorted by date and assigns the earliest 70% to train, next 10% to validation, latest 20% to test. Undated rows are excluded from the temporal protocol and counted. This can cause class-prevalence drift; report class counts.
- Never select parameters from the test split. Save split hashes, package versions, model ID/revision and random seeds.

## Metrics
Classification: accuracy, precision, recall, macro-F1, MCC, ROC-AUC (when both classes are present), confusion matrix, class support.
Confidence: expected calibration error (ECE) and Brier score are descriptive unless calibrated probability semantics have been carefully checked. Fit calibration parameters on validation only.
Google Fact Check API evidence workflow: number of searches, API configured/unconfigured/error status, percentage with at least one ClaimReview result, number of returned reviews, result age, review publisher, textual rating, and human-judged relevance. Do not score a no-match as true or false and do not count raw ratings as ground-truth labels without adjudication.
Human review: number approved/rejected/pending, time-to-review if instrumented, label agreement when multiple reviewers exist, and performance change on an unchanged locked test set.
Explanation: attribution ranking, token masking/perturbation, sensitivity, and runtime. Captum saliency is a model-attribution technique, not factual justification.

## Leakage controls
- Split before training; keep a final test set untouched.
- Approved feedback enters only training data, never validation/test.
- Exclude approved text overlapping with test articles.
- For temporal retraining, only use labels available before the simulated training cutoff, not just articles published before it.
- Do not use model-generated pseudo-labels as verified ground truth.
- Store model/data version IDs and experiment settings in output JSON.

## Limitations to state
ISOT is a historical dataset with source/domain and time biases; a random split can overestimate deployment generalization. Do not claim broad real-time generalization from ISOT scores alone. GDELT coverage is related-news context, not fact-check evidence by itself. Google Fact Check Tools returns indexed published claim reviews only and may return no matches. A review may concern a related but non-identical claim and requires contextual inspection. Classification confidence is not a calibrated probability without validation/calibration. Human review quality depends on reviewer expertise and evidence quality. Real-time current-news claims require independent, timestamped source evidence.
