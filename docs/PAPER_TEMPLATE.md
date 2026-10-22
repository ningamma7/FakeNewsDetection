# Research paper drafting template (fill with measured results only)

## Proposed title
Auditing Reliable Fake-News Text Classification Through Temporal Evaluation, Explainability, Calibration, and Expert-Verified Feedback

## Abstract (fill after experiments)
Fake-news classification models can learn dataset-specific signals and should not be treated as factual verification systems. This study evaluates a TF-IDF + Logistic Regression baseline and DistilBERT using duplicate-aware random and chronological ISOT splits, tests confidence calibration, studies Captum token attributions, and incorporates a Google Fact Check Tools API lookup for published ClaimReview evidence alongside live-news context. The models are evaluated with [metrics] on [exact split sizes]. The evidence workflow is described using [query count, review coverage, relevance audit, and limitations]. Results show [measured result with uncertainty interval]. The study is limited by [dataset, annotation, API coverage, and reviewer limitations].

## Research questions
- RQ1: How does DistilBERT compare with TF-IDF + Logistic Regression on random and chronological hold-out protocols?
- RQ2: How much do scores and confidence reliability vary across random and chronological evaluations?
- RQ3: Does validation-fitted temperature scaling improve calibration on the untouched test set?
- RQ4: Do explanations react predictably when high-attribution tokens are perturbed?
- RQ5: Does adding eligible human-verified examples improve results on an unchanged locked test set?
- RQ6: How often does Google Fact Check Tools return a relevant published claim review for analysed articles, and how does human review assess those results against the model's classification?

## Methods
Describe dataset provenance/license; label mapping (`0=Fake`, `1=True`); exact-deduplication and conflicting-label policy; split seed/proportions/date boundaries; architecture, pretrained checkpoint, tokenizer and maximum length; optimizer, learning rate, batch size, epochs and checkpoint rule; hardware and package versions; calibration on validation only; approved-data eligibility; and metrics.

## Required result tables
1. Dataset size/class counts/date range for every split.
2. Model comparison on random and chronological tests (accuracy, per-class precision/recall, macro-F1, MCC, ROC-AUC, confusion matrix).
3. Calibration comparison (NLL, Brier score, ECE and reliability plots) before/after; temperature must be fitted on validation only.
4. Google Fact Check API evidence-coverage table: number of searches, matches, no-matches, API errors, review publishers/dates/ratings, and human-judged relevance.
5. Expert-feedback ablation with approved rows, exclusions, training rows, test results and uncertainty intervals.
6. Explanation faithfulness and runtime results if explanation claims are made.

## Threats to validity and limitations
ISOT may contain source, topic and time artifacts; its labels are dataset labels and not universal truth. GDELT related-news search is not evidence that a claim is true. Google Fact Check Tools returns indexed reviews only and may have no match. Model scores are not calibrated unless the specified validation-only procedure is applied and evaluated. A local reviewer gate is not production-grade authentication. Human labels depend on reviewer expertise, evidence quality and adjudication. Real-time operation requires fresh timestamps, contradictory-source handling and a correction process.

## Reporting rule
Never fabricate metric values, statistical significance, reviewer agreement, API coverage, or novelty. Include negative results. A working application is not sufficient by itself to establish a research contribution.
