# FakeNewsDetectionPlatform

## Overview

FakeNewsDetectionPlatform is a web-based application designed to help users assess news articles using machine learning and external fact-checking evidence. It uses TF-IDF feature extraction and Logistic Regression to classify article text and integrates the Google Fact Check Tools API to retrieve relevant fact-checks when available.

## Key Features

- Fake and real news classification using TF-IDF and Logistic Regression.
- Google Fact Check Tools API integration for retrieving existing fact-checks.
- FastAPI backend for prediction and application services.
- React frontend for user interaction.
- SQLite database for storing predictions and related records.
- Feedback and expert-review workflow to support verification.
- Clear separation between model predictions and external fact-checking evidence.

## Technology Stack

- **Machine Learning:** Python, scikit-learn, TF-IDF, Logistic Regression
- **Backend:** FastAPI
- **Frontend:** React, Vite
- **Database:** SQLite
- **Fact Checking:** Google Fact Check Tools API

## Disclaimer

The model's output is an automated prediction, not definitive proof that a news article is true or false. External fact-checks may not be available for every article, and users should consult reliable sources before drawing conclusions.

## Project Status

The current implementation focuses on the TF-IDF and Logistic Regression baseline with fact-checking integration. Features should be described as implemented only after they have been tested in the running application.
