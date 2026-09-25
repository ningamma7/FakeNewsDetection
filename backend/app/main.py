from pathlib import Path
from datetime import datetime

import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from .explainability import explain_text


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DISTILBERT_PATH = (
    BASE_DIR
    / "models"
    / "distilbert_fake_news"
)

DATA_DIR = BASE_DIR / "data"

FEEDBACK_PATH = (
    DATA_DIR
    / "prediction_feedback.csv"
)

VERIFIED_PATH = (
    DATA_DIR
    / "verified_articles.csv"
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

print("Loading DistilBERT tokenizer...")

try:

    tokenizer = AutoTokenizer.from_pretrained(
        str(DISTILBERT_PATH)
    )

    print("Loading DistilBERT model...")

    distilbert_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(DISTILBERT_PATH)
        )
    )

    distilbert_model.to("cpu")
    distilbert_model.eval()

    print("DistilBERT model loaded successfully!")

except Exception as e:

    print("Error loading DistilBERT model:")
    print(e)

    tokenizer = None
    distilbert_model = None


# ============================================================
# FILE INITIALIZATION
# ============================================================

def ensure_feedback_file():
    """
    Create the feedback CSV file if it does not exist.
    """

    if not FEEDBACK_PATH.exists():

        columns = [
            "id",
            "text",
            "prediction",
            "confidence",
            "verified_label",
            "verification_status",
            "final_label",
            "expert_notes",
            "date"
        ]

        pd.DataFrame(
            columns=columns
        ).to_csv(
            FEEDBACK_PATH,
            index=False
        )


def ensure_verified_file():
    """
    Create the verified articles CSV file if it does not exist.
    """

    if not VERIFIED_PATH.exists():

        columns = [
            "id",
            "text",
            "final_label",
            "expert_notes",
            "verification_status",
            "original_prediction",
            "original_confidence",
            "date",
            "training_status"
        ]

        pd.DataFrame(
            columns=columns
        ).to_csv(
            VERIFIED_PATH,
            index=False
        )


ensure_feedback_file()
ensure_verified_file()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fake News Detection API",
    description=(
        "AI-powered Fake News Detection using DistilBERT "
        "with Captum Explainability and Expert Review"
    ),
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# REQUEST MODELS
# ============================================================

class NewsRequest(BaseModel):
    article: str


class FeedbackRequest(BaseModel):
    article: str
    prediction: str
    confidence: float
    verified_label: str


class ExpertReviewRequest(BaseModel):
    decision: str
    final_label: str
    expert_notes: str = ""


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Fake News Detection API is running",
        "status": "success",
        "model": "DistilBERT",
        "explainability": "Captum Saliency",
        "endpoints": [
            "/predict",
            "/explain",
            "/feedback",
            "/feedback/pending",
            "/feedback/{feedback_id}/review",
            "/verified"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    if (
        tokenizer is None
        or distilbert_model is None
    ):

        return {
            "status": "error",
            "model_loaded": False
        }

    return {
        "status": "ok",
        "model_loaded": True,
        "model": "DistilBERT"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_news(request: NewsRequest):

    if (
        tokenizer is None
        or distilbert_model is None
    ):

        raise HTTPException(
            status_code=500,
            detail="DistilBERT model is not loaded."
        )

    article = request.article.strip()

    if not article:

        raise HTTPException(
            status_code=400,
            detail="Article cannot be empty."
        )

    try:

        # ----------------------------------------------------
        # Tokenize article
        # ----------------------------------------------------

        inputs = tokenizer(
            article,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = distilbert_model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

            prediction_value = int(
                torch.argmax(probabilities).item()
            )

            confidence = float(
                probabilities[
                    prediction_value
                ].item() * 100
            )

        # ----------------------------------------------------
        # Label mapping
        # ----------------------------------------------------

        # Current mapping:
        # 0 = Fake News
        # 1 = True News

        if prediction_value == 0:

            prediction = "Fake News"

        else:

            prediction = "True News"

        # ----------------------------------------------------
        # Confidence level
        # ----------------------------------------------------

        if confidence >= 90:

            confidence_level = "High confidence"

        elif confidence >= 70:

            confidence_level = "Moderate confidence"

        else:

            confidence_level = "Low confidence"

        # ----------------------------------------------------
        # Recommendation
        # ----------------------------------------------------

        if prediction == "Fake News":

            recommendation = (
                "Consider checking trustworthy sources "
                "before sharing."
            )

        else:

            recommendation = (
                "The model classified this article as "
                "True News, but independent verification "
                "is still recommended."
            )

        return {
            "prediction": prediction,
            "confidence_score": round(
                confidence,
                2
            ),
            "confidence_level": confidence_level,
            "recommendation": recommendation
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ============================================================
# EXPLAINABILITY ENDPOINT
# ============================================================

@app.post("/explain")
def explain_news(request: NewsRequest):

    if (
        tokenizer is None
        or distilbert_model is None
    ):

        raise HTTPException(
            status_code=500,
            detail="DistilBERT model is not loaded."
        )

    article = request.article.strip()

    if not article:

        raise HTTPException(
            status_code=400,
            detail="Article cannot be empty."
        )

    try:

        result = explain_text(
            article=article,
            tokenizer=tokenizer,
            model=distilbert_model,
            top_k=8
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Explainability error: {str(e)}"
        )


# ============================================================
# SAVE USER FEEDBACK
# ============================================================

@app.post("/feedback")
def save_feedback(request: FeedbackRequest):

    article = request.article.strip()

    if not article:

        raise HTTPException(
            status_code=400,
            detail="Article text cannot be empty."
        )

    allowed_labels = [
        "Fake News",
        "True News",
        "Unverified"
    ]

    if request.verified_label not in allowed_labels:

        raise HTTPException(
            status_code=400,
            detail=(
                "verified_label must be Fake News, "
                "True News, or Unverified."
            )
        )

    ensure_feedback_file()

    feedback_id = datetime.now().strftime(
        "%Y%m%d%H%M%S%f"
    )

    new_record = {
        "id": feedback_id,
        "text": article,
        "prediction": request.prediction,
        "confidence": request.confidence,
        "verified_label": request.verified_label,
        "verification_status": "Pending",
        "final_label": "",
        "expert_notes": "",
        "date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    feedback_df = pd.read_csv(
        FEEDBACK_PATH,
        dtype=str,
        keep_default_na=False
    )

    feedback_df = pd.concat(
        [
            feedback_df,
            pd.DataFrame([new_record])
        ],
        ignore_index=True
    )

    feedback_df.to_csv(
        FEEDBACK_PATH,
        index=False
    )

    return {
        "message": "Feedback saved successfully.",
        "feedback_id": feedback_id,
        "verification_status": "Pending"
    }


# ============================================================
# GET PENDING REVIEWS
# ============================================================

@app.get("/feedback/pending")
def get_pending_feedback():

    ensure_feedback_file()

    feedback_df = pd.read_csv(
        FEEDBACK_PATH,
        dtype=str,
        keep_default_na=False
    )

    if feedback_df.empty:

        return {
            "count": 0,
            "records": []
        }

    required_columns = [
        "id",
        "text",
        "prediction",
        "confidence",
        "verified_label",
        "verification_status",
        "final_label",
        "expert_notes",
        "date"
    ]

    for column in required_columns:

        if column not in feedback_df.columns:

            feedback_df[column] = ""

    pending_df = feedback_df[
        feedback_df["verification_status"] == "Pending"
    ]

    records = pending_df.to_dict(
        orient="records"
    )

    return {
        "count": len(records),
        "records": records
    }


# ============================================================
# EXPERT REVIEW ENDPOINT
# ============================================================

@app.post("/feedback/{feedback_id}/review")
def review_feedback(
    feedback_id: str,
    request: ExpertReviewRequest
):

    ensure_feedback_file()
    ensure_verified_file()

    allowed_decisions = [
        "Approve",
        "Reject"
    ]

    allowed_labels = [
        "Fake News",
        "True News"
    ]

    if request.decision not in allowed_decisions:

        raise HTTPException(
            status_code=400,
            detail="Decision must be Approve or Reject."
        )

    if request.final_label not in allowed_labels:

        raise HTTPException(
            status_code=400,
            detail=(
                "Final label must be Fake News or True News."
            )
        )

    feedback_df = pd.read_csv(
        FEEDBACK_PATH,
        dtype=str,
        keep_default_na=False
    )

    if feedback_df.empty:

        raise HTTPException(
            status_code=404,
            detail="No feedback records found."
        )

    feedback_df["id"] = (
        feedback_df["id"].astype(str)
    )

    matching_rows = feedback_df[
        feedback_df["id"] == str(feedback_id)
    ]

    if matching_rows.empty:

        raise HTTPException(
            status_code=404,
            detail="Feedback record not found."
        )

    index = matching_rows.index[0]

    current_status = feedback_df.loc[
        index,
        "verification_status"
    ]

    if current_status != "Pending":

        raise HTTPException(
            status_code=400,
            detail="This feedback has already been reviewed."
        )

    # --------------------------------------------------------
    # Update feedback record
    # --------------------------------------------------------

    if request.decision == "Approve":

        feedback_df.loc[
            index,
            "verification_status"
        ] = "Approved"

        feedback_df.loc[
            index,
            "final_label"
        ] = request.final_label

    else:

        feedback_df.loc[
            index,
            "verification_status"
        ] = "Rejected"

        feedback_df.loc[
            index,
            "final_label"
        ] = ""

    feedback_df.loc[
        index,
        "expert_notes"
    ] = request.expert_notes

    feedback_df.to_csv(
        FEEDBACK_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Save approved article to verified articles
    # --------------------------------------------------------

    if request.decision == "Approve":

        verified_df = pd.read_csv(
            VERIFIED_PATH,
            dtype=str,
            keep_default_na=False
        )

        if "id" not in verified_df.columns:

            verified_df["id"] = ""

        verified_df["id"] = (
            verified_df["id"].astype(str)
        )

        existing_record = verified_df[
            verified_df["id"] == str(feedback_id)
        ]

        if existing_record.empty:

            verified_record = {
                "id": str(feedback_id),

                "text": str(
                    feedback_df.loc[index, "text"]
                ),

                "final_label": request.final_label,

                "expert_notes": request.expert_notes,

                "verification_status": "Approved",

                "original_prediction": str(
                    feedback_df.loc[index, "prediction"]
                ),

                "original_confidence": str(
                    feedback_df.loc[index, "confidence"]
                ),

                "date": str(
                    feedback_df.loc[index, "date"]
                ),

                "training_status": "Not Used"
            }

            verified_df = pd.concat(
                [
                    verified_df,
                    pd.DataFrame([verified_record])
                ],
                ignore_index=True
            )

            verified_df.to_csv(
                VERIFIED_PATH,
                index=False
            )

    return {
        "message": "Expert review completed.",
        "feedback_id": str(feedback_id),
        "decision": request.decision,
        "final_label": (
            request.final_label
            if request.decision == "Approve"
            else None
        ),
        "verification_status": (
            "Approved"
            if request.decision == "Approve"
            else "Rejected"
        )
    }


# ============================================================
# GET VERIFIED ARTICLES
# ============================================================

@app.get("/verified")
def get_verified_articles():

    ensure_verified_file()

    verified_df = pd.read_csv(
        VERIFIED_PATH,
        dtype=str,
        keep_default_na=False
    )

    records = verified_df.to_dict(
        orient="records"
    )

    return {
        "count": len(records),
        "records": records
    }