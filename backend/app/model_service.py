from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np
from .config import MODEL_DIR, BASELINE_MODEL_PATH, MAX_LENGTH, ABSTAIN_THRESHOLD

LABEL_NAMES = {0: "Fake News", 1: "True News"}


class ModelService:
    """Loads a local transformer if available; otherwise uses the TF-IDF baseline.

    The classifier is not a factual verification service. Probabilities are raw model
    scores unless a separate validation-only calibration procedure has been applied.
    """
    def __init__(self) -> None:
        self.model_type: str | None = None
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_path: str | None = None
        self.load_error: str | None = None
        self.reload()

    def reload(self) -> dict[str, Any]:
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.model_path = None
        self.load_error = None

        if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
                self.model.to(self.device)
                self.model.eval()
                self.model_type = "transformer"
                self.model_path = str(MODEL_DIR)
                return self.status()
            except Exception as exc:
                self.load_error = f"Transformer model could not be loaded from {MODEL_DIR}: {exc}"

        if BASELINE_MODEL_PATH.exists():
            try:
                self.model = joblib.load(BASELINE_MODEL_PATH)
                self.model_type = "tfidf"
                self.model_path = str(BASELINE_MODEL_PATH)
                return self.status()
            except Exception as exc:
                self.load_error = f"TF-IDF baseline could not be loaded from {BASELINE_MODEL_PATH}: {exc}"
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.device if self.model_type == "transformer" else "cpu",
            "max_length": MAX_LENGTH if self.model_type == "transformer" else None,
            "confidence_calibrated": False,
            "load_error": self.load_error,
        }

    def predict(self, text: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("No model is available. Train a TF-IDF baseline or set MODEL_DIR to a trained DistilBERT directory.")
        if self.model_type == "transformer":
            return self._predict_transformer(text)
        return self._predict_tfidf(text)

    def _predict_transformer(self, text: str) -> dict[str, Any]:
        import torch
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])
        explanation = self._explain_transformer(encoded, label_id)
        return self._result(label_id, confidence, probs, explanation)

    def _explain_transformer(self, encoded: dict[str, Any], label_id: int) -> dict[str, Any]:
        try:
            import torch
            from captum.attr import Saliency
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            embedding_layer = self.model.get_input_embeddings()
            input_embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

            def forward_embeds(embeddings, mask):
                return self.model(inputs_embeds=embeddings, attention_mask=mask).logits

            self.model.zero_grad(set_to_none=True)
            attribution = Saliency(forward_embeds).attribute(
                input_embeddings, target=label_id, additional_forward_args=(attention_mask,), abs=True
            )
            scores = attribution.abs().sum(dim=-1)[0].detach().cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
            special_ids = set(self.tokenizer.all_special_ids)
            ids = input_ids[0].detach().cpu().tolist()
            candidates = [(i, str(tok), float(scores[i])) for i, tok in enumerate(tokens)
                          if ids[i] not in special_ids and tok.strip() and float(scores[i]) > 0]
            candidates.sort(key=lambda row: row[2], reverse=True)
            peak = max((v for _, _, v in candidates), default=1.0) or 1.0
            top = [{"token": tok.replace("##", ""), "importance": round(val / peak, 4)}
                   for _, tok, val in candidates[:10]]
            return {"method": "Captum Saliency", "status": "available", "features": top,
                    "note": "Token attribution indicates model sensitivity, not proof or evidence of truth."}
        except Exception as exc:
            return {"method": "Captum Saliency", "status": "unavailable", "features": [],
                    "note": f"Explanation was not available: {exc}"}

    def _predict_tfidf(self, text: str) -> dict[str, Any]:
        model = self.model
        pred = int(model.predict([text])[0])
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba([text])[0], dtype=float)
            classes = list(getattr(model, "classes_", [0, 1]))
            ordered = np.zeros(2, dtype=float)
            for cls, prob in zip(classes, probs):
                if int(cls) in (0, 1):
                    ordered[int(cls)] = float(prob)
            probs = ordered
            confidence = float(probs[pred])
        else:
            probs = np.array([0.5, 0.5])
            confidence = 0.5
        explanation = self._explain_tfidf(text, pred)
        return self._result(pred, confidence, probs, explanation)

    def _explain_tfidf(self, text: str, label_id: int) -> dict[str, Any]:
        try:
            if not hasattr(self.model, "named_steps"):
                return {"method": "TF-IDF feature weights", "status": "unavailable", "features": []}
            vec = self.model.named_steps.get("tfidf")
            clf = self.model.named_steps.get("classifier")
            if vec is None or clf is None or not hasattr(clf, "coef_"):
                return {"method": "TF-IDF feature weights", "status": "unavailable", "features": []}
            sparse = vec.transform([text])
            names = np.asarray(vec.get_feature_names_out())
            coef = np.asarray(clf.coef_[0])
            contributions = sparse.toarray()[0] * coef
            if label_id == 0:
                contributions = -contributions
            nz = np.flatnonzero(sparse.toarray()[0])
            ranks = sorted(nz, key=lambda idx: abs(contributions[idx]), reverse=True)[:10]
            peak = max((abs(float(contributions[i])) for i in ranks), default=1.0) or 1.0
            features = [{"token": str(names[i]), "importance": round(abs(float(contributions[i])) / peak, 4),
                         "direction": "supports_prediction" if contributions[i] >= 0 else "opposes_prediction"}
                        for i in ranks]
            return {"method": "TF-IDF feature weights", "status": "available", "features": features,
                    "note": "Feature weights explain the fitted classifier, not the factual status of an article."}
        except Exception as exc:
            return {"method": "TF-IDF feature weights", "status": "unavailable", "features": [], "note": str(exc)}

    @staticmethod
    def _result(label_id: int, confidence: float, probabilities: np.ndarray,
                explanation: dict[str, Any]) -> dict[str, Any]:
        probs = {LABEL_NAMES[0]: round(float(probabilities[0]), 6), LABEL_NAMES[1]: round(float(probabilities[1]), 6)}
        return {
            "prediction_id": label_id,
            "prediction": LABEL_NAMES[label_id],
            "confidence": round(confidence, 6),
            "confidence_level": "high" if confidence >= 0.90 else "moderate" if confidence >= 0.70 else "low",
            "needs_human_review": confidence < ABSTAIN_THRESHOLD,
            "abstention_threshold": ABSTAIN_THRESHOLD,
            "probabilities": probs,
            "confidence_calibrated": False,
            "model_type": "transformer" if explanation.get("method") == "Captum Saliency" else "tfidf",
            "explanation": explanation,
            "disclaimer": "This is a text-classification prediction, not a factual verdict. The displayed confidence is not calibrated unless an independently validated calibration step has been run.",
        }
