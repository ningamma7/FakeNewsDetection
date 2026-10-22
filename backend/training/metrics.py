from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1] if i + 1 < bins else confidence <= edges[i + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float((prediction[mask] == y[mask]).mean()) - float(confidence[mask].mean()))
    return float(ece)


def compute_metrics(y_true, probabilities) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p.argmax(axis=1)
    y_onehot = np.eye(2)[y]
    report = classification_report(y, pred, labels=[0, 1], target_names=["Fake News", "True News"], output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None
    except ValueError:
        auc = None
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else 0.0,
        "roc_auc_true_class": auc,
        "ece_10_bins": expected_calibration_error(y, p),
        "brier_score_multiclass": float(np.mean(np.sum((p - y_onehot) ** 2, axis=1))),
        "confusion_matrix_labels_0_fake_1_true": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": report,
    }
