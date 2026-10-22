import numpy as np
from backend.app.model_service import ModelService


def test_result_labels_and_not_truth_verdict():
    result = ModelService._result(0, 0.7, np.array([0.7, 0.3]), {"method": "Captum Saliency", "status": "unavailable", "features": []})
    assert result["prediction"] == "Fake News"
    assert result["probabilities"]["True News"] == 0.3
    assert result["needs_human_review"] is False
    assert "not a factual verdict" in result["disclaimer"]
    assert result["confidence_calibrated"] is False
