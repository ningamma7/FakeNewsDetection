import numpy as np
from backend.training.metrics import compute_metrics, expected_calibration_error


def test_metrics_perfect_predictions():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    result = compute_metrics(y, p)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["mcc"] == 1.0
    assert result["n"] == 4


def test_ece_is_nonnegative():
    value = expected_calibration_error(np.array([0, 1]), np.array([[0.8, 0.2], [0.2, 0.8]]))
    assert 0.0 <= value <= 1.0
