import numpy as np

from src.validation.metrics import compute_detection_metrics


def test_detection_metrics_expose_core_classification_metrics():
    result = compute_detection_metrics(
        distances=np.array([0.2, 0.4, 3.0, 4.0]),
        states=("NOMINAL", "NOMINAL", "EARLY_DRIFT", "HIGH_DEVIATION"),
        expected_anomaly=(False, False, True, True),
    )

    assert result.true_positive == 2
    assert result.false_positive == 0
    assert result.true_negative == 2
    assert result.false_negative == 0
    assert result.sensitivity == 1.0
    assert result.specificity == 1.0
    assert result.precision == 1.0
    assert result.f1_score == 1.0
    assert result.negative_predictive_value == 1.0
    assert result.balanced_accuracy == 1.0


def test_detection_metrics_accept_single_ground_truth_label():
    result = compute_detection_metrics(
        distances=[0.1, 0.2],
        states=("NOMINAL", "EARLY_DRIFT"),
        expected_anomaly=False,
    )

    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 0
    assert result.true_positive == 0
