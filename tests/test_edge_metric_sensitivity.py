import numpy as np
import pytest

from src.validation.edge_metric_sensitivity import evaluate_edge_metric_sensitivity


def _image():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 180
    image[12:20, 12:20] = 70
    return image


def test_report_is_json_friendly_and_separates_variants():
    image = _image()
    blurred = np.asarray(image, dtype=np.float32)
    # A deterministic digital transformation, not physical acquisition.
    import cv2
    blurred = cv2.GaussianBlur(blurred, (3, 3), 0)

    report = evaluate_edge_metric_sensitivity(
        image, {"blur_3x3": blurred, "identity": image.copy()}
    )
    assert report["evidence_class"] == "simulated_or_digitally_transformed"
    assert len(report["variants"]) == 2
    assert report["variants"][1]["perturbation"] == "identity"
    assert set(report["reference_metrics"]) == {
        "edge_density_fixed_canny",
        "edge_density_adaptive_median_canny",
        "edge_density_gradient_p90",
    }
    import json
    json.dumps(report)


def test_identity_variant_has_zero_changes():
    image = _image()
    report = evaluate_edge_metric_sensitivity(image, {"same": image.copy()})
    for values in report["variants"][0]["metrics"].values():
        assert values["absolute_change"] == pytest.approx(0.0)
        if values["reference"] != 0:
            assert values["relative_change"] == pytest.approx(0.0)


def test_rejects_empty_variants_and_invalid_images():
    with pytest.raises(ValueError, match="At least one"):
        evaluate_edge_metric_sensitivity(_image(), {})
    with pytest.raises(ValueError, match="2D"):
        evaluate_edge_metric_sensitivity(np.zeros((2, 2, 3)), {"copy": _image()})
    with pytest.raises(ValueError, match="non-finite"):
        bad = _image().astype(float)
        bad[0, 0] = np.nan
        evaluate_edge_metric_sensitivity(_image(), {"bad": bad})


def test_rejects_empty_variant_name():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_edge_metric_sensitivity(_image(), {" ": _image()})
