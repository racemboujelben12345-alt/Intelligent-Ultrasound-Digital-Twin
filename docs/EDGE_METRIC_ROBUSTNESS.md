# Edge-metric robustness audit

## Purpose

`src/validation/edge_metric_sensitivity.py` compares three existing edge-density
definitions (fixed Canny, median-adaptive Canny, and gradient-percentile density)
on a reference image and caller-supplied digital variants. For every metric it
reports the reference value, perturbed value, absolute change, and relative
change when the reference value is non-zero.

## How to interpret it

- Provide named variants produced by explicit, reproducible transformations.
- Compare definitions under the same input variants; do not select a winner from
  one image or one transformation.
- A zero baseline yields a null relative change because that ratio is undefined.
- This is a feature-sensitivity analysis, not evidence that a feature is
  physically repeatable, clinically meaningful, or validated for ultrasound
  phantom QA.
- Digital blur, noise, gain, clipping, or resizing only test the algorithm's
  response to those digital operations. They do not reproduce all effects of
  probe, beamforming, scanner settings, tissue, or acquisition conditions.
- Keep public, synthetic, and experimental evidence separately labelled.

## Example

```python
import cv2
from src.validation.edge_metric_sensitivity import evaluate_edge_metric_sensitivity

blurred = cv2.GaussianBlur(image, (3, 3), 0)
report = evaluate_edge_metric_sensitivity(
    image,
    {"gaussian_blur_3x3": blurred},
)
```

The function does not automatically transform images, tune thresholds, or
declare an edge definition universally superior. Threshold selection requires
a pre-specified protocol and representative validation data.
