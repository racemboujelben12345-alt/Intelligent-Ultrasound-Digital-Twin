"""Extract public ultrasound features for methodological characterization.

This script analyzes the public USSimAndSegm dataset only as a
methodological benchmark.

It must not be interpreted as a SCAN A baseline or as a SCAN A
Digital Signature. The real SCAN A signature will be established
later from repeated controlled acquisitions of the real system.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_analysis.edge_quality import fixed_canny_edge_density


ROOT = Path(
    "data/raw/public_ultrasound/"
    "us_simulation/abdominal_US/AUS"
)

OUTPUT_DIR = Path("outputs/public_benchmark")
OUTPUT_FILE = OUTPUT_DIR / "us_simandsegm_features.json"

SPLITS = ("train", "test")


def calculate_entropy(image: np.ndarray) -> float:
    """Calculate Shannon entropy of a grayscale uint8 image."""
    image_u8 = np.asarray(image, dtype=np.uint8)

    histogram = np.bincount(
        image_u8.ravel(),
        minlength=256,
    )

    probabilities = histogram.astype(np.float64)
    probabilities /= probabilities.sum()
    probabilities = probabilities[probabilities > 0]

    return float(
        -np.sum(probabilities * np.log2(probabilities))
    )


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate Laplacian-variance sharpness."""
    image_f = np.asarray(image, dtype=np.float32)

    if image_f.shape[0] < 3 or image_f.shape[1] < 3:
        raise ValueError("Image is too small for Laplacian calculation.")

    laplacian = (
        -4.0 * image_f[1:-1, 1:-1]
        + image_f[:-2, 1:-1]
        + image_f[2:, 1:-1]
        + image_f[1:-1, :-2]
        + image_f[1:-1, 2:]
    )

    return float(np.var(laplacian))


def extract_features(image_path: Path) -> dict[str, float | str]:
    """Extract the canonical public-benchmark feature set."""
    with Image.open(image_path) as image:
        image_gray = image.convert("L")
        image_array = np.asarray(image_gray, dtype=np.uint8)

    image_float = image_array.astype(np.float32) / 255.0

    mean_intensity = float(np.mean(image_array))
    std_intensity = float(np.std(image_array))

    min_intensity = float(np.min(image_array))
    max_intensity = float(np.max(image_array))

    dynamic_range = max_intensity - min_intensity

    p5, p95 = np.percentile(
        image_array,
        [5, 95],
    )

    percentile_contrast = float(p95 - p5)

    entropy = calculate_entropy(image_array)

    gradient_y, gradient_x = np.gradient(image_float)
    gradient_magnitude = np.sqrt(
        gradient_x**2 + gradient_y**2
    )

    mean_gradient = float(np.mean(gradient_magnitude))
    std_gradient = float(np.std(gradient_magnitude))

    # Canonical edge-density candidate currently used by the
    # prototype: fixed-threshold Canny (50/150).
    #
    # This remains a methodological candidate and is NOT yet
    # considered a final SCAN A signature feature.
    edge_density = fixed_canny_edge_density(
        image_array,
        threshold1=50.0,
        threshold2=150.0,
    )

    sharpness = calculate_sharpness(image_float)

    return {
        "image": image_path.name,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "dynamic_range": dynamic_range,
        "percentile_contrast": percentile_contrast,
        "entropy": entropy,
        "mean_gradient": mean_gradient,
        "std_gradient": std_gradient,
        "edge_density": edge_density,
        "sharpness": sharpness,
    }


def analyze_dataset() -> tuple[list[dict], list[dict]]:
    """Analyze all public ultrasound images."""
    results: list[dict] = []
    errors: list[dict] = []

    if not ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {ROOT}"
        )

    for split in SPLITS:
        image_dir = ROOT / "images" / split

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )

        print(f"\n{split.upper()} : {len(image_paths)} images")

        for image_path in image_paths:
            try:
                features = extract_features(image_path)
                features["split"] = split
                results.append(features)

            except Exception as exc:
                error = {
                    "image": image_path.name,
                    "split": split,
                    "error": str(exc),
                }
                errors.append(error)

                print(
                    f"ERROR: {image_path.name} -> {exc}"
                )

    return results, errors


def save_results(
    results: list[dict],
    errors: list[dict],
) -> None:
    """Save public-benchmark feature results."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    error_file = OUTPUT_DIR / "us_simandsegm_feature_errors.json"

    with open(
        error_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            errors,
            file,
            indent=2,
        )


def main() -> None:
    """Run public ultrasound feature extraction."""
    print("=" * 70)
    print("PUBLIC ULTRASOUND FEATURE BENCHMARK")
    print("=" * 70)

    print("\nDataset:")
    print("USSimAndSegm")

    print("\nPurpose:")
    print(
        "Methodological characterization only; "
        "not a SCAN A Digital Signature."
    )

    results, errors = analyze_dataset()

    save_results(
        results,
        errors,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    print(f"Images analyzed : {len(results)}")
    print(f"Errors          : {len(errors)}")

    print("\nFeatures:")

    if results:
        for key in results[0]:
            if key not in {"image", "split"}:
                print(f"  - {key}")

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"Features : {OUTPUT_FILE}")
    print(
        f"Errors   : "
        f"{OUTPUT_DIR / 'us_simandsegm_feature_errors.json'}"
    )

    print("\nPUBLIC BENCHMARK ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
