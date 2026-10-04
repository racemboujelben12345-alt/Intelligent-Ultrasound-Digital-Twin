"""Extract public ultrasound features from the BUSI dataset.

BUSI is used only for public-dataset characterization and
methodological benchmarking.

These results must not be interpreted as a SCAN A baseline,
fault signature, or SCAN A Digital Signature.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_analysis.edge_quality import fixed_canny_edge_density


ROOT = Path(
    "data/raw/public_ultrasound/"
    "busi/Dataset_BUSI_with_GT"
)

OUTPUT_DIR = Path("outputs/public_benchmark")

CLASSES = (
    "benign",
    "malignant",
    "normal",
)

FEATURE_FIELDS = (
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "dynamic_range",
    "percentile_contrast",
    "entropy",
    "mean_gradient",
    "std_gradient",
    "edge_density",
    "sharpness",
)


def compute_entropy(image: np.ndarray) -> float:
    """Calculate Shannon entropy of a grayscale image."""
    image_u8 = np.asarray(
        image,
        dtype=np.uint8,
    )

    histogram = np.bincount(
        image_u8.ravel(),
        minlength=256,
    )

    probabilities = histogram.astype(np.float64)
    probabilities /= probabilities.sum()
    probabilities = probabilities[probabilities > 0]

    return float(
        -np.sum(
            probabilities * np.log2(probabilities)
        )
    )


def compute_sharpness(image: np.ndarray) -> float:
    """Calculate Laplacian-variance sharpness."""
    image_f = np.asarray(
        image,
        dtype=np.float32,
    )

    if image_f.shape[0] < 3 or image_f.shape[1] < 3:
        raise ValueError(
            "Image is too small for sharpness calculation."
        )

    g_x = np.gradient(
        image_f,
        axis=1,
    )

    g_y = np.gradient(
        image_f,
        axis=0,
    )

    g_xx = np.gradient(
        g_x,
        axis=1,
    )

    g_yy = np.gradient(
        g_y,
        axis=0,
    )

    laplacian = g_xx + g_yy

    return float(np.var(laplacian))


def compute_features(
    image_path: Path,
) -> dict[str, float]:
    """Extract the canonical public-benchmark feature set."""
    with Image.open(image_path) as image:
        image_gray = image.convert("L")
        image_array = np.asarray(
            image_gray,
            dtype=np.uint8,
        )

    image_float = (
        image_array.astype(np.float32) / 255.0
    )

    mean_intensity = float(
        np.mean(image_array)
    )

    std_intensity = float(
        np.std(image_array)
    )

    min_intensity = float(
        np.min(image_array)
    )

    max_intensity = float(
        np.max(image_array)
    )

    dynamic_range = (
        max_intensity - min_intensity
    )

    p5, p95 = np.percentile(
        image_array,
        [5, 95],
    )

    percentile_contrast = float(
        p95 - p5
    )

    entropy = compute_entropy(
        image_array
    )

    gradient_y, gradient_x = np.gradient(
        image_float
    )

    gradient_magnitude = np.sqrt(
        gradient_x**2 + gradient_y**2
    )

    mean_gradient = float(
        np.mean(gradient_magnitude)
    )

    std_gradient = float(
        np.std(gradient_magnitude)
    )

    # Canonical edge-density candidate:
    # fixed Canny thresholds 50/150.
    #
    # This is used for public methodological
    # characterization only. It is NOT a final
    # SCAN A Digital Signature feature.
    edge_density = fixed_canny_edge_density(
        image_array,
        threshold1=50.0,
        threshold2=150.0,
    )

    sharpness = compute_sharpness(
        image_float
    )

    return {
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


def collect_dataset() -> tuple[
    list[dict],
    int,
]:
    """Extract features from all BUSI images."""
    results: list[dict] = []
    errors = 0

    if not ROOT.exists():
        raise FileNotFoundError(
            f"BUSI dataset directory not found: {ROOT}"
        )

    for class_name in CLASSES:
        class_dir = ROOT / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"BUSI class directory not found: "
                f"{class_dir}"
            )

        image_files = sorted(
            path
            for path in class_dir.glob("*.png")
            if "_mask" not in path.stem
        )

        print(f"\n{class_name.upper()}")
        print("-" * 50)
        print(
            f"Images found : "
            f"{len(image_files)}"
        )

        for image_path in image_files:
            try:
                features = compute_features(
                    image_path
                )

                results.append(
                    {
                        "class": class_name,
                        "image": image_path.name,
                        **features,
                    }
                )

            except Exception as exc:
                errors += 1

                print(
                    f"ERROR : "
                    f"{image_path.name} -> {exc}"
                )

    return results, errors


def calculate_statistics(
    rows: list[dict],
) -> dict:
    """Calculate per-class and global feature statistics."""
    class_statistics = {}

    for class_name in CLASSES:
        subset = [
            row
            for row in rows
            if row["class"] == class_name
        ]

        class_statistics[class_name] = {
            "count": len(subset),
            "features": {},
        }

        for feature in FEATURE_FIELDS:
            values = np.asarray(
                [
                    row[feature]
                    for row in subset
                ],
                dtype=np.float64,
            )

            if values.size == 0:
                continue

            class_statistics[class_name][
                "features"
            ][feature] = {
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(values)
                ),
                "median": float(
                    np.median(values)
                ),
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }

    global_statistics = {
        "count": len(rows),
        "features": {},
    }

    for feature in FEATURE_FIELDS:
        values = np.asarray(
            [
                row[feature]
                for row in rows
            ],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        global_statistics["features"][
            feature
        ] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
            "median": float(
                np.median(values)
            ),
            "min": float(
                np.min(values)
            ),
            "max": float(
                np.max(values)
            ),
        }

    return {
        "classes": class_statistics,
        "global": global_statistics,
    }


def save_results(
    rows: list[dict],
    errors: int,
    statistics: dict,
) -> tuple[Path, Path]:
    """Save BUSI public-benchmark outputs."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR / "busi_features.csv"
    )

    fieldnames = (
        "class",
        "image",
        *FEATURE_FIELDS,
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    json_path = (
        OUTPUT_DIR
        / "busi_features_statistics.json"
    )

    output = {
        "dataset": "BUSI",
        "purpose": (
            "public methodological "
            "characterization"
        ),
        "not_scan_a_signature": True,
        "total_images": len(rows),
        "errors": errors,
        **statistics,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    return csv_path, json_path


def print_summary(
    rows: list[dict],
    errors: int,
    statistics: dict,
    csv_path: Path,
    json_path: Path,
) -> None:
    """Print a concise benchmark summary."""
    print("\n" + "=" * 70)
    print("BUSI PUBLIC BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"TOTAL IMAGES ANALYZED : "
        f"{len(rows)}"
    )

    print(
        f"ERRORS                 : "
        f"{errors}"
    )

    for class_name in CLASSES:
        print(
            f"\n{class_name.upper()}"
        )

        class_data = (
            statistics["classes"][
                class_name
            ]
        )

        print(
            f"Images : "
            f"{class_data['count']}"
        )

        for feature in (
            "mean_intensity",
            "std_intensity",
            "percentile_contrast",
            "entropy",
            "mean_gradient",
            "std_gradient",
            "edge_density",
            "sharpness",
        ):
            feature_data = (
                class_data["features"]
                .get(feature)
            )

            if feature_data is None:
                continue

            print(
                f"{feature:22s} "
                f"mean="
                f"{feature_data['mean']:.4f} "
                f"std="
                f"{feature_data['std']:.4f} "
                f"median="
                f"{feature_data['median']:.4f}"
            )

    print("\n" + "=" * 70)
    print("GLOBAL STATISTICS")
    print("=" * 70)

    for feature in (
        "mean_intensity",
        "std_intensity",
        "percentile_contrast",
        "entropy",
        "mean_gradient",
        "std_gradient",
        "edge_density",
        "sharpness",
    ):
        feature_data = (
            statistics["global"][
                "features"
            ].get(feature)
        )

        if feature_data is None:
            continue

        print(
            f"{feature:22s} "
            f"mean="
            f"{feature_data['mean']:.4f} "
            f"std="
            f"{feature_data['std']:.4f} "
            f"median="
            f"{feature_data['median']:.4f}"
        )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")

    print(
        "\nBUSI PUBLIC BENCHMARK "
        "ANALYSIS COMPLETE"
    )


def main() -> None:
    """Run BUSI public-benchmark analysis."""
    print("=" * 70)
    print("BUSI PUBLIC ULTRASOUND FEATURE BENCHMARK")
    print("=" * 70)

    print("\nPurpose:")
    print(
        "Methodological characterization only; "
        "not a SCAN A Digital Signature."
    )

    rows, errors = collect_dataset()

    statistics = calculate_statistics(
        rows
    )

    csv_path, json_path = save_results(
        rows,
        errors,
        statistics,
    )

    print_summary(
        rows,
        errors,
        statistics,
        csv_path,
        json_path,
    )


if __name__ == "__main__":
    main()
