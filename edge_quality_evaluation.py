from pathlib import Path
import csv
import json

import cv2
import numpy as np

from src.image_analysis.edge_quality import (
    adaptive_median_canny_edge_density,
    fixed_canny_edge_density,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

USSIM_DIR = (
    ROOT
    / "data"
    / "raw"
    / "public_ultrasound"
    / "us_simulation"
    / "abdominal_US"
    / "AUS"
    / "images"
)

BUSI_DIR = (
    ROOT
    / "data"
    / "raw"
    / "public_ultrasound"
    / "busi"
    / "Dataset_BUSI_with_GT"
)

OUTPUT_DIR = ROOT / "outputs" / "public_benchmark" / "edge_quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

METHODS = [
    "p90_gradient",
    "fixed_gradient_0.05",
    "canny_50_150",
    "adaptive_canny",
]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    return image.astype(np.float32) / 255.0


# ============================================================
# GRADIENT
# ============================================================

def gradient_magnitude(image):
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    return np.sqrt(gx ** 2 + gy ** 2)


# ============================================================
# EDGE DEFINITIONS
# ============================================================

def edge_p90_gradient(image):
    """Historical percentile definition retained for comparison."""
    grad = gradient_magnitude(image)
    threshold = np.percentile(grad, 90)
    return float(np.mean(grad > threshold))


def edge_fixed_gradient(image, threshold=0.05):
    """Historical fixed-gradient definition retained for comparison."""
    grad = gradient_magnitude(image)
    return float(np.mean(grad > threshold))


def edge_canny_fixed(image):
    """Canonical fixed Canny definition from the image-analysis module."""
    return fixed_canny_edge_density(image)


def edge_adaptive_canny(image):
    """Canonical adaptive-median Canny definition from the image-analysis module."""
    return adaptive_median_canny_edge_density(image)


# ============================================================
# OTHER FEATURES
# ============================================================

def compute_reference_features(image):
    """
    Features used only to evaluate redundancy.

    They are NOT being selected or removed here.
    """

    grad = gradient_magnitude(image)

    laplacian = cv2.Laplacian(
        image,
        cv2.CV_32F,
    )

    return {
        "mean_gradient": float(
            np.mean(grad)
        ),
        "std_gradient": float(
            np.std(grad)
        ),
        "sharpness": float(
            np.var(laplacian)
        ),
    }


# ============================================================
# ANALYZE ONE IMAGE
# ============================================================

def analyze_image(path, dataset):
    image = load_grayscale(path)

    reference = compute_reference_features(image)

    result = {
        "dataset": dataset,
        "image": str(path.relative_to(ROOT)),
        "p90_gradient": edge_p90_gradient(image),
        "fixed_gradient_0.05": edge_fixed_gradient(image),
        "canny_50_150": edge_canny_fixed(image),
        "adaptive_canny": edge_adaptive_canny(image),
    }

    result.update(reference)

    return result


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_images(directory, exclude_masks=False):
    paths = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if exclude_masks and "_mask" in path.stem.lower():
            continue

        paths.append(path)

    return sorted(paths)


# ============================================================
# STATISTICS
# ============================================================

def mean_std_cv(values):
    values = np.asarray(values, dtype=np.float64)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    if abs(mean) < 1e-12:
        cv = None
    else:
        cv = float(std / abs(mean))

    return mean, std, cv


def cohens_d(group_a, group_b):
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    n_a = len(a)
    n_b = len(b)

    if n_a < 2 or n_b < 2:
        return None

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled_sd = np.sqrt(
        (
            (n_a - 1) * var_a
            + (n_b - 1) * var_b
        )
        / (n_a + n_b - 2)
    )

    if pooled_sd < 1e-12:
        return None

    return float(
        (np.mean(a) - np.mean(b))
        / pooled_sd
    )


def pearson_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 3:
        return None

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 72)
    print("SCAN A DIGITAL TWIN — EDGE QUALITY EVALUATION")
    print("=" * 72)

    ussim_images = discover_images(
        USSIM_DIR
    )

    busi_images = discover_images(
        BUSI_DIR,
        exclude_masks=True,
    )

    print()
    print(f"USSimAndSegm images : {len(ussim_images)}")
    print(f"BUSI images         : {len(busi_images)}")

    if len(ussim_images) == 0:
        raise RuntimeError(
            f"No USSimAndSegm images found in {USSIM_DIR}"
        )

    if len(busi_images) == 0:
        raise RuntimeError(
            f"No BUSI images found in {BUSI_DIR}"
        )

    results = []

    # --------------------------------------------------------
    # USSimAndSegm
    # --------------------------------------------------------

    print()
    print("Analyzing USSimAndSegm...")

    for index, path in enumerate(ussim_images, start=1):

        try:
            result = analyze_image(
                path,
                "USSimAndSegm",
            )

            results.append(result)

        except Exception as exc:
            print(
                f"[WARNING] {path}: {exc}"
            )

        if index % 100 == 0:
            print(
                f"  {index}/{len(ussim_images)}"
            )

    # --------------------------------------------------------
    # BUSI
    # --------------------------------------------------------

    print()
    print("Analyzing BUSI...")

    for index, path in enumerate(busi_images, start=1):

        try:
            result = analyze_image(
                path,
                "BUSI",
            )

            results.append(result)

        except Exception as exc:
            print(
                f"[WARNING] {path}: {exc}"
            )

        if index % 100 == 0:
            print(
                f"  {index}/{len(busi_images)}"
            )

    # --------------------------------------------------------
    # RAW RESULTS
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "edge_quality_per_image.csv"
    )

    fieldnames = [
        "dataset",
        "image",
        *METHODS,
        "mean_gradient",
        "std_gradient",
        "sharpness",
    ]

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

        writer.writerows(results)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {}

    for method in METHODS:

        all_values = [
            r[method]
            for r in results
        ]

        ussim_values = [
            r[method]
            for r in results
            if r["dataset"] == "USSimAndSegm"
        ]

        busi_values = [
            r[method]
            for r in results
            if r["dataset"] == "BUSI"
        ]

        mean, std, cv = mean_std_cv(
            all_values
        )

        ussim_mean, ussim_std, ussim_cv = mean_std_cv(
            ussim_values
        )

        busi_mean, busi_std, busi_cv = mean_std_cv(
            busi_values
        )

        summary[method] = {
            "pooled": {
                "n": len(all_values),
                "mean": mean,
                "std": std,
                "cv": cv,
            },
            "USSimAndSegm": {
                "n": len(ussim_values),
                "mean": ussim_mean,
                "std": ussim_std,
                "cv": ussim_cv,
            },
            "BUSI": {
                "n": len(busi_values),
                "mean": busi_mean,
                "std": busi_std,
                "cv": busi_cv,
            },
            "cross_dataset_cohens_d": cohens_d(
                ussim_values,
                busi_values,
            ),
        }

    # --------------------------------------------------------
    # REDUNDANCY
    # --------------------------------------------------------

    redundancy = {}

    reference_features = [
        "mean_gradient",
        "std_gradient",
        "sharpness",
    ]

    for method in METHODS:

        redundancy[method] = {}

        for reference in reference_features:

            x = [
                r[method]
                for r in results
            ]

            y = [
                r[reference]
                for r in results
            ]

            redundancy[method][reference] = (
                pearson_correlation(
                    x,
                    y,
                )
            )

    # --------------------------------------------------------
    # WRITE JSON
    # --------------------------------------------------------

    report = {
        "analysis": {
            "total_images": len(results),
            "USSimAndSegm": sum(
                r["dataset"] == "USSimAndSegm"
                for r in results
            ),
            "BUSI": sum(
                r["dataset"] == "BUSI"
                for r in results
            ),
        },
        "methods": summary,
        "correlation_with_reference_features": redundancy,
    }

    json_path = (
        OUTPUT_DIR
        / "edge_quality_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EDGE QUALITY SUMMARY")
    print("=" * 72)

    for method in METHODS:

        data = summary[method]

        print()
        print(method)

        print(
            f"  pooled mean : "
            f"{data['pooled']['mean']:.6f}"
        )

        print(
            f"  pooled std  : "
            f"{data['pooled']['std']:.6f}"
        )

        print(
            f"  pooled CV   : "
            f"{data['pooled']['cv']:.6f}"
        )

        print(
            f"  USSim mean  : "
            f"{data['USSimAndSegm']['mean']:.6f}"
        )

        print(
            f"  BUSI mean   : "
            f"{data['BUSI']['mean']:.6f}"
        )

        print(
            f"  Cohen's d   : "
            f"{data['cross_dataset_cohens_d']:.6f}"
        )

    print()
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(csv_path)
    print(json_path)

    print()
    print("EDGE QUALITY EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
