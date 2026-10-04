from pathlib import Path
from PIL import Image
import numpy as np
import json
import csv

# ============================================================
# DATASET ROOTS
# ============================================================

USSIM_ROOT = Path(
    "data/raw/public_ultrasound/"
    "us_simulation/abdominal_US/AUS/images"
)

BUSI_ROOT = Path(
    "data/raw/public_ultrasound/"
    "busi/Dataset_BUSI_with_GT"
)

OUTPUT_DIR = Path("outputs/public_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["benign", "malignant", "normal"]


# ============================================================
# FEATURE EXTRACTION
# SAME FORMULAS AS USSimAndSegm
# ============================================================

def calculate_entropy(image):

    hist = np.bincount(
        image.astype(np.uint8).ravel(),
        minlength=256
    )

    probability = hist / hist.sum()
    probability = probability[probability > 0]

    return float(
        -np.sum(
            probability *
            np.log2(probability)
        )
    )


def extract_features(image_path):

    image = Image.open(image_path).convert("L")

    x = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # INTENSITY
    # --------------------------------------------------------

    mean_intensity = float(np.mean(x))
    std_intensity = float(np.std(x))

    min_intensity = float(np.min(x))
    max_intensity = float(np.max(x))

    dynamic_range = (
        max_intensity -
        min_intensity
    )

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    p5 = float(
        np.percentile(x, 5)
    )

    p95 = float(
        np.percentile(x, 95)
    )

    percentile_contrast = p95 - p5

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    image_entropy = calculate_entropy(x)

    # --------------------------------------------------------
    # GRADIENT
    # EXACT SAME IMPLEMENTATION
    # --------------------------------------------------------

    gy, gx = np.gradient(x)

    gradient_magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    mean_gradient = float(
        np.mean(gradient_magnitude)
    )

    std_gradient = float(
        np.std(gradient_magnitude)
    )

    # --------------------------------------------------------
    # EDGE DENSITY
    # EXACT SAME IMPLEMENTATION
    # --------------------------------------------------------

    threshold = np.percentile(
        gradient_magnitude,
        90
    )

    edge_density = float(
        np.mean(
            gradient_magnitude > threshold
        )
    )

    # --------------------------------------------------------
    # SHARPNESS
    # EXACT SAME IMPLEMENTATION
    # --------------------------------------------------------

    laplacian = (
        -4 * x[1:-1, 1:-1]
        + x[:-2, 1:-1]
        + x[2:, 1:-1]
        + x[1:-1, :-2]
        + x[1:-1, 2:]
    )

    sharpness = float(
        np.var(laplacian)
    )

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "dynamic_range": dynamic_range,
        "percentile_contrast": percentile_contrast,
        "entropy": image_entropy,
        "mean_gradient": mean_gradient,
        "std_gradient": std_gradient,
        "edge_density": edge_density,
        "sharpness": sharpness
    }


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
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
    "sharpness"
]


# ============================================================
# ANALYZE USSimAndSegm
# ============================================================

print("=" * 70)
print("CROSS-DATASET ULTRASOUND DIGITAL SIGNATURE BENCHMARK")
print("=" * 70)

all_results = []


print("\n[1/2] USSimAndSegm")
print("-" * 70)

us_sim_count = 0

for split in ["train", "test"]:

    image_dir = USSIM_ROOT / split

    image_paths = sorted(
        image_dir.glob("*.png")
    )

    print(
        f"{split.upper()} : "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        try:

            features = extract_features(
                image_path
            )

            row = {
                "dataset": "USSimAndSegm",
                "class": "abdominal_ultrasound",
                "split": split,
                "image": image_path.name,
                **features
            }

            all_results.append(row)

            us_sim_count += 1

        except Exception as e:

            print(
                f"ERROR: {image_path.name} -> {e}"
            )


print(
    f"USSimAndSegm analyzed : {us_sim_count}"
)


# ============================================================
# ANALYZE BUSI
# ============================================================

print("\n[2/2] BUSI")
print("-" * 70)

busi_count = 0

for cls in CLASSES:

    folder = BUSI_ROOT / cls

    image_paths = sorted(
        p for p in folder.glob("*.png")
        if "_mask" not in p.stem
    )

    print(
        f"{cls.upper()} : "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        try:

            features = extract_features(
                image_path
            )

            row = {
                "dataset": "BUSI",
                "class": cls,
                "split": "not_applicable",
                "image": image_path.name,
                **features
            }

            all_results.append(row)

            busi_count += 1

        except Exception as e:

            print(
                f"ERROR: {image_path.name} -> {e}"
            )


print(
    f"BUSI analyzed : {busi_count}"
)


# ============================================================
# SAVE COMBINED CSV
# ============================================================

csv_file = (
    OUTPUT_DIR /
    "cross_dataset_features.csv"
)

fieldnames = [
    "dataset",
    "class",
    "split",
    "image"
] + FEATURES


with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# STATISTICS
# ============================================================

statistics = {}


for dataset in [
    "USSimAndSegm",
    "BUSI"
]:

    subset = [
        row for row in all_results
        if row["dataset"] == dataset
    ]

    statistics[dataset] = {
        "count": len(subset),
        "features": {}
    }

    for feature in FEATURES:

        values = np.array(
            [
                row[feature]
                for row in subset
            ],
            dtype=np.float64
        )

        statistics[dataset]["features"][feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }


# ============================================================
# FEATURE CORRELATION
# ============================================================

correlations = {}

for dataset in [
    "USSimAndSegm",
    "BUSI"
]:

    subset = [
        row for row in all_results
        if row["dataset"] == dataset
    ]

    matrix = np.array(
        [
            [
                row[feature]
                for feature in FEATURES
            ]
            for row in subset
        ],
        dtype=np.float64
    )

    corr = np.corrcoef(
        matrix,
        rowvar=False
    )

    correlations[dataset] = {}

    for i, feature_a in enumerate(FEATURES):

        correlations[dataset][feature_a] = {}

        for j, feature_b in enumerate(FEATURES):

            correlations[dataset][feature_a][feature_b] = float(
                corr[i, j]
            )


# ============================================================
# SAVE JSON
# ============================================================

json_output = {
    "description": (
        "Standardized cross-dataset ultrasound "
        "digital signature benchmark"
    ),
    "datasets": statistics,
    "correlations": correlations
}

json_file = (
    OUTPUT_DIR /
    "cross_dataset_statistics.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        indent=2
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CROSS-DATASET STATISTICS")
print("=" * 70)

for feature in FEATURES:

    print(f"\n{feature}")

    for dataset in [
        "USSimAndSegm",
        "BUSI"
    ]:

        s = (
            statistics
            [dataset]
            ["features"]
            [feature]
        )

        print(
            f"  {dataset:18s} "
            f"mean={s['mean']:.4f} "
            f"std={s['std']:.4f} "
            f"median={s['median']:.4f}"
        )


print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"Combined CSV : {csv_file}"
)

print(
    f"Statistics   : {json_file}"
)

print("\n" + "=" * 70)
print("CROSS-DATASET BENCHMARK COMPLETE")
print("=" * 70)