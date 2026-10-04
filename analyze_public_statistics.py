from pathlib import Path
import json
import numpy as np

INPUT = Path(
    "outputs/public_benchmark/"
    "us_simandsegm_features.json"
)

OUTPUT = Path(
    "outputs/public_benchmark/"
    "us_simandsegm_statistics.json"
)

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

if not data:
    raise RuntimeError("No feature data found.")

excluded = {"image", "split"}

features = [
    key for key in data[0]
    if key not in excluded
]

statistics = {}

for feature in features:

    values = np.array(
        [item[feature] for item in data],
        dtype=float
    )

    statistics[feature] = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


# Train / test comparison
split_statistics = {}

for split in ["train", "test"]:

    subset = [
        item for item in data
        if item["split"] == split
    ]

    split_statistics[split] = {
        "count": len(subset)
    }

    for feature in features:

        values = np.array(
            [item[feature] for item in subset],
            dtype=float
        )

        split_statistics[split][feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }


# Feature correlations
matrix = np.array([
    [item[feature] for feature in features]
    for item in data
], dtype=float)

correlation = np.corrcoef(
    matrix,
    rowvar=False
)

correlation_matrix = {}

for i, feature_a in enumerate(features):

    correlation_matrix[feature_a] = {}

    for j, feature_b in enumerate(features):

        correlation_matrix[feature_a][feature_b] = float(
            correlation[i, j]
        )


report = {
    "dataset": "USSimAndSegm",
    "total_images": len(data),
    "number_of_features": len(features),
    "features": features,
    "statistics": statistics,
    "train_test": split_statistics,
    "correlation": correlation_matrix,
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print("=" * 60)
print("USSimAndSegm STATISTICAL ANALYSIS")
print("=" * 60)

print(f"Images   : {len(data)}")
print(f"Features : {len(features)}")

print("\nFEATURE STATISTICS")
print("-" * 60)

for feature in features:

    s = statistics[feature]

    print(
        f"{feature:22s} "
        f"mean={s['mean']:.4f} "
        f"std={s['std']:.4f} "
        f"median={s['median']:.4f}"
    )

print("\nTRAIN / TEST")
print("-" * 60)

for split in ["train", "test"]:

    print(
        f"{split.upper():5s}: "
        f"{split_statistics[split]['count']} images"
    )

print("\nOUTPUT")
print(OUTPUT)

print("\nANALYSIS COMPLETE")