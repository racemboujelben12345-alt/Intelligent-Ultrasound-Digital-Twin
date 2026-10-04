from pathlib import Path
import json
import numpy as np

INPUT = Path(
    "outputs/public_benchmark/"
    "us_simandsegm_features.json"
)

OUTPUT = Path(
    "outputs/public_benchmark/"
    "feature_correlation.json"
)

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

features = [
    "mean_intensity",
    "std_intensity",
    "percentile_contrast",
    "entropy",
    "mean_gradient",
    "std_gradient",
    "sharpness",
]

X = np.array(
    [
        [item[f] for f in features]
        for item in data
    ],
    dtype=float
)

correlation = np.corrcoef(
    X,
    rowvar=False
)

print("=" * 70)
print("USSimAndSegm FEATURE CORRELATION ANALYSIS")
print("=" * 70)

print(f"Images analyzed : {len(data)}")
print(f"Features        : {len(features)}")

print("\nCORRELATION MATRIX")
print("-" * 70)

print(
    " " * 22 +
    " ".join(f"{f[:8]:>10}" for f in features)
)

for i, feature in enumerate(features):

    values = []

    for j in range(len(features)):
        values.append(
            f"{correlation[i, j]:10.3f}"
        )

    print(
        f"{feature[:20]:20s}" +
        "".join(values)
    )


print("\nKEY PAIRS")
print("-" * 70)

pairs = [
    ("std_intensity", "percentile_contrast"),
    ("mean_gradient", "sharpness"),
    ("mean_intensity", "std_intensity"),
    ("entropy", "sharpness"),
    ("mean_gradient", "std_gradient"),
]

for feature_a, feature_b in pairs:

    i = features.index(feature_a)
    j = features.index(feature_b)

    r = correlation[i, j]

    print(
        f"{feature_a:22s} <-> "
        f"{feature_b:22s} : r = {r:.4f}"
    )


# ----------------------------------------------------------
# Redundancy detection
# ----------------------------------------------------------

print("\nREDUNDANCY CHECK")
print("-" * 70)

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        r = correlation[i, j]

        if abs(r) >= 0.90:

            print(
                f"HIGH correlation : "
                f"{features[i]} <-> "
                f"{features[j]} "
                f"(r={r:.3f})"
            )

        elif abs(r) >= 0.70:

            print(
                f"MODERATE correlation : "
                f"{features[i]} <-> "
                f"{features[j]} "
                f"(r={r:.3f})"
            )


# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

report = {
    "dataset": "USSimAndSegm",
    "images": len(data),
    "features": features,
    "correlation_matrix": {
        features[i]: {
            features[j]: float(correlation[i, j])
            for j in range(len(features))
        }
        for i in range(len(features))
    }
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

print("\nOUTPUT")
print(OUTPUT)

print("\nCORRELATION ANALYSIS COMPLETE")
