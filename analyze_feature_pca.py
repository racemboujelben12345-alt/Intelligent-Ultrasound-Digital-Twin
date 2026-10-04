from pathlib import Path
import json
import numpy as np


INPUT = Path("outputs/public_benchmark/us_simandsegm_features.json")
OUTPUT = Path("outputs/public_benchmark/feature_pca.json")

FEATURES = [
    "mean_intensity",
    "std_intensity",
    "percentile_contrast",
    "entropy",
    "mean_gradient",
    "std_gradient",
    "sharpness",
]


def main():

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    X = np.array(
        [[item[feature] for feature in FEATURES] for item in data],
        dtype=float
    )

    print("=" * 70)
    print("USSimAndSegm PCA ANALYSIS")
    print("=" * 70)

    print(f"Images   : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")

    # ---------------------------------------------------------
    # 1. Standardization
    # ---------------------------------------------------------
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    Z = (X - mean) / std

    # ---------------------------------------------------------
    # 2. PCA using SVD
    # ---------------------------------------------------------
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)

    eigenvalues = (S ** 2) / (len(Z) - 1)

    explained_variance_ratio = (
        eigenvalues / eigenvalues.sum()
    )

    cumulative_variance = np.cumsum(
        explained_variance_ratio
    )

    # PCA loadings
    loadings = Vt.T * np.sqrt(eigenvalues)

    # ---------------------------------------------------------
    # 3. Results
    # ---------------------------------------------------------
    print()
    print("PRINCIPAL COMPONENTS")
    print("-" * 70)

    for i in range(len(FEATURES)):
        print(
            f"PC{i+1:<3} "
            f"explained={explained_variance_ratio[i] * 100:7.3f}% "
            f"cumulative={cumulative_variance[i] * 100:7.3f}%"
        )

    # ---------------------------------------------------------
    # 4. Number of components for 90% and 95%
    # ---------------------------------------------------------
    n90 = np.searchsorted(
        cumulative_variance, 0.90
    ) + 1

    n95 = np.searchsorted(
        cumulative_variance, 0.95
    ) + 1

    print()
    print("DIMENSIONALITY")
    print("-" * 70)
    print(f"Components for 90% variance : {n90}")
    print(f"Components for 95% variance : {n95}")

    # ---------------------------------------------------------
    # 5. Loadings
    # ---------------------------------------------------------
    print()
    print("PCA LOADINGS")
    print("-" * 70)

    for pc in range(len(FEATURES)):
        print(f"\nPC{pc + 1}")

        ranking = sorted(
            zip(FEATURES, loadings[:, pc]),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, value in ranking:
            print(f"  {feature:<22} {value: .4f}")

    # ---------------------------------------------------------
    # 6. Dominant feature
    # ---------------------------------------------------------
    print()
    print("DOMINANT FEATURE PER COMPONENT")
    print("-" * 70)

    dominant_features = {}

    for pc in range(len(FEATURES)):
        idx = np.argmax(np.abs(loadings[:, pc]))
        feature = FEATURES[idx]

        dominant_features[f"PC{pc + 1}"] = feature

        print(
            f"PC{pc + 1:<3} -> "
            f"{feature:<22} "
            f"| loading={loadings[idx, pc]: .4f}"
        )

    # ---------------------------------------------------------
    # 7. Save results
    # ---------------------------------------------------------
    result = {
        "dataset": "USSimAndSegm",
        "n_images": int(X.shape[0]),
        "features": FEATURES,

        "explained_variance_ratio": {
            f"PC{i + 1}": float(explained_variance_ratio[i])
            for i in range(len(FEATURES))
        },

        "cumulative_variance": {
            f"PC{i + 1}": float(cumulative_variance[i])
            for i in range(len(FEATURES))
        },

        "components_for_90_percent": int(n90),
        "components_for_95_percent": int(n95),

        "loadings": {
            f"PC{i + 1}": {
                FEATURES[j]: float(loadings[j, i])
                for j in range(len(FEATURES))
            }
            for i in range(len(FEATURES))
        },

        "dominant_features": dominant_features,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(OUTPUT)
    print()
    print("PCA ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
