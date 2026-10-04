from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

INPUT = Path("outputs/public_benchmark/cross_dataset_features.csv")
OUTPUT_DIR = Path("outputs/public_benchmark")

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
    "sharpness",
]

CORR_THRESHOLD = 0.90


def rank_average(values):
    """Average ranks, including ties, without scipy."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rank_average(x), rank_average(y))


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def stats_for(values):
    x = np.asarray(values, dtype=float)
    q25, q75 = np.percentile(x, [25, 75])
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    return {
        "n": int(len(x)),
        "mean": mean,
        "std": std,
        "median": float(np.median(x)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "unique_values": int(np.unique(x).size),
        "cv_abs": float(std / abs(mean)) if mean != 0 else None,
    }


def correlation_pairs(df, method_name):
    pairs = []
    matrix = df[FEATURES].corr(method=method_name)
    for i, f1 in enumerate(FEATURES):
        for f2 in FEATURES[i + 1:]:
            r = float(matrix.loc[f1, f2])
            if math.isfinite(r) and abs(r) >= CORR_THRESHOLD:
                pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": r,
                    "absolute_correlation": abs(r),
                })
    pairs.sort(key=lambda z: z["absolute_correlation"], reverse=True)
    return pairs


def flag_feature(name, pooled, ussim, busi):
    p = pooled[name]
    flags = []

    # A feature with almost no variation cannot discriminate normal drift
    # under its current definition.
    if p["unique_values"] <= 3:
        flags.append("near_constant")

    # Current edge-density definition uses the 90th percentile per image,
    # so values are expected to cluster around 0.10 by construction.
    if name == "edge_density" and p["std"] < 0.005:
        flags.append("definitionally_constrained")

    # Exact 0/255 extrema can be dominated by image encoding/saturation.
    if name in {"min_intensity", "max_intensity", "dynamic_range"}:
        flags.append("extrema_or_saturation_sensitive")

    # Large public-dataset domain gap means no universal cross-dataset
    # threshold should be transferred to SCAN A.
    d = cohens_d(
        [0] * 0 if False else [],  # replaced below
        []
    ) if False else None

    # Redundancy is evaluated separately from the flags.
    return flags


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run compare_public_datasets.py first."
        )

    df = pd.read_csv(INPUT)

    required = set(FEATURES + ["dataset"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=FEATURES + ["dataset"]).copy()

    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=FEATURES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].unique().tolist())
    if len(datasets) != 2:
        raise ValueError(
            f"Expected exactly 2 datasets (USSimAndSegm and BUSI), got {datasets}"
        )

    ussim = df[df["dataset"] == "USSimAndSegm"]
    busi = df[df["dataset"] == "BUSI"]

    pooled_stats = {f: stats_for(df[f]) for f in FEATURES}
    ussim_stats = {f: stats_for(ussim[f]) for f in FEATURES}
    busi_stats = {f: stats_for(busi[f]) for f in FEATURES}

    pearson_pooled = df[FEATURES].corr(method="pearson")
    spearman_pooled = df[FEATURES].corr(method="spearman")

    pearson_ussim = ussim[FEATURES].corr(method="pearson")
    spearman_ussim = ussim[FEATURES].corr(method="spearman")

    pearson_busi = busi[FEATURES].corr(method="pearson")
    spearman_busi = busi[FEATURES].corr(method="spearman")

    domain_sensitivity = {}
    for f in FEATURES:
        d = cohens_d(ussim[f].values, busi[f].values)
        mean_gap = float(busi[f].mean() - ussim[f].mean())
        domain_sensitivity[f] = {
            "ussim_mean": float(ussim[f].mean()),
            "busi_mean": float(busi[f].mean()),
            "mean_gap_busi_minus_ussim": mean_gap,
            "cohens_d": d,
            "absolute_cohens_d": abs(d) if math.isfinite(d) else None,
            "interpretation": (
                "low domain sensitivity" if abs(d) < 0.2 else
                "moderate domain sensitivity" if abs(d) < 0.5 else
                "high domain sensitivity"
            ),
        }

    flags = {}
    for f in FEATURES:
        flags[f] = flag_feature(f, pooled_stats, ussim_stats, busi_stats)

    # Add redundancy flags from pooled and per-domain analyses.
    for pairs in [
        correlation_pairs(df, "pearson"),
        correlation_pairs(df, "spearman"),
        correlation_pairs(ussim, "pearson"),
        correlation_pairs(busi, "pearson"),
    ]:
        for p in pairs:
            for f in [p["feature_1"], p["feature_2"]]:
                if "high_redundancy" not in flags[f]:
                    flags[f].append("high_redundancy")

    # Explicit expert interpretation of the current implementation.
    if "definitionally_constrained" not in flags["edge_density"]:
        flags["edge_density"].append("review_definition")

    report = {
        "title": "SCAN A Digital Twin — Public Feature Quality Audit",
        "purpose": (
            "Engineering audit of candidate ultrasound image features using "
            "public datasets only. This report does not define the SCAN A baseline "
            "and does not replace validation on real SCAN A acquisitions."
        ),
        "input": str(INPUT),
        "n_total": int(len(df)),
        "datasets": {
            "USSimAndSegm": int(len(ussim)),
            "BUSI": int(len(busi)),
        },
        "features": FEATURES,
        "pooled_statistics": pooled_stats,
        "dataset_statistics": {
            "USSimAndSegm": ussim_stats,
            "BUSI": busi_stats,
        },
        "correlations": {
            "pooled_pearson": pearson_pooled.to_dict(),
            "pooled_spearman": spearman_pooled.to_dict(),
            "USSimAndSegm_pearson": pearson_ussim.to_dict(),
            "USSimAndSegm_spearman": spearman_ussim.to_dict(),
            "BUSI_pearson": pearson_busi.to_dict(),
            "BUSI_spearman": spearman_busi.to_dict(),
        },
        "high_correlation_pairs": {
            "pooled_pearson": correlation_pairs(df, "pearson"),
            "pooled_spearman": correlation_pairs(df, "spearman"),
            "USSimAndSegm_pearson": correlation_pairs(ussim, "pearson"),
            "BUSI_pearson": correlation_pairs(busi, "pearson"),
        },
        "cross_domain_sensitivity": domain_sensitivity,
        "flags": flags,
        "engineering_conclusions": [
            "Public datasets must not be used to construct the SCAN A baseline.",
            "High correlation indicates redundancy, not automatic deletion; the final signature must be validated on repeated real SCAN A acquisitions.",
            "edge_density requires review because the current 90th-percentile threshold makes its value close to 0.10 by construction.",
            "min_intensity, max_intensity and dynamic_range should be treated cautiously because extrema can be dominated by image saturation/encoding.",
            "Large cross-dataset effect sizes mean universal public-dataset thresholds should not be transferred to SCAN A.",
            "The final SCAN A Digital Signature must be selected from real-device repeatability, engineering relevance, and anomaly-detection performance.",
        ],
    }

    json_path = OUTPUT_DIR / "feature_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Human-readable Markdown report.
    md = []
    md.append("# SCAN A Digital Twin — Public Feature Quality Audit")
    md.append("")
    md.append(
        "> Public-data engineering audit only. These results do **not** define "
        "the SCAN A baseline. Final feature selection requires real SCAN A acquisitions."
    )
    md.append("")
    md.append(f"- Total images analysed: **{len(df)}**")
    md.append(f"- USSimAndSegm: **{len(ussim)}**")
    md.append(f"- BUSI: **{len(busi)}**")
    md.append("")
    md.append("## Feature flags")
    md.append("")
    md.append("| Feature | Flags | |Cohen's d| |")
    md.append("|---|---|---:|")
    for f in FEATURES:
        flag_text = ", ".join(flags[f]) if flags[f] else "none"
        effect = domain_sensitivity[f]["absolute_cohens_d"]
        effect_text = f"{effect:.3f}" if effect is not None else "NA"
        md.append(f"| `{f}` | {flag_text} | {effect_text} |")

    md.append("")
    md.append("## High-correlation pairs (pooled Pearson, |r| ≥ 0.90)")
    md.append("")
    pooled_pairs = correlation_pairs(df, "pearson")
    if pooled_pairs:
        md.append("| Feature 1 | Feature 2 | r |")
        md.append("|---|---|---:|")
        for p in pooled_pairs:
            md.append(
                f"| `{p['feature_1']}` | `{p['feature_2']}` | {p['correlation']:.4f} |"
            )
    else:
        md.append("No pooled Pearson pair exceeded the threshold.")

    md.append("")
    md.append("## Engineering interpretation")
    md.append("")
    for item in report["engineering_conclusions"]:
        md.append(f"- {item}")

    md.append("")
    md.append("## Next decision gate")
    md.append("")
    md.append(
        "The next step is **not another public dataset**. The next step is repeated "
        "controlled acquisition on the real SCAN A, followed by repeatability analysis "
        "and validation of candidate features."
    )

    md_path = OUTPUT_DIR / "feature_quality_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    # Export correlation matrices for easy inspection.
    pearson_pooled.to_csv(OUTPUT_DIR / "feature_correlation_pearson_pooled.csv")
    spearman_pooled.to_csv(OUTPUT_DIR / "feature_correlation_spearman_pooled.csv")

    print("=" * 70)
    print("PUBLIC FEATURE QUALITY AUDIT")
    print("=" * 70)
    print(f"Images analysed : {len(df)}")
    print(f"USSimAndSegm    : {len(ussim)}")
    print(f"BUSI            : {len(busi)}")
    print("")
    print("FEATURE FLAGS")
    for f in FEATURES:
        flag_text = ", ".join(flags[f]) if flags[f] else "none"
        print(f"  {f:22s} -> {flag_text}")
    print("")
    print("HIGH CORRELATION PAIRS (POOLED PEARSON)")
    for p in correlation_pairs(df, "pearson"):
        print(
            f"  {p['feature_1']} <-> {p['feature_2']} : "
            f"r={p['correlation']:.4f}"
        )
    print("")
    print("OUTPUTS")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {OUTPUT_DIR / 'feature_correlation_pearson_pooled.csv'}")
    print(f"  {OUTPUT_DIR / 'feature_correlation_spearman_pooled.csv'}")
    print("")
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
