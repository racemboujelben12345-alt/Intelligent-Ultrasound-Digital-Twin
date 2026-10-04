"""Research-grade feature quality analysis for ultrasound candidate features.

The analyzer is intentionally dataset-agnostic: it consumes a feature table with
one row per image and explicit dataset labels. It does not select or delete
features automatically. It produces evidence for the engineering decision.

Tests implemented:
1. Variability: robust spread + coefficient of variation + near-constant flag.
2. Redundancy: Pearson and Spearman correlation matrices + highly correlated pairs.
3. Cross-domain sensitivity: effect size and distribution-shift statistics between
   two public ultrasound domains.
4. Engineering relevance: a transparent, rule-based interpretation table. This
   is a research aid, not a clinical claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CANDIDATE_FEATURES = [
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

ENGINEERING_GROUPS = {
    "mean_intensity": "intensity",
    "std_intensity": "intensity",
    "min_intensity": "intensity",
    "max_intensity": "intensity",
    "dynamic_range": "intensity",
    "percentile_contrast": "intensity",
    "entropy": "information_texture",
    "mean_gradient": "structural_variation",
    "std_gradient": "structural_variation",
    "edge_density": "structural_variation",
    "sharpness": "sharpness",
}

# Conservative thresholds. They are screening thresholds, not universal laws.
NEAR_CONSTANT_CV = 0.01
HIGH_REDUNDANCY_ABS_R = 0.90
VERY_HIGH_REDUNDANCY_ABS_R = 0.95
DOMAIN_SHIFT_SMD = 0.50
DOMAIN_SHIFT_KS = 0.20


@dataclass(frozen=True)
class DomainComparison:
    feature: str
    n_domain_a: int
    n_domain_b: int
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    std_a: float
    std_b: float
    standardized_mean_difference: float
    ks_statistic: float
    ks_pvalue: float


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV feature table and validate the candidate feature schema."""
    frame = pd.read_csv(path)
    _require_columns(frame, ["dataset", *CANDIDATE_FEATURES])
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.strip()

    for feature in CANDIDATE_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    return frame


def variability_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify within-table variability without assuming a normal distribution."""
    rows = []

    for feature in CANDIDATE_FEATURES:
        values = frame[feature].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        iqr = float(q3 - q1)
        mad = float(np.median(np.abs(values - median)))

        # CV is informative for positive-valued quantities, but unstable near zero.
        cv = float(std / abs(mean)) if abs(mean) > 1e-12 else np.nan

        rows.append(
            {
                "feature": feature,
                "group": ENGINEERING_GROUPS[feature],
                "n": int(values.size),
                "mean": mean,
                "std": std,
                "median": median,
                "iqr": iqr,
                "mad": mad,
                "cv": cv,
                "near_constant": bool(
                    np.isfinite(cv) and cv < NEAR_CONSTANT_CV
                ),
            }
        )

    return pd.DataFrame(rows)


def correlation_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Pearson, Spearman and a ranked list of redundant feature pairs."""
    values = frame[CANDIDATE_FEATURES].astype(float)
    pearson = values.corr(method="pearson")
    spearman = values.corr(method="spearman")

    pairs = []
    for i, feature_a in enumerate(CANDIDATE_FEATURES):
        for feature_b in CANDIDATE_FEATURES[i + 1 :]:
            pearson_r = float(pearson.loc[feature_a, feature_b])
            spearman_r = float(spearman.loc[feature_a, feature_b])
            max_abs_r = max(abs(pearson_r), abs(spearman_r))

            if max_abs_r >= HIGH_REDUNDANCY_ABS_R:
                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pearson_r": pearson_r,
                        "spearman_r": spearman_r,
                        "max_abs_r": max_abs_r,
                        "redundancy_level": (
                            "very_high"
                            if max_abs_r >= VERY_HIGH_REDUNDANCY_ABS_R
                            else "high"
                        ),
                    }
                )

    pairs_frame = pd.DataFrame(pairs).sort_values(
        "max_abs_r", ascending=False
    ) if pairs else pd.DataFrame(
        columns=[
            "feature_a",
            "feature_b",
            "pearson_r",
            "spearman_r",
            "max_abs_r",
            "redundancy_level",
        ]
    )

    return pearson, spearman, pairs_frame.reset_index(drop=True)


def _ks_2samp(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Small dependency-free two-sample KS statistic and asymptotic p-value."""
    x = np.sort(x)
    y = np.sort(y)
    n = x.size
    m = y.size

    values = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, values, side="right") / n
    cdf_y = np.searchsorted(y, values, side="right") / m
    d = float(np.max(np.abs(cdf_x - cdf_y)))

    # Kolmogorov asymptotic approximation; adequate for screening.
    effective_n = np.sqrt(n * m / (n + m))
    if effective_n <= 0:
        return d, np.nan

    lam = (effective_n + 0.12 + 0.11 / effective_n) * d
    p = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * np.exp(-2.0 * (lam**2) * (k**2))
        p += term
        if abs(term) < 1e-10:
            break

    return d, float(np.clip(p, 0.0, 1.0))


def cross_domain_analysis(
    frame: pd.DataFrame,
    domain_a: str = "USSimAndSegm",
    domain_b: str = "BUSI",
) -> pd.DataFrame:
    """Measure how strongly each feature changes between two public domains."""
    rows: list[dict] = []

    a_frame = frame[frame["dataset"] == domain_a]
    b_frame = frame[frame["dataset"] == domain_b]

    if a_frame.empty or b_frame.empty:
        raise ValueError(
            f"Both domains are required. Found {domain_a!r}={len(a_frame)}, "
            f"{domain_b!r}={len(b_frame)}."
        )

    for feature in CANDIDATE_FEATURES:
        a = a_frame[feature].dropna().to_numpy(dtype=float)
        b = b_frame[feature].dropna().to_numpy(dtype=float)

        if a.size < 2 or b.size < 2:
            continue

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        std_a = float(np.std(a, ddof=1))
        std_b = float(np.std(b, ddof=1))
        pooled = np.sqrt(max((std_a**2 + std_b**2) / 2.0, 1e-16))
        smd = float((mean_a - mean_b) / pooled)

        ks_d, ks_p = _ks_2samp(a, b)

        rows.append(
            {
                "feature": feature,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "n_domain_a": int(a.size),
                "n_domain_b": int(b.size),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "std_a": std_a,
                "std_b": std_b,
                "standardized_mean_difference": smd,
                "abs_smd": abs(smd),
                "ks_statistic": ks_d,
                "ks_pvalue": ks_p,
                "domain_sensitive": bool(
                    abs(smd) >= DOMAIN_SHIFT_SMD or ks_d >= DOMAIN_SHIFT_KS
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "abs_smd", ascending=False
    ).reset_index(drop=True)


def engineering_relevance_table() -> pd.DataFrame:
    """Document why each candidate can be meaningful for scanner monitoring."""
    interpretations = {
        "mean_intensity": (
            "Global brightness; potentially sensitive to gain, grayscale mapping "
            "and acquisition conditions."
        ),
        "std_intensity": (
            "Global intensity dispersion; reflects contrast/heterogeneity but may "
            "be strongly redundant with percentile-based contrast."
        ),
        "min_intensity": (
            "Dark-tail behavior; potentially sensitive to clipping, background and "
            "dynamic-range handling, but often weakly specific."
        ),
        "max_intensity": (
            "Bright-tail behavior; potentially sensitive to saturation/clipping and "
            "grayscale mapping."
        ),
        "dynamic_range": (
            "Global intensity span; relevant to grayscale utilization and possible "
            "clipping, but highly dependent on image content."
        ),
        "percentile_contrast": (
            "Robust intensity spread; less sensitive to isolated extrema than "
            "min/max and useful for monitoring contrast behavior."
        ),
        "entropy": (
            "Information/distribution complexity; potentially sensitive to texture, "
            "speckle and grayscale distribution, but known to be domain/content "
            "dependent."
        ),
        "mean_gradient": (
            "Average local intensity change; proxy for structural detail and "
            "boundary strength."
        ),
        "std_gradient": (
            "Variability of local gradients; may capture changes in structural "
            "complexity and edge distribution."
        ),
        "edge_density": (
            "Proportion of detected edges; potentially useful for structural change, "
            "but strongly dependent on the edge detector and thresholds."
        ),
        "sharpness": (
            "Laplacian-based high-frequency response; plausible proxy for resolution "
            "or blur, but also affected by speckle and image content."
        ),
    }

    return pd.DataFrame(
        [
            {
                "feature": feature,
                "engineering_group": ENGINEERING_GROUPS[feature],
                "engineering_relevance": interpretations[feature],
            }
            for feature in CANDIDATE_FEATURES
        ]
    )


def build_quality_report(
    frame: pd.DataFrame,
    domain_a: str = "USSimAndSegm",
    domain_b: str = "BUSI",
) -> dict[str, pd.DataFrame]:
    """Run all four evidence layers without automatically deleting features."""
    variability = variability_analysis(frame)
    pearson, spearman, redundancy = correlation_analysis(frame)
    cross_domain = cross_domain_analysis(frame, domain_a, domain_b)
    relevance = engineering_relevance_table()

    return {
        "variability": variability,
        "pearson": pearson,
        "spearman": spearman,
        "redundancy": redundancy,
        "cross_domain": cross_domain,
        "engineering_relevance": relevance,
    }


def save_quality_report(
    report: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Persist each evidence table as a separate CSV for reproducibility."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name, table in report.items():
        table.to_csv(output / f"{name}.csv", index=False)


__all__ = [
    "CANDIDATE_FEATURES",
    "ENGINEERING_GROUPS",
    "load_feature_table",
    "variability_analysis",
    "correlation_analysis",
    "cross_domain_analysis",
    "engineering_relevance_table",
    "build_quality_report",
    "save_quality_report",
]
