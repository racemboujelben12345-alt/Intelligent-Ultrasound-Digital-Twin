from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_output_dir(output_dir: Path) -> Path:
    """
    Crée le dossier de sortie s'il n'existe pas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(
    data: dict[str, Any],
    output_path: Path,
    *,
    indent: int = 2,
) -> None:
    """
    Sauvegarde un dictionnaire au format JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
            default=str,
        )


def save_csv(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Sauvegarde une collection de dictionnaires au format CSV.
    """
    import csv

    rows = list(rows)

    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_text(
    content: str,
    output_path: Path,
) -> None:
    """
    Sauvegarde un texte UTF-8.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        content,
        encoding="utf-8",
    )


def build_summary(
    *,
    project: str,
    n_acquisitions: int,
    source: str,
    baseline_size: int,
    calibration_size: int,
    test_size: int,
    latest_state: str | None = None,
    latest_distance: float | None = None,
    latest_quality: float | None = None,
) -> dict[str, Any]:
    """
    Construit un résumé standardisé du Digital Twin.
    """
    return {
        "project": project,
        "n_acquisitions": n_acquisitions,
        "source": source,
        "baseline_size": baseline_size,
        "calibration_size": calibration_size,
        "test_size": test_size,
        "latest_state": latest_state,
        "latest_mahalanobis_distance": latest_distance,
        "latest_quality_score": latest_quality,
    }