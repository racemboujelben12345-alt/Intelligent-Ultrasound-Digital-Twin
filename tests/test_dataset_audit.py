from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_dataset import build_report, inspect_metadata, inspect_sources, write_report


def test_audit_counts_sources_and_detects_exact_duplicates(tmp_path: Path):
    experimental = tmp_path / "data/raw/experimental_ultrasound"
    public = tmp_path / "data/raw/public_ultrasound"
    simulated = tmp_path / "data/raw/demo_simulated"
    for folder in (experimental, public, simulated):
        folder.mkdir(parents=True)

    image = np.zeros((24, 32), dtype=np.uint8)
    image[5:12, 8:19] = 180
    original = experimental / "a.png"
    assert cv2.imwrite(str(original), image)
    shutil.copyfile(original, public / "copy.png")
    (simulated / "broken.png").write_text("not an image", encoding="utf-8")

    report = inspect_sources(tmp_path)

    assert report["total_images"] == 3
    assert report["source_summary"]["experimental"]["readable"] == 1
    assert report["source_summary"]["public"]["readable"] == 1
    assert report["source_summary"]["simulated"]["unreadable"] == 1
    assert len(report["exact_duplicate_groups"]) == 1
    assert report["exact_duplicate_groups"][0]["files"] == [
        "data/raw/experimental_ultrasound/a.png",
        "data/raw/public_ultrasound/copy.png",
    ]


def test_audit_reports_missing_source_directories_without_inventing_counts(tmp_path: Path):
    report = inspect_sources(tmp_path)
    assert report["total_images"] == 0
    assert set(report["missing_source_directories"]) == {
        "experimental", "public", "simulated"
    }
    assert all(v["image_files_found"] == 0 for v in report["source_summary"].values())


def test_metadata_missingness_and_duplicate_ids_are_reported(tmp_path: Path):
    path = tmp_path / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["acquisition_id", "session_id", "gain"]
        )
        writer.writeheader()
        writer.writerow({"acquisition_id": "a1", "session_id": "s1", "gain": "40"})
        writer.writerow({"acquisition_id": "a1", "session_id": "", "gain": ""})

    result = inspect_metadata(path)
    assert result["rows"] == 2
    assert result["duplicate_acquisition_ids"] == ["a1"]
    assert result["missingness"]["session_id"]["missing_count"] == 1
    assert result["missingness"]["gain"]["missing_count"] == 1


def test_report_writes_machine_and_human_readable_outputs(tmp_path: Path):
    (tmp_path / "data/raw/experimental_ultrasound").mkdir(parents=True)
    (tmp_path / "data/raw/public_ultrasound").mkdir(parents=True)
    (tmp_path / "data/raw/demo_simulated").mkdir(parents=True)

    report = build_report(tmp_path)
    json_path, md_path = write_report(report, tmp_path / "outputs/data_audit")

    assert json_path.exists()
    assert md_path.exists()
    assert '"total_images": 0' in json_path.read_text(encoding="utf-8")
    assert "Counts by provenance" in md_path.read_text(encoding="utf-8")
