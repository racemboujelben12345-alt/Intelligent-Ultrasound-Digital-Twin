"""Metadata utilities for real ultrasound acquisition campaigns.

The metadata layer is intentionally separate from image loading. It lets the
experimental campaign record device settings, session information and phase
without changing the canonical Acquisition API.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"acquisition_id"}
OPTIONAL_COLUMNS = {
    "session_id",
    "timestamp",
    "campaign_phase",
    "device_id",
    "probe_id",
    "preset",
    "frequency",
    "gain",
    "depth",
    "focus",
    "target_id",
    "operator_id",
    "notes",
    "file_path",
    "file_format",
    "tgc",
    "dynamic_range",
    "dimensions",
    "bit_depth",
    "frame_count",
    "quality_flag",
}


def load_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load acquisition metadata from a CSV keyed by acquisition_id."""
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Metadata CSV missing required columns: "
                + ", ".join(sorted(missing))
            )

        unknown = columns - REQUIRED_COLUMNS - OPTIONAL_COLUMNS
        if unknown:
            raise ValueError(
                "Unknown metadata columns: "
                + ", ".join(sorted(unknown))
            )

        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            acquisition_id = (row.get("acquisition_id") or "").strip()
            if not acquisition_id:
                raise ValueError("Every metadata row needs acquisition_id.")
            if acquisition_id in rows:
                raise ValueError(
                    f"Duplicate acquisition_id in metadata: {acquisition_id}"
                )

            clean = {
                key: value.strip()
                for key, value in row.items()
                if key and value is not None and value.strip() != ""
            }

            if "timestamp" in clean:
                try:
                    clean["timestamp"] = datetime.fromisoformat(
                        clean["timestamp"]
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid timestamp for {acquisition_id}: "
                        f"{clean['timestamp']}"
                    ) from exc

            for key in ("frequency", "gain", "depth", "focus", "dynamic_range"):
                if key in clean:
                    try:
                        clean[key] = float(clean[key])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid numeric value for {key} "
                            f"in {acquisition_id}: {clean[key]}"
                        ) from exc

            for key in ("bit_depth", "frame_count"):
                if key in clean:
                    try:
                        clean[key] = int(clean[key])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid integer value for {key} "
                            f"in {acquisition_id}: {clean[key]}"
                        ) from exc

            rows[acquisition_id] = clean

    return rows
