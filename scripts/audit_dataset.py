"""Reproducible, source-aware inventory of ultrasound image datasets.

This script reports local files only. It does not train models or establish
physical scanner performance. Images remain in place; only metadata summaries
and file hashes are written to outputs/data_audit/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SOURCE_DIRS = {
    "experimental": Path("data/raw/experimental_ultrasound"),
    "public": Path("data/raw/public_ultrasound"),
    "simulated": Path("data/raw/demo_simulated"),
}
METADATA_FIELDS = (
    "session_id", "timestamp", "device_id", "probe_id", "mode",
    "frequency", "gain", "depth", "focus", "tgc", "target_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: np.ndarray) -> int:
    """Simple DCT perceptual hash; use as a review cue, not proof of duplication."""
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8].flatten()
    median = float(np.median(block[1:]))
    bits = block > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def inspect_sources(root: Path) -> dict:
    records = []
    missing_sources = []
    for source, relative_dir in SOURCE_DIRS.items():
        folder = root / relative_dir
        if not folder.exists():
            missing_sources.append(source)
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            record = {
                "source": source,
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "readable": False,
                "width": None,
                "height": None,
                "dtype": None,
                "finite": None,
                "perceptual_hash": None,
                "error": None,
            }
            try:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.size == 0:
                    raise ValueError("OpenCV could not decode image or image is empty")
                record.update({
                    "readable": True,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "dtype": str(image.dtype),
                    "finite": bool(np.isfinite(image).all()),
                    "perceptual_hash": f"{perceptual_hash(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 and image.shape[2] >= 3 else image):016x}",
                })
                if not record["finite"]:
                    record["error"] = "image contains non-finite values"
            except Exception as exc:
                record["error"] = str(exc)
            records.append(record)

    exact_groups = defaultdict(list)
    for rec in records:
        exact_groups[rec["sha256"]].append(rec["relative_path"])
    exact_duplicates = [
        {"sha256": digest, "files": files}
        for digest, files in exact_groups.items() if len(files) > 1
    ]

    hashes = [(r["relative_path"], int(r["perceptual_hash"], 16))
              for r in records if r["perceptual_hash"] is not None and r["readable"]]
    near_duplicates = []
    # Keep pairs with a small Hamming distance; this is a candidate list for manual review.
    for i, (path_a, hash_a) in enumerate(hashes):
        for path_b, hash_b in hashes[i + 1:]:
            distance = hamming_distance(hash_a, hash_b)
            if distance <= 4:
                near_duplicates.append({
                    "file_a": path_a, "file_b": path_b,
                    "phash_hamming_distance": distance,
                    "review_required": True,
                })

    by_source = {}
    for source in SOURCE_DIRS:
        subset = [r for r in records if r["source"] == source]
        dims = Counter(f'{r["width"]}x{r["height"]}' for r in subset if r["readable"])
        by_source[source] = {
            "image_files_found": len(subset),
            "readable": sum(r["readable"] for r in subset),
            "unreadable": sum(not r["readable"] for r in subset),
            "non_finite": sum(r["finite"] is False for r in subset),
            "dimensions": dict(sorted(dims.items())),
        }

    return {
        "root": str(root),
        "missing_source_directories": missing_sources,
        "source_summary": by_source,
        "total_images": len(records),
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_candidates": near_duplicates,
        "files": records,
        "limitations": [
            "Only files in the configured local directories were inventoried.",
            "Perceptual-hash matches are candidates for manual review, not confirmed duplicates.",
            "Image files alone do not establish acquisition/session independence or temporal order.",
            "No model training, performance evaluation, or physical SCAN A validation was performed.",
        ],
    }


def inspect_metadata(path: Path | None) -> dict:
    if path is None:
        return {"provided": False, "rows": None, "missingness": {},
                "warning": "No metadata CSV supplied; metadata completeness was not assessed."}
    import csv
    if not path.exists():
        return {"provided": True, "path": str(path), "rows": 0,
                "missingness": {}, "error": "Metadata file does not exist."}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    missingness = {}
    for field in METADATA_FIELDS:
        if field in fields:
            missingness[field] = {
                "missing_count": sum(not (row.get(field) or "").strip() for row in rows),
                "total_rows": len(rows),
            }
    ids = [(row.get("acquisition_id") or "").strip() for row in rows]
    duplicate_ids = sorted({value for value in ids if value and ids.count(value) > 1})
    return {
        "provided": True, "path": str(path), "rows": len(rows),
        "columns": fields, "missingness": missingness,
        "missing_acquisition_id_rows": sum(not value for value in ids),
        "duplicate_acquisition_ids": duplicate_ids,
    }


def build_report(root: Path, metadata_path: Path | None = None) -> dict:
    report = inspect_sources(root)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["metadata"] = inspect_metadata(metadata_path)
    return report


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_audit.json"
    md_path = output_dir / "data_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = report["source_summary"]
    lines = [
        "# Local ultrasound dataset audit", "",
        f'- Generated (UTC): {report["generated_at_utc"]}',
        f'- Total supported image files: {report["total_images"]}', "",
        "## Counts by provenance", "",
        "| Source | Files | Readable | Unreadable | Non-finite |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, values in summary.items():
        lines.append(
            f'| {source} | {values["image_files_found"]} | {values["readable"]} | '
            f'{values["unreadable"]} | {values["non_finite"]} |'
        )
    lines += ["", "## Integrity findings", "",
              f'- Missing source directories: {", ".join(report["missing_source_directories"]) or "none"}',
              f'- Exact duplicate groups: {len(report["exact_duplicate_groups"])}',
              f'- Near-duplicate candidates (manual review required): {len(report["near_duplicate_candidates"])}',
              f'- Metadata supplied: {report["metadata"]["provided"]}', "",
              "## Limitations", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository/data root (default: repository root)")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Optional acquisition metadata CSV")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: ROOT/outputs/data_audit)")
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = args.metadata
    if metadata is not None and not metadata.is_absolute():
        metadata = root / metadata
    output_dir = args.output_dir or (root / "outputs" / "data_audit")
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report = build_report(root, metadata)
    json_path, md_path = write_report(report, output_dir)
    print(f'Inventoried {report["total_images"]} supported image files.')
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print("Counts describe local files only; no model or physical-device validation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
