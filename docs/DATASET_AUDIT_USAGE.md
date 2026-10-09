# Reproducible dataset audit

Run from the repository root after installing `requirements.txt`:

```bash
python scripts/audit_dataset.py
```

The script scans the configured local folders:

- `data/raw/experimental_ultrasound/`
- `data/raw/public_ultrasound/`
- `data/raw/demo_simulated/`

It writes `outputs/data_audit/data_audit.json` (machine-readable inventory) and
`outputs/data_audit/data_audit.md` (summary). These folders are ignored by Git;
image files and generated reports are not added to the repository.

To include an acquisition metadata CSV:

```bash
python scripts/audit_dataset.py --metadata data/path/to/metadata.csv
```

To inspect a different local project/data root:

```bash
python scripts/audit_dataset.py --root /path/to/project --metadata /path/to/metadata.csv
```

## Interpretation rules

- Counts refer only to files found locally when the command runs. Missing source
  folders are reported; no expected counts are filled in automatically.
- SHA-256 matches identify exact file duplicates.
- Perceptual-hash matches are only candidates for manual review. Similar images
  may be distinct acquisitions, and transformed derivatives may not be detected.
- Metadata missingness is reported only for columns present in the supplied CSV.
  The script does not invent metadata or infer a missing timestamp/session.
- This inventory does not prove independence of sessions, temporal ordering,
  train/test safety, model accuracy, clinical validity, or physical repeatability.
  Those require protocol-aware metadata and separate evaluations.
- The script does not change, rename, or delete image files.

The first run is an inventory step, not a pass/fail certification of the dataset.
Review the JSON findings and resolve unexplained corrupt files, duplicates,
metadata gaps, and lineage questions before training or benchmarking.
