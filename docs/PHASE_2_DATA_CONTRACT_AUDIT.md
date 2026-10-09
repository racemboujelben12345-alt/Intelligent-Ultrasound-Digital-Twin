# Phase 2 — Data contract and dataset audit

**Status:** static repository audit; no local dataset was available to inspect through the repository tree. This document does not report observed image counts or data-quality metrics.

## 1. Findings from the repository

| Area | Current implementation | Finding / evidence gap |
|---|---|---|
| Dataset files | Configuration points to `data/raw/experimental_ultrasound`, `data/raw/public_ultrasound`, and `data/raw/demo_simulated`. | The tracked Git tree contains no `data/` directory. Actual image counts, duplicates, corruption, class/family distribution, and metadata completeness cannot be reported from the repository alone. |
| Source separation | `AcquisitionLoader` assigns `experimental`, `public`, and `simulated` provenance; public and synthetic demo images are marked as non-SCAN-A evidence. | Good design intent. Confirm source labels survive every downstream artifact and dashboard export. |
| Image handling | Loader reads grayscale images, converts values to float32 in [0,1], and by default resizes to 128×128. | Resizing changes spatial sampling and can affect sharpness, edge density, entropy, and texture features. Preserve original dimensions/checksum and record preprocessing version before comparing acquisitions. |
| Metadata | CSV loader requires `acquisition_id`, rejects duplicate IDs/unknown columns, parses ISO-like timestamps and numeric settings. Most device fields are optional. | A syntactically valid metadata CSV can still be scientifically incomplete. The real campaign should require the fields needed for its protocol, such as session/time, device/probe, preset, frequency, gain, depth and acquisition target where applicable. |
| Simulation lineage | Controlled degradation records parent ID, scenario, severity, seed and version. Audit helpers can resolve lineage roots and partition complete families. | Run lineage checks on the actual full manifest. Missing parents are represented by parent IDs; verify those roots correspond to known original acquisitions or explicitly documented external parents. |
| Generic split helper | `src/validation/splits.py::split_dataset` preserves input order and slices arrays into training, calibration and test partitions. | It receives image arrays only, so it cannot itself verify source, parent lineage, session identity or duplicate content. It is unsafe for derivative-rich datasets unless callers enforce these constraints before calling it. |
| Lineage split helper | `partition_by_lineage` groups acquisitions by root and checks disjoint families. | It sorts root IDs deterministically, not by timestamp. It is suitable for family separation but is not, by itself, a temporal train/test split. |
| Forecast evaluator | Rolling-origin evaluator compares predictions against persistence on identical forecast origins and targets. | It accepts a single ordered numeric series and explicitly leaves session/group and data-quality checks to its caller. It does not prove that an appropriate SCAN A time series exists or that model performance has been measured. |

## 2. Minimum reproducible data audit

Before training or reporting performance, run a manifest-based audit and save the report with a timestamp and code version. At minimum, report:

1. Number of images by source, dataset, session and device when known.
2. File readability, dimensions, channel mode, bit depth and missing/invalid metadata.
3. Duplicate file hashes and likely duplicate images; distinguish exact duplicates from transformed derivatives.
4. Unique acquisition IDs and uniqueness/consistency of timestamps within each session.
5. Missing parent IDs, invalid lineage chains, cycles, and children whose parent is absent.
6. Counts of independent lineage families per proposed partition.
7. Preprocessing and feature version for every derived artifact.
8. Explicit exclusions and reasons; never silently drop malformed records.
9. A machine-readable manifest plus human-readable summary under `outputs/data_audit/`.

Do not publish guessed counts. If data are external, record the source, version, retrieval date and local manifest/hash without committing sensitive or restricted images.

## 3. Partition policy

Use different partition strategies for different scientific questions:

- **Baseline/reference fitting:** nominal acquisitions only, from the designated reference population.
- **Threshold calibration:** a separate nominal calibration population; do not reuse test outcomes to tune thresholds.
- **Robustness testing:** keep each original acquisition and all its digital derivatives within the same information-family partition.
- **Temporal prediction:** use real, ordered acquisitions from the same meaningful sequence/session/domain where justified; train on earlier observations and evaluate on later observations. Do not create a time series by sorting unrelated public images.
- **Final test:** lock the partition before tuning; do not repeatedly adjust model or threshold based on its results.

If there are too few independent sessions or lineage families, report that limitation rather than pretending image-level random splitting creates independent evidence.

## 4. Acceptance criteria for Phase 2

Phase 2 is complete only when:

- [ ] The data source and version are documented for every file family.
- [ ] A reproducible manifest and source-aware counts are generated from the actual files.
- [ ] Duplicate and lineage audits have no unexplained failures.
- [ ] Metadata missingness is quantified and justified by protocol.
- [ ] Train/calibration/test assignment is reproducible and has no parent-family overlap.
- [ ] Temporal evaluation is restricted to genuinely ordered, comparable acquisitions.
- [ ] Reports distinguish public, simulated and experimental evidence.
- [ ] The exact audit command and output location are documented.

## Evidence boundary

This is a static assessment of repository code and configuration. No real dataset was inventoried in this change, no model was trained, and no SCAN A experimental validation is claimed.
