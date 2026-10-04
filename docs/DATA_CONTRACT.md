# Ultrasound Data Contract

Every acquisition entering the Digital Twin pipeline has declared provenance.

| Field | Example | Meaning |
|---|---|---|
| source_type | public / experimental / simulated | origin |
| sample_id | US_000123 | immutable ID |
| timestamp | ISO-8601 | acquisition time |
| session_id | S001 | repeated-observation group |
| device_id | optional | physical device |
| probe_id | optional | probe |
| preset | optional | ultrasound preset |
| frequency | optional | frequency |
| gain | optional | gain |
| depth | optional | imaging depth |
| focus | optional | focus |
| image_format | PNG/JPEG/TIFF (DICOM requires a dedicated adapter) | representation |
| preprocessing_version | vX.Y | preprocessing version |
| feature_version | vX.Y | feature version |

## Rules
Public data remain labelled public. Simulated data retain parent ID, degradation type, severity, random seed and simulation version. Experimental data preserve original metadata and checksum when possible. A named device is a validation layer, not the identity of the whole project.

## Scientific purpose
This contract prevents leakage and invalid claims when public, synthetic and experimental data coexist.


## 6. Simulation lineage

Every controlled synthetic derivative must retain:

- `parent_acquisition_id`: identifier of the source acquisition;
- `simulation_scenario`: degradation family;
- `simulation_severity`: normalized severity in `[0,1]`;
- `simulation_seed`: reproducibility seed;
- `simulation_version`: version of the degradation engine.

The exception is a synthetic root/demo acquisition, which uses the explicit lineage marker
`synthetic_root`. This is not presented as experimental evidence.

## 7. Source separation

The canonical source categories are:

| Category | Meaning | Permitted for device-specific claims? |
|---|---|---|
| `experimental` | Real ultrasound acquisition | Yes, subject to protocol and validation |
| `public` | Public reference dataset | No, not by itself |
| `simulated` | Controlled digital perturbation | No, used for sensitivity/robustness experiments |
| `scan_a` | Legacy compatibility label | Treated as experimental only when backed by real acquisition metadata |

Metrics must remain source-aware. Public, simulated and experimental results must never be silently pooled into one validation score.
