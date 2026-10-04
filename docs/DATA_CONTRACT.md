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
| image_format | PNG/DICOM/etc. | representation |
| preprocessing_version | vX.Y | preprocessing version |
| feature_version | vX.Y | feature version |

## Rules
Public data remain labelled public. Simulated data retain parent ID, degradation type, severity, random seed and simulation version. Experimental data preserve original metadata and checksum when possible. A named device is a validation layer, not the identity of the whole project.

## Scientific purpose
This contract prevents leakage and invalid claims when public, synthetic and experimental data coexist.
