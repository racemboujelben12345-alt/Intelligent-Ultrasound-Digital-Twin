# Final V&V and Robustness Workflow

## One-command software and simulation audit

```bash
python scripts/run_final_vv.py --n 10 --repeats 20 --seed 42
```

The audit combines:
- software/data-contract checks;
- deterministic controlled-degradation checks;
- severity-response verification;
- a **simulated perturbation noise floor** from seeded Gaussian-noise variants;
- sensitivity/robustness summaries;
- JSON artifacts under `outputs/vv/`.

The repeatability calculation currently used by this command perturbs an image digitally. It does **not** measure physical acquisition repeatability.

## Separate experimental repeatability analysis

When repeated images from a real, documented acquisition protocol become available, use
`src.validation.experimental_repeatability.analyze_experimental_repeatability()`.
See `docs/EXPERIMENTAL_REPEATABILITY.md` for the required context, protocol and interpretation limits.

Do not label synthetic-noise results as measured scanner repeatability. Report simulated, public-data and experimental results separately.

## Interpretation

`robustness_margin_l2 > 0` means the simulated signature change exceeds the selected **simulated** perturbation threshold. It is evidence about the response of the implemented pipeline to the chosen digital perturbations; it is not a calibrated physical alarm threshold.

## Scientific boundary

This audit verifies software behavior and controlled simulations. It does not prove a hardware fault, scanner degradation, clinical effect, or device-specific mechanism. Device-specific conclusions require traceable repeated acquisitions, suitable phantom QA, probe-specific characterization, documented settings and authorized controlled experiments.

## Final architecture

```text
Ultrasound acquisition
        ↓
Provenance + metadata
        ↓
Digital Signature + physics-informed features
        ↓
Statistical baseline
        ↓
Statistical anomaly evidence
        ↓
AI ensemble evidence
        ↓
Evidence fusion + uncertainty
        ↓
Unified Twin State
        ↓
Temporal drift / trend
        ↓
Software/simulation robustness audit
        ↓
Experimental repeatability (when real repeated acquisitions exist)
        ↓
Engineering report / dashboard
```
