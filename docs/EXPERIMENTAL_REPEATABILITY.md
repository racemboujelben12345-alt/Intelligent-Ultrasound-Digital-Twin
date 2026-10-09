# Experimental repeatability: protocol and interpretation

## Purpose

The existing estimate_repeatability() routine injects seeded Gaussian noise into
one image. It estimates simulated perturbation sensitivity; it is not a measurement
of physical ultrasound acquisition repeatability.

The separate analyze_experimental_repeatability() routine summarizes feature
variability from a sequence of observed images. It does not generate replacement
images or inject noise.

## Minimum acquisition record

For every image, preserve a unique acquisition ID, timestamp, session ID, scanner
and probe identifiers where available, imaging mode, depth, gain, frequency,
focus/TGC settings where available, phantom/reference identity, operator if known,
and the original unmodified file. Record unavailable metadata as unavailable;
do not infer or fabricate it.

## Suggested protocol

1. Use an appropriate QA phantom or another documented stable reference.
2. Acquire at least three images under comparable conditions; more observations
   and multiple sessions are preferable.
3. Preserve acquisition settings and probe placement as consistently as possible.
4. If studying session variability, collect at least two acquisitions per session
   across at least two sessions.
5. Predefine exclusions and document repositioning or setting changes.
6. Report per-feature means, sample standard deviations and coefficients of
   variation. Inspect features near zero carefully because their CV can be unstable.
7. Keep images from the same session or related lineage together when splitting
   data for model evaluation.

## What the outputs mean

- Per-feature SD/CV describe variability in the supplied image set.
- Within-session SD and the range of session means are descriptive summaries,
  not a variance-components model.
- Results depend on phantom, settings, probe, operator, image export and
  preprocessing.
- Image-derived signatures alone cannot identify the physical cause of a change.
- Synthetic degradation tests, public-data benchmarks and experimental
  repeatability results must be reported separately.

A successful unit test verifies software behavior only. A small set of repeated
images does not establish general scanner performance, diagnostic validity, a
hardware fault, or a universal alarm threshold.
