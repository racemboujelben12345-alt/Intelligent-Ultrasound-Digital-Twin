# Physics-Informed Ultrasound Model

## Purpose

This module provides the physical backbone of the Intelligent Ultrasound Digital Twin. It connects acquisition metadata to measurable image signatures without claiming calibrated acoustic measurements.

The current reference manual used during development is the GE LOGIQ 7 Advanced Reference Manual supplied with the project documentation. It contains operating-condition tables exposing the engineering vocabulary needed by the model: transducer/mode, frequency, image depth, pulse duration, PRF, focal parameters, acoustic output and velocity scale.

IMPORTANT: the LOGIQ 7 manual is a physics/engineering reference, not a SCAN A dataset and not a specification of the SCAN A system. SCAN A values must be measured and recorded during the experimental campaign.

## Core relationships

### Wavelength

    lambda = c / f

### Pulse length and axial resolution

When pulse duration PD is available:

    SPL = c * PD
    axial_resolution ~= SPL / 2

If PD is unavailable, the implementation falls back to a wavelength-scale proxy and marks the result as model-derived rather than scanner-QA measured.

### Depth and pulse-repetition timing

For conventional pulse-echo timing:

    PRF_max ~= c / (2D)

The implementation reports:

    prf_depth_margin = PRF_max / PRF

A value below 1 is treated as a metadata/physics consistency warning. It is not interpreted as proof of hardware failure because real scanners can use architecture and acquisition strategies that change the simple bound.

### Relative attenuation proxy

The image module estimates a depth-dependent intensity trend. Because display gray levels are not calibrated acoustic pressure, the result is reported only as attenuation_proxy_db_cm_mhz. It must not be interpreted as an absolute tissue attenuation coefficient or calibrated acoustic measurement.

## Physics-to-image chain

    Probe / Mode
         |
         +--> Frequency ------> wavelength
         |
         +--> Pulse duration -> SPL -> axial-resolution proxy
         |
         +--> Depth + PRF ----> timing consistency
         |
         +--> Focus ----------> expected beam/resolution behavior
         |
         +--> Output ---------> acquisition metadata context
         |
         v
    Ultrasound image
         |
         +--> depth profile
         +--> uniformity
         +--> near/far response
         +--> sharpness / gradients
         +--> speckle / texture
         |
         v
    Digital Signature
         |
         v
    Statistical + AI evidence
         |
         v
    Unified Twin State

## Experimental requirements

For each real SCAN A acquisition, the campaign should record, whenever the device exposes the value:

- acquisition ID
- timestamp/session
- device ID
- probe ID
- imaging mode/preset
- frequency
- depth
- focus
- gain
- TGC
- dynamic range
- pulse duration, if available
- PRF, if available
- output power, if available
- velocity scale, if relevant to the selected mode
- target/phantom ID
- image file and format
- operator/session information

The metadata contract is deliberately extensible because the exact controls available on SCAN A must be established experimentally.

## Engineering interpretation

The physics layer is an evidence source, not the final diagnosis.

A strong Twin state should combine:

    physics consistency
    + image quality
    + statistical deviation
    + AI evidence
    + temporal drift

A simulated degradation remains a controlled software scenario. It is not evidence that the physical scanner has failed.

## Validation hierarchy

1. Unit/software validation.
2. Physics-model numerical checks.
3. Reproducible controlled simulations.
4. Public ultrasound benchmark.
5. Repeated SCAN A acquisitions.
6. Phantom-based performance measurements.
7. Authorized controlled perturbations, only where safe and documented.

Physical conclusions about SCAN A require levels 5-7; software V&V alone is not physical validation.