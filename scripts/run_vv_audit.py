"""Command-line V&V audit for the Intelligent Ultrasound Digital Twin."""

from __future__ import annotations

import argparse

import numpy as np

from src.acquisition.loader import get_acquisitions
from src.simulation.degradation import DegradationType
from src.validation.audit import (
    audit_acquisitions,
    verify_reproducibility,
    verify_severity_response,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ultrasound Digital Twin V&V audit.")
    parser.add_argument("--n", type=int, default=10, help="Number of acquisitions to audit.")
    args = parser.parse_args()

    acquisitions = get_acquisitions(
        prefer_experimental=True,
        minimum_experimental=args.n,
        prefer_public=True,
        minimum_public=args.n,
        demo_size=max(args.n, 10),
    )[:args.n]

    report = audit_acquisitions(acquisitions)
    print(report.summary())

    image = acquisitions[0].image
    checks = [
        verify_reproducibility(
            image,
            DegradationType.GAUSSIAN_NOISE,
            severity=0.5,
            seed=42,
        ),
        verify_severity_response(
            image,
            DegradationType.BLUR,
            severities=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            seed=42,
        ),
    ]

    for check in (*report.checks, *checks):
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.details}")

    return 0 if report.passed and all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
