"""Final engineering V&V audit for the Intelligent Ultrasound Digital Twin."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from src.acquisition.loader import get_acquisitions
from src.simulation.degradation import DegradationType
from src.validation.audit import audit_acquisitions, verify_reproducibility, verify_severity_response
from src.validation.robustness_matrix import run_robustness_matrix, summarize_robustness

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('outputs/vv'))
    args = parser.parse_args()
    acquisitions = get_acquisitions(prefer_experimental=True, minimum_experimental=args.n,
                                    prefer_public=True, minimum_public=args.n,
                                    demo_size=max(args.n, 10), seed=args.seed)[:args.n]
    if not acquisitions:
        raise RuntimeError('No acquisitions available for final V&V audit.')
    report = audit_acquisitions(acquisitions)
    image = acquisitions[0].image
    checks = [verify_reproducibility(image, DegradationType.GAUSSIAN_NOISE, severity=0.5, seed=args.seed),
              verify_severity_response(image, DegradationType.BLUR, severities=(0.0,0.2,0.4,0.6,0.8,1.0), seed=args.seed)]
    robustness = run_robustness_matrix(image, repeats=args.repeats, seed=args.seed)
    summary = {'audit_passed': bool(report.passed),
               'checks': [{'name': c.name, 'passed': bool(c.passed), 'details': c.details} for c in (*report.checks, *checks)],
               'robustness': summarize_robustness(robustness),
               'n_cases': len(robustness), 'repeats': args.repeats, 'seed': args.seed,
               'scientific_boundary': 'Software/simulation V&V only; not proof of a physical SCAN A fault.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'final_vv_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    (args.output / 'robustness_matrix.json').write_text(json.dumps(robustness, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report.passed and all(c.passed for c in checks) else 1

if __name__ == '__main__':
    raise SystemExit(main())