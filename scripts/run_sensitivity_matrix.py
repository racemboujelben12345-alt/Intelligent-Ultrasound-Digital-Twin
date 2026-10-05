#!/usr/bin/env python3
"""Run a controlled sensitivity matrix and print a compact V&V summary."""

import argparse
import json
import numpy as np

from src.simulation.degradation import DegradationType
from src.validation.sensitivity_matrix import run_sensitivity_matrix, summarize_sensitivity_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    x = np.linspace(0.05, 0.95, 128)
    image = np.tile(x, (128, 1))
    rows = run_sensitivity_matrix(image, seed=args.seed)
    print(json.dumps({
        'scenarios': len(tuple(DegradationType)),
        'severities_per_scenario': 5,
        'total_cases': len(rows),
        'summary': summarize_sensitivity_matrix(rows),
    }, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()