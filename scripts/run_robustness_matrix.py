#!/usr/bin/env python3
"""Run the noise-aware robustness matrix."""
import argparse
import json
import numpy as np
from src.validation.robustness_matrix import run_robustness_matrix, summarize_robustness

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--repeats', type=int, default=20)
args = parser.parse_args()
x = np.linspace(0.05, 0.95, 128)
image = np.tile(x, (128, 1))
rows = run_robustness_matrix(image, repeats=args.repeats, seed=args.seed)
print(json.dumps({'cases': len(rows), 'summary': summarize_robustness(rows)}, indent=2, sort_keys=True))