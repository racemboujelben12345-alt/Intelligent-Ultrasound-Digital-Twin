from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BASELINE_DIR = DATA_DIR / "baseline"
SIMULATIONS_DIR = DATA_DIR / "simulations"
OUTPUT_DIR = ROOT / "outputs"

EXPERIMENTAL_DATA_DIR = RAW_DIR / "experimental_ultrasound"
PUBLIC_DATA_DIR = RAW_DIR / "public_ultrasound"
DEMO_DATA_DIR = RAW_DIR / "demo_simulated"

# Backward-compatible path alias retained for legacy callers.
# New code should use EXPERIMENTAL_DATA_DIR.
SCAN_A_DIR = EXPERIMENTAL_DATA_DIR

RANDOM_SEED = 42
ANALYSIS_SIZE = 128

BASELINE_SIZE = 30
HOLDOUT_SIZE = 10

PERSISTENCE = 3
EWMA_LAMBDA = 0.20
EWMA_LIMIT = 3.0
CUSUM_K = 0.5
CUSUM_H = 5.0
TREND_WINDOW = 10

ALPHA_EARLY = 0.95
ALPHA_SIGNIFICANT = 0.99
ALPHA_CRITICAL = 0.999

RUN_MODE = "AUTO"

# Acquisition source policy: AUTO | EXPERIMENTAL | PUBLIC | SIMULATED.
# AUTO may fall back only when the preferred source is unavailable.
ACQUISITION_SOURCE_MODE = "AUTO"

PROJECT_NAME = "Intelligent Ultrasound Digital Twin"
PROJECT_VERSION = "4.0"
SIGNATURE_VERSION = "2.0"
PIPELINE_SCHEMA_VERSION = "1.0"

MIN_TOTAL_ACQUISITIONS = BASELINE_SIZE + HOLDOUT_SIZE + 1
MIN_BASELINE_ACQUISITIONS = BASELINE_SIZE
MIN_HOLDOUT_ACQUISITIONS = HOLDOUT_SIZE
