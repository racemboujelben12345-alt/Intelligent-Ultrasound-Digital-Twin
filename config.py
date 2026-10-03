from pathlib import Path


# ============================================================
# SCAN A DIGITAL TWIN V2
# Central configuration
# ============================================================

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BASELINE_DIR = DATA_DIR / "baseline"
SIMULATIONS_DIR = DATA_DIR / "simulations"

OUTPUT_DIR = ROOT / "outputs"


# ------------------------------------------------------------
# Data sources
# ------------------------------------------------------------

SCAN_A_DIR = RAW_DIR / "scan_a"

PUBLIC_DATA_DIR = RAW_DIR / "public_ultrasound"

DEMO_DATA_DIR = RAW_DIR / "demo_simulated"


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42


# ------------------------------------------------------------
# Image processing
# ------------------------------------------------------------

# Important:
# This is NOT the physical resolution of SCAN A.
# It is only an optional computational representation.

ANALYSIS_SIZE = 128


# ------------------------------------------------------------
# Baseline configuration
# ------------------------------------------------------------

BASELINE_SIZE = 30

HOLDOUT_SIZE = 10


# ------------------------------------------------------------
# Temporal monitoring
# ------------------------------------------------------------

PERSISTENCE = 3

EWMA_LAMBDA = 0.20
EWMA_LIMIT = 3.0

CUSUM_K = 0.5
CUSUM_H = 5.0

TREND_WINDOW = 10


# ------------------------------------------------------------
# Statistical thresholds
# ------------------------------------------------------------

ALPHA_EARLY = 0.95
ALPHA_SIGNIFICANT = 0.99
ALPHA_CRITICAL = 0.999


# ------------------------------------------------------------
# Runtime mode
# ------------------------------------------------------------

RUN_MODE = "AUTO"


# Possible values:
#
# "REAL_SCAN_A"
# "PUBLIC_REFERENCE"
# "SIMULATION"
# "AUTO"