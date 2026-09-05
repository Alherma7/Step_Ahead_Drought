"""Paths, column names, and constants shared across the pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = DATA_RAW_DIR / "Train.csv"
TEST_PATH = DATA_RAW_DIR / "Test.csv"
SAMPLE_SUBMISSION_PATH = DATA_RAW_DIR / "SampleSubmission.csv"

ID_COL = "ID"
TIME_COL = "time"
LAT_COL = "lat"
LON_COL = "lon"
TARGET_COL = "target"
TWS_COL = "TWS_t"
TWS_MASKED_COL = "TWS_t_masked"

# Columns present in both Train.csv and Test.csv that may be used as model inputs.
BASE_FEATURE_COLS = ["TWS_t", "month_sin", "month_cos"]
OPTIONAL_FEATURE_COLS = [
    "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t",
]

# Spatial neighbourhood features (src/features.py::add_neighbourhood_features).
NEIGHBOUR_RADIUS_KM = 500.0
NEIGHBOUR_MEAN_COL = "tws_neighbour_mean"
LOCAL_DEVIATION_COL = "tws_local_deviation"
NEIGHBOURHOOD_FEATURE_COLS = [NEIGHBOUR_MEAN_COL, LOCAL_DEVIATION_COL]

# Seasonal climatology features (src/features.py::add_seasonal_climatology_features).
# CLIMATOLOGY_STD_COL is an intermediate (used to compute the deviation) and is not
# itself a model feature - only the mean and the standardised deviation are.
CLIMATOLOGY_MEAN_COL = "tws_climatology_mean"
CLIMATOLOGY_STD_COL = "tws_climatology_std"
CLIMATOLOGY_DEVIATION_COL = "tws_climatology_deviation"
CLIMATOLOGY_FEATURE_COLS = [CLIMATOLOGY_MEAN_COL, CLIMATOLOGY_DEVIATION_COL]

# Anchor-age feature (src/features.py::compute_anchor_age). Only learnable if the
# model is trained on masking-augmented data (src/evaluate.py::
# augment_with_simulated_masking) - see src/train.py's final-fit step.
ANCHOR_AGE_COL = "months_since_anchor"

# SPEI_12 prior-reading feature (src/features.py::compute_prior_reading), computed
# unconditionally inside build_all_features. Never masked in Test.csv, so unlike
# ANCHOR_AGE_COL it needs no masking-augmentation to be learnable.
SPEI_12_COL = "SPEI_12_t"
SPEI12_PRIOR_VALUE_COL = "spei12_prior_value"
SPEI12_PRIOR_AGE_COL = "spei12_prior_age"
SPEI12_DYNAMICS_FEATURE_COLS = [SPEI12_PRIOR_VALUE_COL, SPEI12_PRIOR_AGE_COL]

# Zindi clarification (docs/chats/Neighbouring cells' past TWS - permitted or not.txt,
# 19 Aug): raw coordinates, cell IDs and coordinate-derived encodings must never be fed
# to the model as predictors. lat/lon may only be used as a lookup key (own-cell history,
# neighbourhood aggregation, sampling external gridded covariates).
FORBIDDEN_MODEL_FEATURES = {LAT_COL, LON_COL, ID_COL}

RANDOM_STATE = 42
