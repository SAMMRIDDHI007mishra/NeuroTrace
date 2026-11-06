"""
NeuroTrace Configuration Module
===============================
Central configuration for data paths, clinical features, model hyperparameters,
and random seed management across the pipeline.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"

# Create directories if they do not exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, FIGURES_DIR, DOCS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42
CV_FOLDS = 5

# -------------------------------------------------------------
# Parkinson's Disease Feature Configuration
# -------------------------------------------------------------
PARKINSONS_RAW_FILE = RAW_DATA_DIR / "parkinsons.data"
PARKINSONS_PROCESSED_FILE = PROCESSED_DATA_DIR / "parkinsons_processed.csv"
PARKINSONS_TARGET = "status"

# Acoustic Feature Subgroups
PARKINSONS_FREQ_FEATURES = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]
PARKINSONS_JITTER_FEATURES = [
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"
]
PARKINSONS_SHIMMER_FEATURES = [
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"
]
PARKINSONS_NOISE_FEATURES = ["NHR", "HNR"]
PARKINSONS_NONLINEAR_FEATURES = ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]

PARKINSONS_ALL_FEATURES = (
    PARKINSONS_FREQ_FEATURES
    + PARKINSONS_JITTER_FEATURES
    + PARKINSONS_SHIMMER_FEATURES
    + PARKINSONS_NOISE_FEATURES
    + PARKINSONS_NONLINEAR_FEATURES
)

# -------------------------------------------------------------
# Alzheimer's Disease (OASIS) Feature Configuration
# -------------------------------------------------------------
ALZHEIMERS_RAW_FILE = RAW_DATA_DIR / "oasis_longitudinal.csv"
ALZHEIMERS_PROCESSED_FILE = PROCESSED_DATA_DIR / "oasis_processed.csv"
ALZHEIMERS_TARGET = "Dementia"

ALZHEIMERS_DEMOGRAPHIC_FEATURES = ["M/F", "Age", "EDUC", "SES"]
ALZHEIMERS_COGNITIVE_FEATURES = ["MMSE", "CDR"]
ALZHEIMERS_VOLUMETRIC_FEATURES = ["eTIV", "nWBV", "ASF"]

ALZHEIMERS_ALL_FEATURES = [
    "Gender_Encoded", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"
]

# Features for non-CDR model (pure neuroimaging + cognitive screening)
ALZHEIMERS_SCREENING_FEATURES = [
    "Gender_Encoded", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"
]

# -------------------------------------------------------------
# Model File Paths
# -------------------------------------------------------------
PARKINSONS_MODEL_PATH = SAVED_MODELS_DIR / "parkinsons_ensemble.joblib"
PARKINSONS_SCALER_PATH = SAVED_MODELS_DIR / "parkinsons_scaler.joblib"
PARKINSONS_METADATA_PATH = SAVED_MODELS_DIR / "parkinsons_metadata.json"

ALZHEIMERS_MODEL_PATH = SAVED_MODELS_DIR / "alzheimers_ensemble.joblib"
ALZHEIMERS_SCALER_PATH = SAVED_MODELS_DIR / "alzheimers_scaler.joblib"
ALZHEIMERS_METADATA_PATH = SAVED_MODELS_DIR / "alzheimers_metadata.json"

BENCHMARK_RESULTS_PATH = SAVED_MODELS_DIR / "benchmark_comparison.json"
