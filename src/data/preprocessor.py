"""
NeuroTrace Data Preprocessing Pipeline
======================================
Robust data cleaning, feature engineering, missing value imputation,
subject grouping, and scaling for Parkinson's and Alzheimer's cohorts.
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

from ..config import (
    PARKINSONS_RAW_FILE,
    PARKINSONS_PROCESSED_FILE,
    PARKINSONS_ALL_FEATURES,
    PARKINSONS_TARGET,
    PARKINSONS_SCALER_PATH,
    ALZHEIMERS_RAW_FILE,
    ALZHEIMERS_PROCESSED_FILE,
    ALZHEIMERS_ALL_FEATURES,
    ALZHEIMERS_TARGET,
    ALZHEIMERS_SCALER_PATH,
)
from ..utils.logger import setup_logger

logger = setup_logger("Preprocessor")

# -------------------------------------------------------------
# Parkinson's Preprocessing
# -------------------------------------------------------------
def extract_subject_id(name: str) -> str:
    """Extracts subject identifier from recording name (e.g., 'phon_R01_S01_1' -> 'S01')."""
    match = re.search(r"_S([0-9]+)_", name)
    if match:
        return f"S{match.group(1)}"
    return name.split("_")[0]

def preprocess_parkinsons(save_artifacts: bool = True) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Cleans and preprocesses the UCI Parkinson's voice dataset.
    Returns:
        X (DataFrame): Clean feature matrix
        y (Series): Binary target (1 = Parkinson's, 0 = Healthy)
        groups (ndarray): Subject identifiers for GroupKFold
    """
    logger.info(f"Loading Parkinson's dataset from {PARKINSONS_RAW_FILE}")
    if not PARKINSONS_RAW_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PARKINSONS_RAW_FILE}. Run download_datasets.py first.")

    df = pd.read_csv(PARKINSONS_RAW_FILE)

    # Extract subject grouping
    df["subject_id"] = df["name"].apply(extract_subject_id)

    # Ensure required features exist
    missing_feats = [f for f in PARKINSONS_ALL_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Parkinson's dataset missing expected features: {missing_feats}")

    # Outlier / sanity check
    df[PARKINSONS_ALL_FEATURES] = df[PARKINSONS_ALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=PARKINSONS_ALL_FEATURES + [PARKINSONS_TARGET])

    X = df[PARKINSONS_ALL_FEATURES].copy()
    y = df[PARKINSONS_TARGET].astype(int)
    groups = df["subject_id"].values

    if save_artifacts:
        # Fit RobustScaler to handle voice tremor spikes
        scaler = RobustScaler()
        scaler.fit(X)
        joblib.dump(scaler, PARKINSONS_SCALER_PATH)
        logger.info(f"Saved Parkinson's scaler to {PARKINSONS_SCALER_PATH}")

        # Save processed CSV for analysis
        df_processed = df.copy()
        df_processed.to_csv(PARKINSONS_PROCESSED_FILE, index=False)
        logger.info(f"Saved processed Parkinson's data to {PARKINSONS_PROCESSED_FILE}")

    logger.info(f"Parkinson's preprocessing complete. Shape: {X.shape}, Subjects: {len(np.unique(groups))}")
    return X, y, groups

# -------------------------------------------------------------
# Alzheimer's (OASIS) Preprocessing
# -------------------------------------------------------------
def preprocess_alzheimers(save_artifacts: bool = True) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Cleans and preprocesses the OASIS Longitudinal Alzheimer's MRI dataset.
    Imputes clinical SES/MMSE scores, encodes gender, and derives clinical dementia label.
    Returns:
        X (DataFrame): Clean feature matrix
        y (Series): Binary target (1 = Dementia / Impaired, 0 = Nondemented)
        groups (ndarray): Subject identifiers for GroupKFold
    """
    logger.info(f"Loading Alzheimer's dataset from {ALZHEIMERS_RAW_FILE}")
    if not ALZHEIMERS_RAW_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ALZHEIMERS_RAW_FILE}. Run download_datasets.py first.")

    df = pd.read_csv(ALZHEIMERS_RAW_FILE)

    # 1. Gender encoding: Male = 1, Female = 0
    df["Gender_Encoded"] = (df["M/F"] == "M").astype(int)

    # 2. Impute missing SES based on Education level median
    ses_by_educ = df.groupby("EDUC")["SES"].transform("median")
    df["SES"] = df["SES"].fillna(ses_by_educ).fillna(df["SES"].median())

    # 3. Impute missing MMSE with overall median
    df["MMSE"] = df["MMSE"].fillna(df["MMSE"].median())

    # 4. Target derivation:
    # A patient is classified as Dementia (1) if Group is 'Demented' OR if 'Converted' with CDR > 0.
    is_demented_group = df["Group"] == "Demented"
    is_converted_impaired = (df["Group"] == "Converted") & (df["CDR"] > 0)
    df[ALZHEIMERS_TARGET] = (is_demented_group | is_converted_impaired).astype(int)

    groups = df["Subject ID"].values
    X = df[ALZHEIMERS_ALL_FEATURES].copy()
    y = df[ALZHEIMERS_TARGET].astype(int)

    if save_artifacts:
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, ALZHEIMERS_SCALER_PATH)
        logger.info(f"Saved Alzheimer's scaler to {ALZHEIMERS_SCALER_PATH}")

        df.to_csv(ALZHEIMERS_PROCESSED_FILE, index=False)
        logger.info(f"Saved processed Alzheimer's data to {ALZHEIMERS_PROCESSED_FILE}")

    logger.info(f"Alzheimer's preprocessing complete. Shape: {X.shape}, Demented: {y.sum()}/{len(y)}")
    return X, y, groups

if __name__ == "__main__":
    preprocess_parkinsons()
    preprocess_alzheimers()
