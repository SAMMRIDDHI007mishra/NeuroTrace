"""
Unit Tests for NeuroTrace Data Ingestion and Preprocessing Pipelines
"""

import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path

from NeuroTrace.src.config import (
    PARKINSONS_RAW_FILE,
    ALZHEIMERS_RAW_FILE,
    PARKINSONS_ALL_FEATURES,
    ALZHEIMERS_ALL_FEATURES
)
from NeuroTrace.src.data.preprocessor import preprocess_parkinsons, preprocess_alzheimers
from NeuroTrace.src.data.validator import validate_patient_data

def test_raw_dataset_existence():
    """Verify raw datasets were acquired and are non-empty."""
    assert PARKINSONS_RAW_FILE.exists(), "UCI Parkinson's dataset missing"
    assert ALZHEIMERS_RAW_FILE.exists(), "OASIS Alzheimer's dataset missing"
    assert os.path.getsize(PARKINSONS_RAW_FILE) > 5000
    assert os.path.getsize(ALZHEIMERS_RAW_FILE) > 5000

def test_parkinsons_preprocessing():
    """Verify Parkinson's preprocessor extracts correct feature matrix and target."""
    X, y, groups = preprocess_parkinsons(save_artifacts=False)
    assert len(X) == 195
    assert len(y) == 195
    assert set(X.columns) == set(PARKINSONS_ALL_FEATURES)
    assert set(np.unique(y)) == {0, 1}
    assert len(np.unique(groups)) > 20

def test_alzheimers_preprocessing():
    """Verify Alzheimer's preprocessor handles missing values and derives targets."""
    X, y, groups = preprocess_alzheimers(save_artifacts=False)
    assert len(X) == 373
    assert len(y) == 373
    assert not X.isnull().values.any(), "Preprocessed Alzheimer's matrix contains NaNs"
    assert set(np.unique(y)) == {0, 1}

def test_clinical_validator_valid_payload():
    """Verify validator accepts valid feature ranges."""
    sample_park = {f: 0.1 for f in PARKINSONS_ALL_FEATURES}
    sample_park["MDVP:Fo(Hz)"] = 150.0
    sample_park["MDVP:Fhi(Hz)"] = 200.0
    sample_park["MDVP:Flo(Hz)"] = 100.0
    sample_park["HNR"] = 22.0
    sample_park["spread1"] = -4.5
    sample_park["D2"] = 2.4

    is_valid, msgs = validate_patient_data(sample_park, "parkinsons")
    assert is_valid is True

def test_clinical_validator_missing_payload():
    """Verify validator flags missing essential fields."""
    sample_incomplete = {"MDVP:Fo(Hz)": 150.0}
    is_valid, msgs = validate_patient_data(sample_incomplete, "parkinsons")
    assert is_valid is False
    assert len(msgs) > 0
