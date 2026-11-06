"""
Unit Tests for NeuroTrace Model Artifacts and Inference Calibration
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from NeuroTrace.src.config import (
    PARKINSONS_MODEL_PATH,
    PARKINSONS_SCALER_PATH,
    ALZHEIMERS_MODEL_PATH,
    ALZHEIMERS_SCALER_PATH,
    PARKINSONS_ALL_FEATURES,
    ALZHEIMERS_ALL_FEATURES
)
from NeuroTrace.src.models.evaluate import compute_clinical_metrics

def test_model_files_exist():
    """Verify serialized ensemble models and scalers exist on disk."""
    assert PARKINSONS_MODEL_PATH.exists()
    assert PARKINSONS_SCALER_PATH.exists()
    assert ALZHEIMERS_MODEL_PATH.exists()
    assert ALZHEIMERS_SCALER_PATH.exists()

def test_parkinsons_model_inference():
    """Verify Parkinson's model produces calibrated probabilities."""
    model = joblib.load(PARKINSONS_MODEL_PATH)
    scaler = joblib.load(PARKINSONS_SCALER_PATH)

    dummy_input = np.zeros((1, len(PARKINSONS_ALL_FEATURES)))
    scaled = scaler.transform(dummy_input)
    probs = model.predict_proba(scaled)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0

def test_alzheimers_model_inference():
    """Verify Alzheimer's model produces calibrated probabilities."""
    model = joblib.load(ALZHEIMERS_MODEL_PATH)
    scaler = joblib.load(ALZHEIMERS_SCALER_PATH)

    dummy_input = np.zeros((1, len(ALZHEIMERS_ALL_FEATURES)))
    scaled = scaler.transform(dummy_input)
    probs = model.predict_proba(scaled)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0

def test_compute_clinical_metrics():
    """Verify evaluation metric calculations match epidemiological definitions."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.1])

    metrics = compute_clinical_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.75
    assert metrics["sensitivity_recall"] == 0.5  # 1/2
    assert metrics["specificity"] == 1.0         # 2/2
