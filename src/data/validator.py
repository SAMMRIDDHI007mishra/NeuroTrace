"""
NeuroTrace Clinical Input Validator
===================================
Enforces clinical boundary assertions and schema verification
for inference payloads (single patient & batch).
"""

from typing import Dict, Any, Tuple, List
import numpy as np

# Clinically plausible ranges for Parkinson's Voice Acoustic Features
PARKINSONS_FEATURE_RANGES = {
    "MDVP:Fo(Hz)": (50.0, 300.0),       # Fundamental frequency
    "MDVP:Fhi(Hz)": (70.0, 600.0),      # Maximum fundamental frequency
    "MDVP:Flo(Hz)": (40.0, 260.0),      # Minimum fundamental frequency
    "MDVP:Jitter(%)": (0.0001, 0.05),   # Vocal jitter percentage
    "MDVP:Jitter(Abs)": (0.000001, 0.001),
    "MDVP:RAP": (0.0001, 0.03),
    "MDVP:PPQ": (0.0001, 0.03),
    "Jitter:DDP": (0.0001, 0.1),
    "MDVP:Shimmer": (0.001, 0.25),      # Amplitude variation
    "MDVP:Shimmer(dB)": (0.01, 2.5),
    "Shimmer:APQ3": (0.001, 0.15),
    "Shimmer:APQ5": (0.001, 0.15),
    "MDVP:APQ": (0.001, 0.20),
    "Shimmer:DDA": (0.001, 0.45),
    "NHR": (0.0, 0.4),                  # Noise to Harmonics Ratio
    "HNR": (5.0, 40.0),                 # Harmonics to Noise Ratio (dB)
    "RPDE": (0.1, 0.9),                 # Recurrence Period Density Entropy
    "DFA": (0.4, 0.9),                  # Detrended Fluctuation Analysis
    "spread1": (-9.0, 0.0),             # Non-linear pitch variation 1
    "spread2": (0.0, 0.6),              # Non-linear pitch variation 2
    "D2": (1.0, 4.0),                   # Correlation Dimension
    "PPE": (0.01, 0.6)                  # Pitch Period Entropy
}

# Clinically plausible ranges for Alzheimer's / OASIS Features
ALZHEIMERS_FEATURE_RANGES = {
    "Gender_Encoded": (0, 1),           # 0 = Female, 1 = Male
    "Age": (50, 105),                   # Patient age
    "EDUC": (0, 30),                    # Years of education
    "SES": (1.0, 5.0),                  # Socioeconomic status (1=Highest, 5=Lowest)
    "MMSE": (0.0, 30.0),                # Mini-Mental State Examination score
    "CDR": (0.0, 3.0),                  # Clinical Dementia Rating (0, 0.5, 1, 2, 3)
    "eTIV": (900.0, 2300.0),            # Estimated Total Intracranial Volume (cm3)
    "nWBV": (0.60, 0.90),               # Normalized Whole Brain Volume
    "ASF": (0.75, 1.8)                  # Atlas Scaling Factor
}

def validate_patient_data(features: Dict[str, Any], disease: str) -> Tuple[bool, List[str]]:
    """
    Validates dictionary of patient features against expected ranges and clinical rules.
    Returns (is_valid, list_of_errors_or_warnings).
    """
    errors = []
    ranges = PARKINSONS_FEATURE_RANGES if disease.lower() == "parkinsons" else ALZHEIMERS_FEATURE_RANGES

    for feat, (low, high) in ranges.items():
        if feat not in features:
            errors.append(f"Missing required clinical biomarker: '{feat}'")
            continue
        try:
            val = float(features[feat])
            if np.isnan(val) or np.isinf(val):
                errors.append(f"Feature '{feat}' has invalid value: NaN/Inf")
            elif val < low or val > high:
                # Warning rather than hard stop for extreme clinical outliers
                errors.append(f"Feature '{feat}' value {val} is outside standard clinical reference range [{low}, {high}].")
        except (ValueError, TypeError):
            errors.append(f"Feature '{feat}' cannot be parsed as numeric float.")

    is_valid = len([e for e in errors if "Missing" in e or "NaN/Inf" in e or "cannot be parsed" in e]) == 0
    return is_valid, errors
