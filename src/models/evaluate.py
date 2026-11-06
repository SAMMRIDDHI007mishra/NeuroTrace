"""
NeuroTrace Comprehensive Model Evaluation Suite
===============================================
Clinical-grade performance metrics:
- Sensitivity (Recall) & Specificity (Crucial for clinical screening)
- Positive Predictive Value (Precision) & Negative Predictive Value (NPV)
- Balanced Accuracy & F1-Score
- Area Under the ROC Curve (ROC-AUC) & PR-AUC
- Brier Calibration Score
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of machine learning and clinical diagnostic metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical epidemiological metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # Negative Predictive Value

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    brier = brier_score_loss(y_true, y_prob)

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "sensitivity_recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision_ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
