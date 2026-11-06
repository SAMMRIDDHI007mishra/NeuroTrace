"""
NeuroTrace Explainable AI (XAI) Engine
======================================
Provides clinical interpretability for multi-modal neurodegenerative predictions:
1. Global Feature Importance (MDI and Permutation Importance)
2. Local Risk Factor Attribution (Waterfall decomposition of individual patient risk)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.inspection import permutation_importance

def calculate_local_attribution(
    model,
    scaler,
    patient_df: pd.DataFrame,
    background_df: pd.DataFrame,
    feature_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Computes an additive local feature attribution (SHAP-style) for a single patient
    relative to the baseline reference population.

    Returns a list of feature attributions sorted by absolute impact:
    [
        {"feature": "PPE", "value": 0.38, "population_mean": 0.20, "impact": "+0.24", "direction": "risk_escalator"},
        ...
    ]
    """
    patient_scaled = scaler.transform(patient_df[feature_names])
    base_scaled = scaler.transform(background_df[feature_names])

    # Base patient probability
    if hasattr(model, "predict_proba"):
        base_pred_prob = model.predict_proba(patient_scaled)[0, 1]
    else:
        base_pred_prob = float(model.predict(patient_scaled)[0])

    attributions = []
    # Approximate marginal contribution by replacing feature with population median
    for idx, feat in enumerate(feature_names):
        perturbed = patient_scaled.copy()
        median_val = np.median(base_scaled[:, idx])
        perturbed[0, idx] = median_val

        if hasattr(model, "predict_proba"):
            counterfactual_prob = model.predict_proba(perturbed)[0, 1]
        else:
            counterfactual_prob = float(model.predict(perturbed)[0])

        impact = base_pred_prob - counterfactual_prob
        orig_val = float(patient_df[feat].iloc[0])
        pop_mean = float(background_df[feat].mean())

        attributions.append({
            "feature": feat,
            "patient_value": round(orig_val, 4),
            "population_mean": round(pop_mean, 4),
            "impact_score": round(float(impact), 4),
            "category": "risk_escalator" if impact > 0.005 else ("protective_factor" if impact < -0.005 else "neutral"),
            "description": f"{feat}: patient has {round(orig_val, 3)} vs normal mean {round(pop_mean, 3)}"
        })

    # Sort by absolute impact magnitude
    attributions.sort(key=lambda x: abs(x["impact_score"]), reverse=True)
    return attributions

def extract_global_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts normalized global feature importance from ensemble or tree models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # Soft voting classifier
        sub_imps = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                sub_imps.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                sub_imps.append(np.abs(est.coef_[0]))
        if sub_imps:
            importances = np.mean(sub_imps, axis=0)
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to 100%
    total = np.sum(importances)
    if total > 0:
        importances = (importances / total) * 100.0

    results = [
        {"feature": feat, "importance_percentage": round(float(imp), 2)}
        for feat, imp in zip(feature_names, importances)
    ]
    results.sort(key=lambda x: x["importance_percentage"], reverse=True)
    return results
