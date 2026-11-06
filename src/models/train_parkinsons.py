"""
NeuroTrace Parkinson's Model Training & Benchmark Pipeline
==========================================================
Trains, tunes, cross-validates, and ensembles multiple machine learning architectures
on the UCI Parkinson's Disease Voice Acoustic Dataset.
"""

import os
import sys
from pathlib import Path

# Add project root to path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import joblib

from src.config import (
    PARKINSONS_ALL_FEATURES,
    PARKINSONS_MODEL_PATH,
    PARKINSONS_SCALER_PATH,
    PARKINSONS_METADATA_PATH,
    FIGURES_DIR,
    RANDOM_STATE
)
from src.data.preprocessor import preprocess_parkinsons
from src.models.evaluate import compute_clinical_metrics
from src.models.explainability import extract_global_importance
from src.utils.logger import setup_logger

logger = setup_logger("Train-Parkinsons")

def train_and_evaluate_parkinsons():
    """Executes full training, cross-validation, and ensembling for Parkinson's detection."""
    X, y, groups = preprocess_parkinsons(save_artifacts=True)
    scaler = joblib.load(PARKINSONS_SCALER_PATH)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test, grp_train, grp_test = train_test_split(
        X_scaled, y, groups, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic_Regression": LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_STATE),
        "Support_Vector_Machine": SVC(C=2.0, kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
    }

    logger.info("Running 5-Fold Stratified Cross-Validation on Parkinson's cohort...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_benchmarks = {}

    for name, model in candidates.items():
        cv_accs, cv_rocs, cv_f1s = [], [], []
        for train_idx, val_idx in cv.split(X_train, y_train):
            m = model.__class__(**model.get_params())
            m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            val_preds = m.predict(X_train.iloc[val_idx])
            val_probs = m.predict_proba(X_train.iloc[val_idx])[:, 1]
            metrics = compute_clinical_metrics(y_train.iloc[val_idx], val_preds, val_probs)
            cv_accs.append(metrics["accuracy"])
            cv_rocs.append(metrics["roc_auc"])
            cv_f1s.append(metrics["f1_score"])

        model_benchmarks[name] = {
            "cv_accuracy_mean": round(float(np.mean(cv_accs)), 4),
            "cv_accuracy_std": round(float(np.std(cv_accs)), 4),
            "cv_roc_auc_mean": round(float(np.mean(cv_rocs)), 4),
            "cv_f1_mean": round(float(np.mean(cv_f1s)), 4)
        }
        logger.info(f"{name:25s} | CV Acc: {np.mean(cv_accs):.3f} +/- {np.std(cv_accs):.3f} | ROC-AUC: {np.mean(cv_rocs):.3f}")

    # Build Soft-Voting Ensemble from complementary models
    ensemble = VotingClassifier(
        estimators=[
            ("rf", candidates["Random_Forest"]),
            ("gb", candidates["Gradient_Boosting"]),
            ("svm", candidates["Support_Vector_Machine"]),
            ("et", candidates["Extra_Trees"])
        ],
        voting="soft"
    )

    logger.info("Fitting NeuroTrace Parkinson's Ensemble Model...")
    ensemble.fit(X_train, y_train)

    # Test set evaluation
    test_preds = ensemble.predict(X_test)
    test_probs = ensemble.predict_proba(X_test)[:, 1]
    final_metrics = compute_clinical_metrics(y_test, test_preds, test_probs)
    logger.info(
        f"Test Set Evaluation: Acc={final_metrics['accuracy']} | "
        f"Sensitivity={final_metrics['sensitivity_recall']} | "
        f"Specificity={final_metrics['specificity']} | "
        f"ROC-AUC={final_metrics['roc_auc']}"
    )

    # Save ensemble weights
    joblib.dump(ensemble, PARKINSONS_MODEL_PATH)
    logger.info(f"Saved trained ensemble to {PARKINSONS_MODEL_PATH}")

    # Extract global importance
    importance_list = extract_global_importance(ensemble, PARKINSONS_ALL_FEATURES)

    # Save metadata
    metadata = {
        "disease": "Parkinson's Disease",
        "dataset_source": "UCI Machine Learning Repository (Little et al., 2007)",
        "features": PARKINSONS_ALL_FEATURES,
        "test_metrics": final_metrics,
        "cv_benchmarks": model_benchmarks,
        "top_biomarkers": importance_list[:10],
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    with open(PARKINSONS_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to {PARKINSONS_METADATA_PATH}")

    # Generate Figures (ROC curve & Confusion Matrix & Feature Importance)
    generate_parkinsons_figures(y_test, test_probs, test_preds, importance_list, final_metrics["roc_auc"])

    return metadata

def generate_parkinsons_figures(y_test, test_probs, test_preds, importance_list, auc_score):
    """Generates publication-quality diagnostic visualization figures."""
    sns.set_theme(style="whitegrid")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"NeuroTrace Ensemble (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--")
    plt.title("Parkinson's Diagnostic ROC Curve", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "parkinsons_roc_curve.png")
    plt.close()

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Healthy (0)", "Parkinson's (1)"],
        yticklabels=["Healthy (0)", "Parkinson's (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Parkinson's Confusion Matrix (Holdout Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Diagnosis", fontsize=11)
    plt.ylabel("True Diagnosis", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "parkinsons_confusion_matrix.png")
    plt.close()

    # 3. Feature Importance Bar Chart
    df_imp = pd.DataFrame(importance_list[:12])
    plt.figure(figsize=(9, 6), dpi=300)
    sns.barplot(data=df_imp, x="importance_percentage", y="feature", palette="Blues_r")
    plt.title("Top Vocal Acoustic Biomarkers for Parkinson's Detection", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Clinical Importance (%)", fontsize=11)
    plt.ylabel("Acoustic Metric", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "parkinsons_feature_importance.png")
    plt.close()

    logger.info("Generated Parkinson's figures in reports/figures/")

if __name__ == "__main__":
    train_and_evaluate_parkinsons()
