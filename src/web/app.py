"""
NeuroTrace Clinical Intelligence Server
=======================================
Full-stack Flask web application and production REST API:
- Dual-condition screening (Parkinson's & Alzheimer's)
- Real-time probabilistic inference with ensemble confidence
- Explainable AI (XAI) feature attribution
- Batch CSV screening & report generation
"""

import os
import sys
from pathlib import Path

# Ensure NeuroTrace root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import joblib

from src.config import (
    PARKINSONS_MODEL_PATH,
    PARKINSONS_SCALER_PATH,
    PARKINSONS_METADATA_PATH,
    PARKINSONS_PROCESSED_FILE,
    PARKINSONS_ALL_FEATURES,
    ALZHEIMERS_MODEL_PATH,
    ALZHEIMERS_SCALER_PATH,
    ALZHEIMERS_METADATA_PATH,
    ALZHEIMERS_PROCESSED_FILE,
    ALZHEIMERS_ALL_FEATURES,
    FIGURES_DIR,
    DATA_DIR
)
from src.data.validator import validate_patient_data
from src.models.explainability import calculate_local_attribution
from src.utils.logger import setup_logger

logger = setup_logger("NeuroTrace-Web")

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static")
)

# Load Models, Scalers, and Background Reference Cohorts
logger.info("Initializing NeuroTrace models and scalers...")
try:
    parkinsons_model = joblib.load(PARKINSONS_MODEL_PATH)
    parkinsons_scaler = joblib.load(PARKINSONS_SCALER_PATH)
    with open(PARKINSONS_METADATA_PATH, "r") as f:
        parkinsons_meta = json.load(f)
    parkinsons_bg = pd.read_csv(PARKINSONS_PROCESSED_FILE)
    logger.info("Parkinson's model and reference cohort loaded.")
except Exception as e:
    logger.error(f"Error loading Parkinson's assets: {e}")
    parkinsons_model, parkinsons_scaler, parkinsons_meta, parkinsons_bg = None, None, {}, None

try:
    alzheimers_model = joblib.load(ALZHEIMERS_MODEL_PATH)
    alzheimers_scaler = joblib.load(ALZHEIMERS_SCALER_PATH)
    with open(ALZHEIMERS_METADATA_PATH, "r") as f:
        alzheimers_meta = json.load(f)
    alzheimers_bg = pd.read_csv(ALZHEIMERS_PROCESSED_FILE)
    logger.info("Alzheimer's model and reference cohort loaded.")
except Exception as e:
    logger.error(f"Error loading Alzheimer's assets: {e}")
    alzheimers_model, alzheimers_scaler, alzheimers_meta, alzheimers_bg = None, None, {}, None

# Load Sample Presets
SAMPLE_CASES_FILE = DATA_DIR / "sample_cases.json"
sample_presets = {}
if SAMPLE_CASES_FILE.exists():
    with open(SAMPLE_CASES_FILE, "r") as f:
        sample_presets = json.load(f)

# -------------------------------------------------------------
# Frontend Routes
# -------------------------------------------------------------
@app.route("/")
def home():
    """Main diagnostic overview and portal landing."""
    return render_template(
        "index.html",
        p_meta=parkinsons_meta,
        a_meta=alzheimers_meta
    )

@app.route("/parkinsons")
def parkinsons_page():
    """Interactive Vocal Acoustic Screening Portal."""
    presets = sample_presets.get("parkinsons", {})
    return render_template("parkinsons.html", presets=presets, meta=parkinsons_meta)

@app.route("/alzheimers")
def alzheimers_page():
    """Interactive Cognitive & MRI Screening Portal."""
    presets = sample_presets.get("alzheimers", {})
    return render_template("alzheimers.html", presets=presets, meta=alzheimers_meta)

@app.route("/benchmarks")
def benchmarks_page():
    """Model benchmarking, metrics tables, and evaluation figures."""
    return render_template(
        "benchmarks.html",
        p_meta=parkinsons_meta,
        a_meta=alzheimers_meta
    )

@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serves high-resolution evaluation figures."""
    return send_from_directory(FIGURES_DIR, filename)

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.route("/api/v1/health", methods=["GET"])
def api_health():
    """System health check and loaded model status."""
    return jsonify({
        "status": "online",
        "platform": "NeuroTrace",
        "version": "1.0.0",
        "models": {
            "parkinsons": parkinsons_model is not None,
            "alzheimers": alzheimers_model is not None
        }
    })

@app.route("/api/v1/presets", methods=["GET"])
def api_presets():
    """Returns curated patient presets for demonstration and testing."""
    return jsonify(sample_presets)

@app.route("/api/v1/predict/parkinsons", methods=["POST"])
def predict_parkinsons():
    """
    Inference endpoint for Parkinson's Disease voice acoustic biomarker vector.
    """
    if parkinsons_model is None or parkinsons_scaler is None:
        return jsonify({"error": "Parkinson's model assets are not loaded."}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload provided."}), 400

    features = data.get("features", data)
    is_valid, validation_msgs = validate_patient_data(features, "parkinsons")
    if not is_valid:
        return jsonify({"error": "Validation failed", "details": validation_msgs}), 422

    # Assemble feature vector
    row = [float(features[f]) for f in PARKINSONS_ALL_FEATURES]
    df_single = pd.DataFrame([row], columns=PARKINSONS_ALL_FEATURES)

    scaled_x = parkinsons_scaler.transform(df_single)
    prob_pd = float(parkinsons_model.predict_proba(scaled_x)[0, 1])
    pred_label = int(prob_pd >= 0.5)

    risk_level = "High" if prob_pd >= 0.70 else ("Moderate" if prob_pd >= 0.35 else "Low")

    # Local XAI attribution
    attributions = calculate_local_attribution(
        parkinsons_model,
        parkinsons_scaler,
        df_single,
        parkinsons_bg,
        PARKINSONS_ALL_FEATURES
    )

    response = {
        "disease": "Parkinson's Disease",
        "diagnosis": "Parkinson's Detected (Positive)" if pred_label == 1 else "Normal / Low Risk (Negative)",
        "probability": round(prob_pd, 4),
        "confidence_percentage": round(prob_pd * 100.0, 1),
        "risk_level": risk_level,
        "warnings": validation_msgs,
        "top_contributing_biomarkers": attributions[:6],
        "disclaimer": "Experimental AI screener. Not a standalone diagnostic substitute for neurological clinical evaluation."
    }
    return jsonify(response)

@app.route("/api/v1/predict/alzheimers", methods=["POST"])
def predict_alzheimers():
    """
    Inference endpoint for Alzheimer's / Dementia clinical and neuroimaging vector.
    """
    if alzheimers_model is None or alzheimers_scaler is None:
        return jsonify({"error": "Alzheimer's model assets are not loaded."}), 500

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload provided."}), 400

    features = data.get("features", data)
    is_valid, validation_msgs = validate_patient_data(features, "alzheimers")
    if not is_valid:
        return jsonify({"error": "Validation failed", "details": validation_msgs}), 422

    row = [float(features[f]) for f in ALZHEIMERS_ALL_FEATURES]
    df_single = pd.DataFrame([row], columns=ALZHEIMERS_ALL_FEATURES)

    scaled_x = alzheimers_scaler.transform(df_single)
    prob_ad = float(alzheimers_model.predict_proba(scaled_x)[0, 1])
    pred_label = int(prob_ad >= 0.5)

    risk_level = "High" if prob_ad >= 0.70 else ("Moderate" if prob_ad >= 0.35 else "Low")

    attributions = calculate_local_attribution(
        alzheimers_model,
        alzheimers_scaler,
        df_single,
        alzheimers_bg,
        ALZHEIMERS_ALL_FEATURES
    )

    response = {
        "disease": "Alzheimer's Disease / Dementia",
        "diagnosis": "Dementia Detected (Positive)" if pred_label == 1 else "Normal / Preserved Cognition (Negative)",
        "probability": round(prob_ad, 4),
        "confidence_percentage": round(prob_ad * 100.0, 1),
        "risk_level": risk_level,
        "warnings": validation_msgs,
        "top_contributing_biomarkers": attributions[:6],
        "disclaimer": "Experimental AI screener. Not a standalone diagnostic substitute for clinical neurological and PET/MRI evaluation."
    }
    return jsonify(response)

@app.route("/api/v1/batch_predict", methods=["POST"])
def batch_predict():
    """Batch prediction from uploaded CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    disease = request.form.get("disease", "parkinsons").lower()

    try:
        df = pd.read_csv(file)
        if disease == "parkinsons":
            model, scaler, feats = parkinsons_model, parkinsons_scaler, PARKINSONS_ALL_FEATURES
        else:
            model, scaler, feats = alzheimers_model, alzheimers_scaler, ALZHEIMERS_ALL_FEATURES

        missing = [f for f in feats if f not in df.columns]
        if missing:
            return jsonify({"error": f"CSV missing required columns: {missing}"}), 422

        scaled = scaler.transform(df[feats])
        probs = model.predict_proba(scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df["NeuroTrace_Risk_Score"] = np.round(probs, 4)
        df["NeuroTrace_Prediction"] = preds
        df["NeuroTrace_Risk_Tier"] = [
            "High" if p >= 0.7 else ("Moderate" if p >= 0.35 else "Low") for p in probs
        ]

        out_path = DATA_DIR / f"batch_results_{disease}.csv"
        df.to_csv(out_path, index=False)

        return jsonify({
            "message": "Batch processing completed",
            "total_records": len(df),
            "positive_count": int(np.sum(preds)),
            "average_risk": round(float(np.mean(probs)), 4),
            "preview": df[["NeuroTrace_Prediction", "NeuroTrace_Risk_Score", "NeuroTrace_Risk_Tier"]].head(10).to_dict(orient="records")
        })
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting NeuroTrace Web Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
