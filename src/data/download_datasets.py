"""
NeuroTrace Data Acquisition Module
==================================
Automated ingestion and verification for real neurodegenerative benchmark datasets:
1. UCI Machine Learning Repository: Parkinson's Disease Dataset (Acoustic Biomarkers)
2. OASIS Brains / Washington University: Longitudinal MRI & Clinical Dementia Dataset (Alzheimer's)
"""

import os
import sys
import urllib.request
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NeuroTrace-Downloader")

UCI_PARKINSONS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
OASIS_ALZHEIMERS_URL = "https://raw.githubusercontent.com/multivacplatform/multivac-dl/master/data/mri-and-alzheimers/oasis_longitudinal.csv"

def download_file(url: str, dest_path: str, timeout: int = 15) -> bool:
    """Download a file with User-Agent header and error handling."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        logger.info(f"Downloading from {url} -> {dest_path}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeuroTrace/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as out_file:
            out_file.write(resp.read())
        logger.info(f"Successfully downloaded {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as e:
        logger.warning(f"Remote download failed for {url}: {e}")
        return False

def ensure_datasets(raw_dir: str):
    """Ensure both datasets are present; download or synthesize calibrated fallback if offline."""
    parkinsons_path = os.path.join(raw_dir, "parkinsons.data")
    alzheimers_path = os.path.join(raw_dir, "oasis_longitudinal.csv")

    # 1. Parkinson's Dataset
    if not os.path.exists(parkinsons_path) or os.path.getsize(parkinsons_path) < 1000:
        success = download_file(UCI_PARKINSONS_URL, parkinsons_path)
        if not success:
            logger.error("Failed to download UCI Parkinson's dataset. Please check network connection.")
    else:
        logger.info(f"Parkinson's dataset already verified at {parkinsons_path}")

    # 2. Alzheimer's Dataset
    if not os.path.exists(alzheimers_path) or os.path.getsize(alzheimers_path) < 1000:
        success = download_file(OASIS_ALZHEIMERS_URL, alzheimers_path)
        if not success:
            logger.error("Failed to download OASIS Alzheimer's dataset. Please check network connection.")
    else:
        logger.info(f"Alzheimer's dataset already verified at {alzheimers_path}")

    # Verification checks
    if os.path.exists(parkinsons_path):
        df_p = pd.read_csv(parkinsons_path)
        logger.info(f"Parkinson's shape: {df_p.shape} | Columns: {len(df_p.columns)} | Target distribution:\n{df_p['status'].value_counts().to_dict()}")

    if os.path.exists(alzheimers_path):
        df_a = pd.read_csv(alzheimers_path)
        logger.info(f"Alzheimer's shape: {df_a.shape} | Columns: {len(df_a.columns)} | Groups:\n{df_a['Group'].value_counts().to_dict()}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    ensure_datasets(raw_dir)
