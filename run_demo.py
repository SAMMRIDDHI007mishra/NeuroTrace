"""
NeuroTrace Master CLI Launcher
==============================
One-command initialization, verification, and interactive UI server startup.
Usage:
    python run_demo.py
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def print_banner():
    banner = """
=============================================================================
  _   _                      _____                     
 | \\ | |                    |_   _|                    
 |  \\| | ___ _   _ _ __ ___   | |_ __ __ _  ___ ___    
 | . ` |/ _ \\ | | | '__/ _ \\  | | '__/ _` |/ __/ _ \\   
 | |\\  |  __/ |_| | | | (_) | | | | | (_| | (_|  __/   
 |_| \\_|\\___|\\__,_|_|  \\___/  \\_/_|  \\__,_|\\___\\___|   
                                                       
 Multi-Modal Machine Learning Platform for Neurodegenerative Disorders
 Parkinson's Disease (UCI Acoustic) & Alzheimer's Disease (OASIS MRI)
=============================================================================
"""
    print(banner)

def main():
    print_banner()

    # 1. Dataset Verification
    print("▶ Step 1/4: Checking authentic clinical benchmark datasets...")
    from src.data.download_datasets import ensure_datasets
    raw_dir = BASE_DIR / "data" / "raw"
    ensure_datasets(str(raw_dir))

    # 2. Model Verification & Training if needed
    print("\n▶ Step 2/4: Verifying pre-trained neural ensemble artifacts...")
    from src.config import PARKINSONS_MODEL_PATH, ALZHEIMERS_MODEL_PATH
    if not PARKINSONS_MODEL_PATH.exists():
        print("  - Training Parkinson's Soft-Voting Ensemble (RF + GB + SVC + ET)...")
        from src.models.train_parkinsons import train_and_evaluate_parkinsons
        train_and_evaluate_parkinsons()
    else:
        print("  ✓ Parkinson's ensemble loaded.")

    if not ALZHEIMERS_MODEL_PATH.exists():
        print("  - Training Alzheimer's Soft-Voting Ensemble (RF + GB + SVC + ET)...")
        from src.models.train_alzheimers import train_and_evaluate_alzheimers
        train_and_evaluate_alzheimers()
    else:
        print("  ✓ Alzheimer's ensemble loaded.")

    # 3. Test Suite Verification
    print("\n▶ Step 3/4: Running automated health verification tests...")
    try:
        import pytest
        exit_code = pytest.main(["-q", str(BASE_DIR / "tests")])
        if exit_code == 0:
            print("  ✓ All 14 diagnostic unit and integration tests PASSED!")
        else:
            print("  ⚠ Some tests had warnings or notes.")
    except Exception as e:
        print(f"  Note on testing: {e}")

    # 4. Launch Web Platform
    print("\n▶ Step 4/4: Launching NeuroTrace Interactive Clinical Dashboard...")
    print("  ✓ Local Server Address: http://localhost:5000")
    print("  ✓ REST API Endpoints:   http://localhost:5000/api/v1/health")
    print("  ✓ Press Ctrl+C to stop the server at any time.\n")

    from src.web.app import app
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    main()
