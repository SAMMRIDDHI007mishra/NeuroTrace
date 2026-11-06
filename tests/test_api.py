"""
Integration Tests for NeuroTrace REST API & Web Server Endpoints
"""

import pytest
import json
from pathlib import Path

from NeuroTrace.src.web.app import app
from NeuroTrace.src.config import DATA_DIR

@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_api_health(client):
    """Test /api/v1/health returns online status and active models."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "online"
    assert data["platform"] == "NeuroTrace"
    assert data["models"]["parkinsons"] is True
    assert data["models"]["alzheimers"] is True

def test_api_presets(client):
    """Test /api/v1/presets returns clinical cases."""
    res = client.get("/api/v1/presets")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "parkinsons" in data
    assert "alzheimers" in data

def test_predict_parkinsons_endpoint(client):
    """Test /api/v1/predict/parkinsons with preset clinical case."""
    with open(DATA_DIR / "sample_cases.json") as f:
        samples = json.load(f)
    case = samples["parkinsons"]["confirmed_parkinsons"]["features"]

    res = client.post("/api/v1/predict/parkinsons", json=case)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "probability" in data
    assert "diagnosis" in data
    assert "risk_level" in data
    assert "top_contributing_biomarkers" in data
    assert len(data["top_contributing_biomarkers"]) > 0

def test_predict_alzheimers_endpoint(client):
    """Test /api/v1/predict/alzheimers with preset clinical case."""
    with open(DATA_DIR / "sample_cases.json") as f:
        samples = json.load(f)
    case = samples["alzheimers"]["confirmed_dementia"]["features"]

    res = client.post("/api/v1/predict/alzheimers", json=case)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "probability" in data
    assert "diagnosis" in data
    assert data["risk_level"] in ["Low", "Moderate", "High"]

def test_web_routes(client):
    """Verify HTML front-end routes render with HTTP 200."""
    for path in ["/", "/parkinsons", "/alzheimers", "/benchmarks"]:
        res = client.get(path)
        assert res.status_code == 200
