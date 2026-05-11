"""
Unit tests for the Salary Prediction API.
Run with: pytest backend/tests/ -v
"""
import sys
import os

# Allow importing app from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_returns_info():
    r = client.get("/")
    assert r.status_code == 200
    assert "Salary Prediction API" in r.json()["service"]


def test_health_check():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input():
    r = client.post("/predict", json={"years_experience": 5.5})
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_salary"] > 0
    assert data["years_experience"] == 5.5
    assert data["currency"] == "USD"


def test_predict_low_experience():
    r = client.post("/predict", json={"years_experience": 0.5})
    assert r.status_code == 200
    assert r.json()["predicted_salary"] > 0


def test_predict_high_experience():
    r = client.post("/predict", json={"years_experience": 40.0})
    assert r.status_code == 200
    assert r.json()["predicted_salary"] > 0


def test_predict_zero_experience_rejected():
    """years_experience must be > 0"""
    r = client.post("/predict", json={"years_experience": 0})
    assert r.status_code == 422


def test_predict_negative_rejected():
    """Negative experience is invalid"""
    r = client.post("/predict", json={"years_experience": -1})
    assert r.status_code == 422


def test_predict_over_limit_rejected():
    """years_experience > 50 should be rejected"""
    r = client.post("/predict", json={"years_experience": 55})
    assert r.status_code == 422


def test_predict_string_input_rejected():
    """Non-numeric input should be rejected"""
    r = client.post("/predict", json={"years_experience": "abc"})
    assert r.status_code == 422


def test_predict_missing_field_rejected():
    """Missing required field should return 422"""
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_metadata_endpoint():
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert "model_type" in data
    assert "r2_score" in data
    assert data["r2_score"] > 0
