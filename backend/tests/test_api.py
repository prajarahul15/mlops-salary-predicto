"""
API tests — run with: pytest backend/tests/ -v
The TestClient context manager triggers FastAPI lifespan (model loading).
"""
import sys
import os

# Allow importing app from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import app   # safe: load_artifacts() no longer runs at import time


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["model_loaded"] is True


def test_predict_valid():
    with TestClient(app) as client:
        r = client.post("/predict", json={"years_experience": 5.0})
        assert r.status_code == 200
        assert r.json()["predicted_salary"] > 0


def test_predict_invalid_zero():
    with TestClient(app) as client:
        r = client.post("/predict", json={"years_experience": 0})
        assert r.status_code == 422    # validation error — gt=0


def test_metadata():
    with TestClient(app) as client:
        r = client.get("/metadata")
        assert r.status_code == 200
        assert "model_type" in r.json()
