from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_valid():
    r = client.post("/predict", json={"years_experience": 5.0})
    assert r.status_code == 200
    assert r.json()["predicted_salary"] > 0

def test_predict_invalid_zero():
    r = client.post("/predict", json={"years_experience": 0})
    assert r.status_code == 422   # validation error

def test_metadata():
    r = client.get("/metadata")
    assert r.status_code == 200
    assert "model_type" in r.json()
