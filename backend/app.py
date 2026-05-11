"""
Salary Prediction API — FastAPI Backend
========================================
Endpoints:
  GET  /           → API info
  GET  /health     → Health check (used by Kubernetes liveness probe)
  POST /predict    → Predict salary from years of experience
  GET  /metadata   → Model metadata (type, R2, RMSE, etc.)
"""

import os
import pickle
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Model globals (populated on startup, NOT at import time) ──
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

model    = None
scaler   = None
metadata = {}


def load_artifacts():
    """Load model, scaler and metadata from disk."""
    try:
        with open(os.path.join(MODEL_DIR, "model.pkl"),  "rb") as f:
            mdl = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scl = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
            meta = json.load(f)
        logger.info("Model and scaler loaded successfully.")
        return mdl, scl, meta
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


# ── Lifespan — runs on startup/shutdown, NOT at import time ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when the server starts; clean up on shutdown."""
    global model, scaler, metadata
    model, scaler, metadata = load_artifacts()
    yield                          # server runs here
    model = scaler = None          # cleanup on shutdown
    metadata = {} 


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Salary Prediction API",
    description="Predicts salary based on years of experience (MLOps Capstone Project)",
    version="1.0.0",
    lifespan=lifespan,             # ← replaces @app.on_event("startup")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    years_experience: float = Field(
        ..., gt=0, le=50,
        description="Years of professional experience (must be > 0 and ≤ 50)",
        example=5.5
    )

    @validator("years_experience")
    def round_to_one_decimal(cls, v):
        return round(v, 1)


class PredictResponse(BaseModel):
    years_experience:  float
    predicted_salary:  float
    currency:          str = "USD"
    model_type:        str
    timestamp:         str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    timestamp:    str


# ── Routes ────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Salary Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health":   "GET  /health",
            "predict":  "POST /predict",
            "metadata": "GET  /metadata",
            "docs":     "GET  /docs",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Kubernetes liveness / readiness probe endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """Predict annual salary given years of experience."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        X        = np.array([[request.years_experience]])
        X_scaled = scaler.transform(X)
        salary   = float(model.predict(X_scaled)[0])
        salary   = max(salary, 0)

        logger.info(f"Prediction: years={request.years_experience} → salary={salary:.2f}")

        return PredictResponse(
            years_experience=request.years_experience,
            predicted_salary=round(salary, 2),
            currency="USD",
            model_type=metadata.get("model_type", "LinearRegression"),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metadata", tags=["Info"])
def get_metadata():
    """Return model metadata: type, R2 score, RMSE, training details."""
    return metadata
