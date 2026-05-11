"""
Lightweight model training script for CI/CD.
Trains the Linear Regression model and saves model.pkl, scaler.pkl,
and metadata.json into backend/model/ — no MLflow or WhyLogs required.

Run with:
    python backend/train_model.py
"""

import os
import pickle
import json
import pathlib

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Paths ─────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).parent.parent   # repo root
DATA_PATH = ROOT / "2687_capstone_project_dataset_v1_vv6_ahjq7xz.csv"
MODEL_DIR = pathlib.Path(__file__).parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, index_col=0)
X  = df[["YearsExperience"]].values
y  = df["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LinearRegression(fit_intercept=True)
model.fit(X_train_sc, y_train)

# ── Metrics ───────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
r2   = round(float(r2_score(y_test, y_pred)), 4)
rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
mae  = round(float(mean_absolute_error(y_test, y_pred)), 2)

# ── Save artifacts ────────────────────────────────────────
with open(MODEL_DIR / "model.pkl",  "wb") as f: pickle.dump(model,  f)
with open(MODEL_DIR / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)

metadata = {
    "model_type":       "LinearRegression",
    "feature":          "YearsExperience",
    "target":           "Salary",
    "r2_score":         r2,
    "rmse":             rmse,
    "mae":              mae,
    "training_samples": len(X_train),
}
with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved  → {MODEL_DIR / 'model.pkl'}")
print(f"Scaler saved → {MODEL_DIR / 'scaler.pkl'}")
print(f"R2={r2}  RMSE={rmse}  MAE={mae}")
