"""
conftest.py — pytest runs this automatically before any test.
Ensures model.pkl and scaler.pkl exist before app.py is imported.
"""
import subprocess
import sys
import os
from pathlib import Path

# Path to train_model.py (one level up from tests/)
TRAIN_SCRIPT = Path(__file__).parent.parent / "train_model.py"
MODEL_PKL    = Path(__file__).parent.parent / "model" / "model.pkl"


def pytest_configure(config):
    """Called once before test collection — train model if pkl is missing."""
    if not MODEL_PKL.exists():
        print("\n[conftest] model.pkl not found — training model now...")
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[conftest] Training failed:\n{result.stderr}"
            )
        print(f"[conftest] Training complete:\n{result.stdout}")
    else:
        print(f"\n[conftest] model.pkl found — skipping training.")
