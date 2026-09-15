"""Reproducible entry point for the complete local ML stack.

No OpenAI/GPT API is used.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(ROOT / "scripts" / "train_local_models.py")], check=True)
print("All local training jobs completed. See backend/models/TRAINING_REPORT.json")
print("The hallucination detector artifact is included in backend/models/sql_hallucination_classifier.pkl")
