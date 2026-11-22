"""ML-based hallucination detector used as an independent verification signal."""
from pathlib import Path
from typing import Any, Dict

import joblib

from app import local_ml
from app.config import ML_HALLUCINATION_ENABLED, ML_HALLUCINATION_THRESHOLD, ML_HALLUCINATION_MODEL_PATH

LABELS = {0: "valid", 1: "schema", 2: "semantic", 3: "structural"}
_model = None
_load_error = None


def enabled() -> bool:
    return ML_HALLUCINATION_ENABLED and Path(ML_HALLUCINATION_MODEL_PATH).exists()


def _load():
    global _model, _load_error
    if _model is not None:
        return _model
    try:
        _model = local_ml.restore_compat(joblib.load(ML_HALLUCINATION_MODEL_PATH))
        return _model
    except Exception as exc:  # noqa: BLE001
        _load_error = f"{type(exc).__name__}: {exc}"
        return None


def schema_text(catalog) -> str:
    parts = []
    for table in sorted(catalog.tables):
        columns = ", ".join(c.name for c in catalog.tables[table])
        parts.append(f"TABLE {table}: {columns}")
    for fk in catalog.foreign_keys:
        parts.append(f"FK {fk.from_table}.{fk.from_column} -> {fk.to_table}.{fk.to_column}")
    return "\n".join(parts)


def predict(question: str, catalog, sql: str) -> Dict[str, Any]:
    if not ML_HALLUCINATION_ENABLED:
        return {"enabled": False, "passed": True, "skipped": True, "reason": "ML hallucination detector disabled"}
    if not Path(ML_HALLUCINATION_MODEL_PATH).exists():
        return {"enabled": False, "passed": True, "skipped": True, "reason": f"Model file not found: {ML_HALLUCINATION_MODEL_PATH}"}

    model = _load()
    if model is None:
        return {"enabled": False, "passed": True, "skipped": True, "reason": f"Model could not be loaded: {_load_error}"}

    text = f"QUESTION: {question}\nSCHEMA: {schema_text(catalog)}\nSQL: {sql}"
    label = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))
    label_name = LABELS.get(label, str(label))
    high_confidence = confidence >= ML_HALLUCINATION_THRESHOLD
    passed = label_name == "valid" or not high_confidence

    return {
        "enabled": True,
        "skipped": False,
        "passed": passed,
        "prediction": label_name,
        "confidence": round(confidence, 4),
        "threshold": ML_HALLUCINATION_THRESHOLD,
        "probabilities": {LABELS.get(i, str(i)): round(float(p), 4) for i, p in enumerate(probabilities)},
        "error": None if passed else {
            "code": "ml_hallucination",
            "message": f"ML detector predicts a {label_name} hallucination with {confidence:.0%} confidence",
            "repairable": True,
        },
    }
