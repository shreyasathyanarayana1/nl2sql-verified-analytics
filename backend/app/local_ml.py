"""Local ML inference helpers.

All models are scikit-learn/joblib artifacts trained by backend/scripts/train_local_models.py.
No hosted LLM or API is required at runtime.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from app.config import BASE_DIR

MODEL_DIR = BASE_DIR / "models"
_CACHE: Dict[str, Any] = {}


def _load(name: str):
    if name not in _CACHE:
        path = MODEL_DIR / name
        _CACHE[name] = joblib.load(path) if path.exists() else None
    return _CACHE[name]


def available(name: str) -> bool:
    return (_load(name) is not None)


def predict_intent_kind(question: str) -> Dict[str, Any]:
    bundle = _load("intent_classifier.pkl")
    if not bundle:
        return {"enabled": False}
    X = bundle["vectorizer"].transform([question])
    clf = bundle["classifier"]
    probs = clf.predict_proba(X)[0]
    idx = int(probs.argmax())
    return {"enabled": True, "prediction": str(clf.classes_[idx]), "confidence": round(float(probs[idx]), 4),
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(clf.classes_, probs)}}


def predict_clarification(features: List[float]) -> Dict[str, Any]:
    bundle = _load("clarification_classifier.pkl")
    if not bundle:
        return {"enabled": False}
    clf = bundle["classifier"]
    probs = clf.predict_proba([features])[0]
    idx = int(probs.argmax())
    return {"enabled": True, "prediction": int(clf.classes_[idx]), "confidence": round(float(probs[idx]), 4),
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(clf.classes_, probs)}}


def predict_sql_template(question: str) -> Dict[str, Any]:
    bundle = _load("sql_template_retriever.pkl")
    if not bundle:
        return {"enabled": False}
    X = bundle["vectorizer"].transform([question])
    distances, indices = bundle["nn"].kneighbors(X, n_neighbors=1)
    i = int(indices[0][0])
    similarity = 1.0 - float(distances[0][0])
    return {"enabled": True, "template": bundle["labels"][i], "similarity": round(max(0.0, similarity), 4),
            "matched_question": bundle["texts"][i]}


def predict_repair_type(message: str) -> Dict[str, Any]:
    bundle = _load("repair_classifier.pkl")
    if not bundle:
        return {"enabled": False}
    X = bundle["vectorizer"].transform([message])
    clf = bundle["classifier"]
    probs = clf.predict_proba(X)[0]
    idx = int(probs.argmax())
    return {"enabled": True, "prediction": str(clf.classes_[idx]), "confidence": round(float(probs[idx]), 4)}


def predict_confidence(features: List[float]) -> Optional[float]:
    bundle = _load("confidence_model.pkl")
    if not bundle:
        return None
    value = float(bundle["model"].predict([features])[0])
    return max(5.0, min(99.0, value))


def schema_vectorizer():
    bundle = _load("schema_tfidf.pkl")
    return bundle["vectorizer"] if bundle else None


def schema_embed(texts: List[str]):
    vec = schema_vectorizer()
    if vec is None:
        return None
    X = vec.transform(texts)
    return X.toarray().tolist()
