from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "sql_hallucination_classifier.pkl"
LABELS = {0: "valid", 1: "schema", 2: "semantic", 3: "structural"}
_model = None

def predict_hallucination(question, schema, sql):
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    text = f"QUESTION: {question}\nSCHEMA: {schema}\nSQL: {sql}"
    label = int(_model.predict([text])[0])
    probs = _model.predict_proba([text])[0]
    return {
        "label": label,
        "label_name": LABELS[label],
        "confidence": float(max(probs)),
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(probs))}
    }
