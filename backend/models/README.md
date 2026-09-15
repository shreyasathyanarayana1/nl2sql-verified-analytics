# Local ML models

These artifacts are used by the NL2SQL pipeline. They are trained without OpenAI or any hosted API.

## Train / retrain

```bash
cd backend
python scripts/train_local_models.py
```

The training script uses the real dataset manifests under `backend/datasets/` and generates synthetic training examples.

## Artifacts

- `intent_classifier.pkl` — intent-family classifier.
- `clarification_classifier.pkl` — clarification/ambiguity classifier.
- `schema_tfidf.pkl` — local TF-IDF schema retriever.
- `sql_template_retriever.pkl` — learned query-family retrieval model used before local SQL rendering.
- `sql_hallucination_classifier.pkl` — question + schema + SQL hallucination classifier (`valid/schema/semantic/structural`).
- `repair_classifier.pkl` — repair/error-type classifier.
- `confidence_model.pkl` — local confidence estimator.
- `TRAINING_REPORT.json` — training metadata.

The deterministic verification and repair rules remain the final safety boundary.
