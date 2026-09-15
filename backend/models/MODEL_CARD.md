# Local ML Model Card

This build is designed to run without GPT-4/OpenAI at runtime.

## Models

| Artifact | Method | Pipeline role |
|---|---|---|
| `intent_classifier.pkl` | Word + character TF-IDF + Logistic Regression | Learned intent-family signal |
| `clarification_classifier.pkl` | Logistic Regression | Learned ambiguity signal |
| `schema_tfidf.pkl` | TF-IDF | Local schema retrieval representation |
| `sql_template_retriever.pkl` | TF-IDF + nearest-neighbor retrieval | Learned query-family selection |
| `sql_hallucination_classifier.pkl` | Word + character TF-IDF + Logistic Regression | Valid/schema/semantic/structural hallucination detection |
| `repair_classifier.pkl` | TF-IDF + Logistic Regression | Learned repair/error-type signal |
| `confidence_model.pkl` | Random Forest Regressor | Local confidence estimate |

## Training

Run from `backend/`:

```bash
python scripts/train_local_models.py
```

The training script creates the six general local models from the project's real dataset manifests. The hallucination detector is the previously trained 4-class model and its reproducible training data/script are included here.

### Hallucination detector result

- 4,000 synthetic examples: 1,000 per class
- Classes: `valid`, `schema`, `semantic`, `structural`
- Held-out synthetic accuracy: 79.75%

This is a baseline engineering result. The synthetic split is not evidence of production-level generalization to unseen real-world SQL.

## Important architecture note

The local SQL generator is intentionally **not** a free-form text generator. The learned query-family retriever identifies the closest query family, while the existing verified structured-plan renderer creates the final SQL using the live schema. This prevents a small local model from inventing table/column identifiers.

The deterministic verification layer remains authoritative for live schema existence, semantic requirements, foreign-key structure, read-only execution, result verification, and repair.
