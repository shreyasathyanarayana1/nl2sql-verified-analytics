# Hallucination-Resistant Multi-Agent NL2SQL Platform

Working prototype of the Phase-1 design: an agent pipeline that answers plain-English questions over real datasets, checks every SQL query (and every answer) for hallucinations, repairs what it can, and explains how far each answer can be trusted.

- **backend/** — FastAPI + a state-machine orchestrator (LangGraph-style) over the agents
- **frontend/** — Next.js 14 + React 18 + Chart.js

Datasets (in `backend/datasets/`, each imported into its own SQLite database on startup):

| Dataset | Source file | Rows | Tables |
|---------|-------------|------|--------|
| Student Performance Factors | `StudentPerformanceFactors.csv` (Kaggle) | 6,607 students | `students` |
| Heart Disease (Hungarian) | `data.csv.xls` (UCI Heart Disease, Hungarian cohort) | 294 patients | `patients` + 4 lookup tables for coded values |

This GitHub build is **local-only**: no OpenAI API key and no GPT-4 are required or used. The repository includes trained scikit-learn models plus deterministic safety checks. SQL is produced locally from the verified structured plan, with a learned query-family retriever assisting generation.

## Run it

```bash
# 1. Backend (Python 3.9+)
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload --port 8000
# imports backend/datasets/*/source.csv into data/*.db on first start

# 2. Frontend (Node 18+), in a second terminal
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Demo logins:

| Persona       | Username  | Password     |
|---------------|-----------|--------------|
| Insights User | `analyst` | `analyst123` |
| Administrator | `admin`   | `admin123`   |

Optional environment variables: `SCHEMA_TOP_K`, `AMBIGUITY_THRESHOLD` (0.6), `MAX_REPAIR_ATTEMPTS` (3), `JWT_SECRET`, `CORS_ORIGINS`, `NEXT_PUBLIC_API_URL`, `ML_HALLUCINATION_THRESHOLD`.

## Where each agent lives

| # | Agent | File | Offline behaviour |
|---|-------|------|-------------------|
| 1 | Intent Understanding | `app/agents/intent.py` | Deterministic structured extraction + trained local intent classifier as an independent signal |
| 2 | Query Clarification | `app/agents/clarification.py` | Weighted ambiguity signals + trained local clarification classifier |
| 3 | Schema Reasoning (RAG) | `app/agents/schema_reasoning.py` | Trained local TF-IDF retrieval, follow-up retrieval, FK bridges |
| 3b | RBAC Access Control | `app/agents/access_control.py` | Column denials per role (runs in parallel with 3) |
| 4 | Query Planning | `app/agents/planning.py` | Numbered plain-English plan + structured spec, limited to retrieved tables |
| 5 | SQL Generation | `app/agents/sql_generation.py` | Trained local query-family retriever + schema-safe deterministic renderer; no hosted LLM |
| 6 | Anti-Hallucination | `app/agents/anti_hallucination.py` | Deterministic schema/semantic/FK checks + trained hallucination classifier |
| 7 | Query Repair | `app/agents/repair.py` | Trained repair-type classifier + fuzzy match, FK join rewrite, local regeneration |
| 8 | Execution | `app/agents/execution.py` | Read-only SQLite connection with a time budget |
| 8b | Result Verification | `app/agents/result_checks.py` | Checks the answer itself with independent probe queries |
| 8c | Data Quality & Confidence | `app/agents/data_quality.py`, `confidence.py` | Statistical warnings + trained local confidence model blended with rule evidence |
| 9 | Visualization | `app/agents/visualization.py` | Bar, line, pie, histogram or scatter chart with a plain-English explanation |

Orchestration and the two feedback loops: `app/orchestrator.py`. API: `app/main.py`.

## What users can ask

| Type | Examples |
|------|----------|
| Averages, totals, rankings | "Average CGPA by department", "Top 5 students by CGPA in CSE" |
| Thresholds | "Students with attendance below 75%", "How many per department?" |
| Pass/fail | "Pass rate by subject in SEE", "How many students failed in CIE1?" (pass = 40% of max marks) |
| Distributions | "CGPA distribution", "How are SEE marks spread?" |
| Trends | "Marks trend across exams", "Average CGPA trend over semesters" |
| Share | "Share of students by department" |
| Relationships | "Does attendance affect marks?", "Attendance vs CGPA" (scatter plot + correlation) |
| Follow-ups | "average cgpa by department", then "only for semester 5" |

## Trust features (beyond checking the SQL)

| Feature | File | What it does |
|---------|------|--------------|
| Result verification | `app/agents/result_checks.py` | Probe queries test the *answer*: an independently planned query agrees, values stay in range, groups add up to the total, removing a filter never shrinks a count, no duplicate groups. If the answer disagrees with the reference query, the reference replaces it. |
| Data-quality warnings | `app/agents/data_quality.py` | Small groups, ties at a top-N cut-off, Simpson's paradox (a comparison that reverses within semesters), too few students for a correlation. |
| Confidence score | `app/agents/confidence.py` | A 0–100 score with every reason listed. Below 55 the UI hides the answer by default ("Show the result anyway"). |
| Ask only when it matters | `orchestrator.py` (`compare_interpretations`) | For an ambiguous question, every interpretation is run first. If they agree, it answers without asking; if not, it asks and previews each option's answer. |
| Rows behind a number | `app/agents/evidence.py` | Click a bar, slice or table row to see the underlying records, derived from the verified query and re-checked before running. |
| Suggested follow-ups | `app/agents/followups.py` | "Explore further" chips under each answer. |

## Evaluation: hallucination injection (API only)

The UI no longer has a demo switch, but `POST /api/query` still accepts `"inject": "schema" | "semantic" | "structural" | "random"`. It corrupts the first SQL draft so detection and repair can be measured.

```bash
cd backend && PYTHONPATH=. .venv/bin/python scripts/smoke.py random
```

## Known gaps before Phase-2

- SQLite stands in for PostgreSQL/pgvector, and in-memory sessions stand in for Redis.
- The Student Performance dataset is a single table, so join validation is only exercised on the Heart Disease dataset (via its lookup tables).
- The local models are trained from synthetic examples generated from the project manifests; their benchmark scores should not be interpreted as real-world generalization.
- SQL generation intentionally uses the verified deterministic renderer after the learned query-family selection. This is safer than allowing a free-form ML model to invent identifiers.
- The project no longer contains an active hosted-LLM execution path.
- The demo users are hard-coded in `app/auth.py`.

## Adding a dataset

1. Create `backend/datasets/<id>/` with `source.csv` and a `manifest.json` (copy an existing one).
2. In the manifest, give each column a `role` (`measure`, `category` or `flag`), a `label`, `synonyms` people might use, and for coded columns either a `value_map` or a `lookup` table.
3. Restart the backend. The CSV is imported, profiled and indexed automatically, and the dataset appears in the switcher.

## ML hallucination detector

The project now includes a trained ML hallucination classifier at `backend/models/sql_hallucination_classifier.pkl`.

The classifier receives the user's question, the live database schema, and the generated SQL and predicts one of `valid`, `schema`, `semantic`, or `structural`. It is used as an additional learned verification signal inside Agent 6 (`app/agents/anti_hallucination.py`). The existing deterministic schema, semantic, and foreign-key checks remain authoritative. The ML check can reject a query only when those deterministic checks all pass and the classifier confidence is at least `ML_HALLUCINATION_THRESHOLD` (default `0.80`). If a deterministic check already fails, the ML prediction is retained as informational evidence and does not weaken the existing rejection/repair path.

Configuration:

- `ML_HALLUCINATION_ENABLED=true|false`
- `ML_HALLUCINATION_THRESHOLD=0.80`
- `ML_HALLUCINATION_MODEL_PATH=backend/models/sql_hallucination_classifier.pkl`

Dependencies added: `scikit-learn==1.8.0` and `joblib==1.5.3`.
