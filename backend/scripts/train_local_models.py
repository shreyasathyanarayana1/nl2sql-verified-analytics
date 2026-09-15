"""Train the local ML models used by the NL2SQL pipeline.

No OpenAI/API calls are used. Training data is generated from the project's real
manifest schemas and existing hallucination injection taxonomy.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
RNG = random.Random(42)

KINDS = ["aggregate", "ranking", "list", "distribution", "correlation", "drivers", "unknown"]


def load_manifests():
    out = []
    for p in sorted(DATASETS.glob("*/manifest.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def columns(m):
    return m.get("columns", [])


def measures(m):
    return [c for c in columns(m) if c.get("role") in ("measure", "flag")]


def cats(m):
    return [c for c in columns(m) if c.get("role") == "category"]


def label(c):
    return c.get("label", c["name"]).lower()


def q_templates(m):
    ms = measures(m)
    cs = cats(m)
    qs = []
    # Aggregate
    for c in ms[:8]:
        for g in cs[:4]:
            qs += [
                (f"average {label(c)} by {label(g)}", "aggregate"),
                (f"what is the mean {label(c)} for each {label(g)}", "aggregate"),
                (f"{label(c)} by {label(g)}", "aggregate"),
            ]
        qs += [(f"average {label(c)}", "aggregate"), (f"total {label(c)}", "aggregate"),
               (f"how many {m['entity']['plural']} are there", "aggregate")]
    # Ranking
    for c in ms[:8]:
        for g in cs[:4]:
            qs += [(f"top 10 {label(g)} by average {label(c)}", "ranking"),
                   (f"bottom 5 {label(g)} with the lowest {label(c)}", "ranking")]
        qs += [(f"top 10 {m['entity']['plural']} by {label(c)}", "ranking"),
               (f"which {m['entity']['plural']} have the highest {label(c)}", "ranking")]
    # List
    for c in ms[:5] + cs[:5]:
        qs += [(f"show {m['entity']['plural']} with {label(c)}", "list"),
               (f"list {m['entity']['plural']} and their {label(c)}", "list")]
    # Distribution
    for c in ms[:8]:
        qs += [(f"{label(c)} distribution", "distribution"),
               (f"show the distribution of {label(c)}", "distribution"),
               (f"how is {label(c)} distributed", "distribution")]
    # Correlation
    for i, a in enumerate(ms[:7]):
        for b in ms[i+1:8]:
            qs += [(f"{label(a)} vs {label(b)}", "correlation"),
                   (f"is {label(a)} correlated with {label(b)}", "correlation"),
                   (f"relationship between {label(a)} and {label(b)}", "correlation")]
    # Drivers
    outcome = next((c for c in ms if c.get("outcome")), ms[0] if ms else None)
    if outcome:
        for g in cs[:6]:
            qs += [(f"what factors affect {label(outcome)}", "drivers"),
                   (f"what drives {label(outcome)}", "drivers"),
                   (f"which variables influence {label(outcome)}", "drivers")]
            break
    # Unknown / clarification-like
    qs += [(f"show me useful information", "unknown"), (f"tell me something interesting", "unknown"),
           (f"what should I look at", "unknown")]
    return qs


def train_intent(manifests):
    rows = []
    for m in manifests:
        rows += q_templates(m)
    rng = np.random.default_rng(42)
    rng.shuffle(rows)
    texts, y = zip(*rows)
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True)),
    ])
    X = vec.fit_transform(texts)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    report = classification_report(yte, pred, zero_division=0)
    joblib.dump({"vectorizer": vec, "classifier": clf, "labels": KINDS}, MODELS / "intent_classifier.pkl")
    return {"accuracy": accuracy_score(yte, pred), "report": report, "rows": len(rows)}


def train_clarification(manifests):
    rows = []
    for m in manifests:
        ms = measures(m)
        cs = cats(m)
        for c in ms[:8]:
            rows += [(f"average {label(c)}", [1,0,0,0,0], 0),
                     (f"average", [0,1,0,0,1], 1),
                     (f"what should I analyze", [0,0,1,0,1], 1),
                     (f"top 10", [0,0,0,1,1], 1),
                     (f"average {label(c)} by {label(cs[0]) if cs else label(c)}", [0,0,0,0,0], 0)]
    X = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X, y)
    joblib.dump({"classifier": clf, "feature_names": ["has_measure","missing_measure","unsupported","ranking","weak_signal"]}, MODELS / "clarification_classifier.pkl")
    return {"accuracy_train": accuracy_score(y, clf.predict(X)), "rows": len(rows)}


def train_sql_template(manifests):
    rows = []
    for m in manifests:
        rows += q_templates(m)
    texts, labels = zip(*rows)
    # A learned retrieval model selects the closest known NL query family.
    vec = TfidfVectorizer(ngram_range=(1,2), analyzer="word", sublinear_tf=True)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(5, len(texts)), metric="cosine").fit(X)
    joblib.dump({"vectorizer": vec, "nn": nn, "texts": list(texts), "labels": list(labels)}, MODELS / "sql_template_retriever.pkl")
    return {"rows": len(rows)}


def train_repair(manifests):
    examples = []
    for m in manifests:
        fact = m["fact_table"]
        ms = measures(m)
        col = ms[0]["name"] if ms else m["primary_measure"]
        examples += [
            (f"unknown column {fact}.{col}x", "schema"),
            (f"unknown table {fact}x", "schema"),
            ("SQL does not parse", "schema"),
            ("used SUM instead of AVG", "semantic"),
            ("missing WHERE condition", "semantic"),
            ("wrong ranking direction", "semantic"),
            ("invalid join not a foreign key relationship", "structural"),
            ("cross join without key condition", "structural"),
        ]
    texts, y = zip(*examples)
    vec = TfidfVectorizer(ngram_range=(1,2), analyzer="word", sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(X, y)
    joblib.dump({"vectorizer": vec, "classifier": clf}, MODELS / "repair_classifier.pkl")
    return {"rows": len(examples), "train_accuracy": accuracy_score(y, clf.predict(X))}


def train_confidence(manifests):
    # Target is generated from the same evidence factors, then learned so the runtime
    # can produce a stable local confidence estimate without an LLM.
    rows = []
    for _ in range(6000):
        rejected = RNG.randint(0,3)
        regenerated = RNG.randint(0,rejected) if rejected else 0
        result_failed = RNG.randint(0,2)
        result_repaired = int(result_failed > 0 and RNG.random() < .35)
        warnings = RNG.randint(0,4)
        ignored = RNG.randint(0,1)
        followup = RNG.randint(0,1)
        retrieval = RNG.randint(0,1)
        clarify = RNG.randint(0,2)
        features = [rejected, regenerated, result_failed, result_repaired, warnings, ignored, followup, retrieval, clarify]
        score = 100 - 12*regenerated - 6*(rejected-regenerated) - 35*result_failed - 10*result_repaired - 8*warnings - 10*ignored - 5*followup - 3*retrieval - 15*max(0,clarify-1)
        score = max(5, min(99, score))
        rows.append((features, score))
    X = np.array([x for x,_ in rows], dtype=float)
    y = np.array([y for _,y in rows], dtype=float)
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(X,y)
    joblib.dump({"model": model, "feature_names": ["rejected","regenerated","result_failed","result_repaired","warnings","ignored","followup","retrieval","clarify_rounds"]}, MODELS / "confidence_model.pkl")
    return {"rows": len(rows), "rmse_train": float(np.sqrt(np.mean((model.predict(X)-y)**2)))}


def train_schema_retriever(manifests):
    docs = []
    for m in manifests:
        docs.append(m.get("table_description", "") + " " + " ".join(label(c) + " " + c.get("description","") for c in columns(m)))
        for c in columns(m):
            docs.append(label(c) + " " + " ".join(c.get("synonyms", [])) + " " + c.get("description", ""))
        for lk in m.get("lookups", []):
            docs.append(lk.get("description", "") + " " + " ".join(map(str, lk.get("values", {}).values())))
    vec = TfidfVectorizer(ngram_range=(1,2), analyzer="word", sublinear_tf=True)
    for m in manifests:
        docs.extend([q for q,_ in q_templates(m)])
    vec.fit(docs)
    joblib.dump({"vectorizer": vec}, MODELS / "schema_tfidf.pkl")
    return {"documents": len(docs), "features": len(vec.vocabulary_)}


def main():
    manifests = load_manifests()
    results = {
        "intent": train_intent(manifests),
        "clarification": train_clarification(manifests),
        "sql_template_retriever": train_sql_template(manifests),
        "repair": train_repair(manifests),
        "confidence": train_confidence(manifests),
        "schema_retriever": train_schema_retriever(manifests),
    }
    (MODELS / "TRAINING_REPORT.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
