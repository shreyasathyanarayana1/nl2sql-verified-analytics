"""Suggested next questions, built from the dataset's own columns and phrased so the Intent Agent reads them unambiguously."""
from typing import Any, Dict, List, Optional, Set

from app.agents import access_control
from app.datasets import ColumnMeta, Dataset


def _phrase(col: ColumnMeta) -> str:
    return f"{col.label} rate" if col.role == "flag" else f"average {col.label}"


def suggest(ds: Dataset, intent: Dict[str, Any], result: Optional[Dict[str, Any]], viz: Optional[Dict[str, Any]],
            scope: Dict[str, Any], asked: str) -> List[str]:
    allowed = lambda c: not access_control.is_denied(scope, ds.fact_table, c.name) and c.nulls < 0.5 * ds.row_count
    measure_col = next((ds.columns[m["col"]] for m in intent.get("measures", []) if m.get("col")), ds.primary_measure)
    if measure_col.role == "measure" and intent["kind"] == "correlation":
        measure_col = ds.columns[intent["measures"][-1]["col"]]
    used: Set[str] = {g["col"] for g in intent.get("group_by", [])} | {f["col"] for f in intent.get("filters", [])} | {measure_col.name}
    categories = [c for c in ds.columns.values() if c.role in ("category", "flag") and c.name not in used and allowed(c)]
    stratifiers = [ds.columns[s] for s in ds.manifest.get("stratify_by", []) if s in ds.columns and s not in used]
    preferred: List[ColumnMeta] = []
    for c in stratifiers + [c for c in categories if c.order or c.lookup] + categories:
        if c.name not in [p.name for p in preferred]:
            preferred.append(c)
    numeric = [c for c in ds.columns.values() if c.role == "measure" and c.name not in used and allowed(c)]
    out: List[str] = []

    if intent["kind"] == "drivers" and viz and viz.get("drivers") and viz["drivers"]["factors"]:
        for f in viz["drivers"]["factors"][:2]:
            col = ds.columns[f["factor"]]
            out.append(f"{_phrase(measure_col)} by {col.label}")
        if numeric:
            out.append(f"{numeric[0].label} vs {measure_col.label}" if measure_col.role == "measure" else f"{_phrase(measure_col)} by {numeric[0].label}")
    else:
        for c in preferred[:2]:
            out.append(f"{_phrase(measure_col)} by {c.label}")
        if intent["kind"] != "drivers":
            out.append(f"what factors affect {measure_col.label}?")
        if measure_col.role == "measure" and intent["kind"] != "distribution":
            out.append(f"{measure_col.label} distribution")
        if measure_col.role == "measure" and numeric and intent["kind"] != "correlation":
            out.append(f"{numeric[0].label} vs {measure_col.label}")
    seen, unique = {asked.strip().lower()}, []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q[:1].upper() + q[1:])
    return unique[:4]
