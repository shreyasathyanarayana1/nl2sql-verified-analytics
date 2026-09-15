"""Confidence scoring — one calibrated-looking number, always with its reasons.

Combines evidence the pipeline already produced: SQL verification history, result-level
checks, data-quality warnings, assumptions made while reading the question, and how the
schema was grounded. Below the abstention threshold the UI withholds the answer by default.
"""
from typing import Any, Dict, List
from app import local_ml

HIGH, MEDIUM = 85, 55
SEVERITY_PENALTY = {"high": 20, "medium": 10, "low": 4}


def compute(state: Dict[str, Any]) -> Dict[str, Any]:
    score = 100
    reasons: List[Dict[str, Any]] = []

    def add(text: str, impact: int) -> None:
        nonlocal score
        score += impact
        reasons.append({"text": text, "impact": impact})

    attempts = state["attempts"]
    rejected = [a for a in attempts if a["validation"] and not a["validation"]["passed"]]
    if not rejected:
        add("The first SQL draft passed all three hallucination checks", 0)
    for a in rejected:
        regenerated = a["repair"] and any("regenerate" in s for s in a["repair"]["strategies"])
        types = ", ".join(a["validation"]["hallucination_types"])
        add(f"Draft {a['attempt']} was rejected ({types}) and {'regenerated' if regenerated else 'patched'}", -12 if regenerated else -6)

    rc = state.get("result_checks")
    if rc:
        passed = [c for c in rc["checks"] if c["applicable"] and c["passed"]]
        if passed:
            add(f"{len(passed)} independent result check{'s' if len(passed) != 1 else ''} passed", 0)
        for c in rc["checks"]:
            if c["applicable"] and not c["passed"]:
                add(f"Result check failed: {c['title'].lower()} — {c['detail']}", -35)
        if state.get("result_repaired"):
            add("The answer disagreed with an independently planned query and was replaced by it", -10)

    for w in state.get("data_warnings", []):
        add(f"{w['title']}: {w['detail']}", -SEVERITY_PENALTY.get(w["severity"], 5))

    intent = state.get("intent") or {}
    for note in intent.get("notes", []):
        if note.startswith("Follow-up"):
            add("Interpreted as a follow-up to the previous question", -5)
        elif note.startswith("Ignored words"):
            add(f"{note} The answer may not reflect everything you asked", -10)
        elif note.startswith("Compared"):
            add(f"Interpretation: {note}", -3)
        elif note.startswith("Grouped"):
            add(f"Assumption: {note}", -1)
    if state.get("auto_resolved"):
        add(f"The question was ambiguous, but {state['auto_resolved']['detail'].lower()}", -2)
    if state.get("clarification_rounds", 0) >= 2:
        add("The question stayed ambiguous after clarification", -15)

    ctx = state.get("schema_context") or {}
    if any("targeted retrieval" in a for a in ctx.get("actions", [])):
        add("Schema retrieval needed a follow-up search to find every table", -3)
    for w in state.get("warnings", []):
        add(w, -8)

    score = max(5, min(99, score))
    ml_features = [
        float(len(rejected)),
        float(sum(1 for a in rejected if a["repair"] and any("regenerate" in s for s in a["repair"]["strategies"]))),
        float(sum(1 for c in (state.get("result_checks") or {}).get("checks", []) if c["applicable"] and not c["passed"])),
        float(bool(state.get("result_repaired"))),
        float(len(state.get("data_warnings", []))),
        float(any(n.startswith("Ignored words") for n in intent.get("notes", []))),
        float(any(n.startswith("Follow-up") for n in intent.get("notes", []))),
        float(any("targeted retrieval" in a for a in ctx.get("actions", []))),
        float(state.get("clarification_rounds", 0)),
    ]
    ml_score = local_ml.predict_confidence(ml_features)
    if ml_score is not None:
        score = round(0.65 * score + 0.35 * ml_score)
        reasons.append({"text": f"Local ML confidence model estimate: {ml_score:.0f}/100", "impact": 0})
    level = "high" if score >= HIGH else "medium" if score >= MEDIUM else "low"
    return {
        "score": score,
        "level": level,
        "label": {"high": "High confidence", "medium": "Check before relying on it", "low": "Low confidence"}[level],
        "abstain": level == "low",
        "reasons": reasons,
    }
