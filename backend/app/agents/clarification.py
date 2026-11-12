
import re
from typing import Any, Dict, List, Set

from app.agents.intent import parser_for
from app.config import AMBIGUITY_THRESHOLD
from app.datasets import Dataset
from app import local_ml


def _measure_options(ds: Dataset, question: str, denied: Set[str], numeric_only: bool) -> List[Dict[str, Any]]:
    cols = [ds.primary_measure] + [c for c in ds.columns.values() if c.role == "measure" and c is not ds.primary_measure]
    if not numeric_only:
        cols += [c for c in ds.columns.values() if c.role == "flag" and c.outcome and c is not ds.primary_measure]
    options = []
    for c in cols:
        if c.name in denied or (numeric_only and c.role != "measure") or c.nulls > 0.5 * ds.row_count:
            continue
        phrase = f"{c.label} rate" if c.role == "flag" else c.label
        options.append({"id": c.name, "label": phrase[:1].upper() + phrase[1:], "question": f"{question} ({phrase})"})
    return options[:4]


def _signals(ds: Dataset, intent: Dict[str, Any], denied: Set[str]) -> List[Dict[str, Any]]:
    raw = intent["raw"]
    signals: List[Dict[str, Any]] = []
    parser = parser_for(ds)

    if intent["needs_measure"]:
        numeric_only = bool(intent.get("ranking") and intent["ranking"]["target"] == "rows")
        unknown = intent["unknown_terms"]
        suggestions = []
        for term in unknown:
            for c in parser.suggest_columns(term, roles=("measure",) if numeric_only else ("measure", "flag"), cutoff=0.75):
                if c.name not in denied and c.name not in [s["id"] for s in suggestions]:
                    new_q = re.sub(r"\b%s\b" % re.escape(term), c.label, raw, flags=re.I)
                    suggestions.append({"id": c.name, "label": c.label[:1].upper() + c.label[1:], "question": new_q})
        if unknown and suggestions:
            signals.append({
                "type": "unknown_measure", "weight": 0.8,
                "detail": f"This dataset has no column called “{unknown[0]}”.",
                "question": f"There's no “{unknown[0]}” in {ds.name}. Did you mean one of these?",
                "options": suggestions[:4],
            })
        else:
            what = f"“{unknown[0]}”" if unknown else "that"
            signals.append({
                "type": "missing_measure", "weight": 0.7,
                "detail": "The question ranks or aggregates but doesn't say which measure to use"
                          + (f" (“{unknown[0]}” isn't a column in this dataset)." if unknown else "."),
                "question": f"Which measure should I use for {what}?",
                "options": _measure_options(ds, raw, denied, numeric_only),
            })
    
    # Detect an unrecognized grouping requested after "by".
    group_match = re.search(r"\bby\s+([^,?.;]+)", raw, re.I)
    if group_match and intent["unknown_terms"] and not intent.get("group_by"):
        start, end = group_match.span(1)
        options = []

        for col in ds.columns.values():
            if col.name in denied:
                continue

            label = col.label[:1].upper() + col.label[1:]
            options.append({
                "id": col.name,
                "label": label,
                "question": raw[:start] + col.label + raw[end:]
            })

        if options:
            signals.append({
                "type": "unsupported_intent",
                "weight": 1.0,
                "detail": (
                    f"Couldn't identify the requested grouping "
                    f"“{group_match.group(1).strip()}” in {ds.name}."
                ),
                "question": "Which column did you want to group by?",
                "options": options[:5],
            })

    for amb in intent["ambiguous_values"]:
        options = []
        for name in amb["candidates"]:
            if name in denied:
                continue
            col = ds.columns[name]
            new_q = raw[:0] + re.sub(r"\b%s\b" % re.escape(amb["phrase"]), f"{amb['phrase']} {col.label}", raw, count=1, flags=re.I)
            options.append({"id": name, "label": col.label[:1].upper() + col.label[1:], "question": new_q})
        if len(options) > 1:
            signals.append({
                "type": "ambiguous_value", "weight": 0.75,
                "detail": f"“{amb['phrase']}” is a value in {len(amb['candidates'])} different columns.",
                "question": f"“{amb['phrase'].capitalize()}” what?",
                "options": options[:5],
            })

    # Words that matched nothing. Answering without them silently changes the question
    # ("students who scored in cs" would become "all students"), so ask first.
    unknown = intent["unknown_terms"]
    if unknown and not signals and intent["kind"] != "unknown" and not intent["needs_measure"]:
        options = []
        for term in unknown:
            for c in parser.suggest_columns(term, roles=("measure", "flag", "category"), cutoff=0.75):
                if c.name not in denied and c.name not in [o["id"] for o in options]:
                    new_q = re.sub(r"\b%s\b" % re.escape(term), c.label, raw, flags=re.I)
                    options.append({"id": c.name, "label": c.label[:1].upper() + c.label[1:], "question": new_q})
        without = re.sub(r"\b(?:%s)\b" % "|".join(re.escape(t) for t in unknown), " ", raw, flags=re.I)
        without = re.sub(r"\s+", " ", without).strip()
        quoted = ", ".join(f"“{t}”" for t in unknown)
        options = options[:3] + [{"id": "ignore_unmatched", "label": f"Answer without {quoted}", "question": without}]
        signals.append({
            "type": "unmatched_terms", "weight": 0.7,
            "detail": f"Couldn't match {quoted} to anything in {ds.name}; answering without {'it' if len(unknown) == 1 else 'them'} would change the question.",
            "question": f"I couldn't find {quoted} in {ds.name}. What did you mean?",
            "options": options,
        })

    if intent["kind"] == "unknown":
        unknown = intent["unknown_terms"]
        suggestions = ds.manifest.get("suggestions", [])
        picks = [q for g in suggestions if not g["group"].startswith("Needs") for q in g["questions"][:1]][:4]
        signals.append({
            "type": "unsupported_intent", "weight": 0.9,
            "detail": (f"Couldn't match “{', '.join(unknown[:3])}” to anything in {ds.name}." if unknown
                       else "No measure, grouping or filter could be identified."),
            "question": f"I couldn't work out what to look up in {ds.name}. Is it one of these?",
            "options": [{"id": f"s{i}", "label": q, "question": q, "replace": q} for i, q in enumerate(picks)],
        })
    return signals


def assess(ds: Dataset, intent: Dict[str, Any], suppressed: List[str], denied: Set[str]) -> Dict[str, Any]:
    signals = [s for s in _signals(ds, intent, denied) if s["type"] not in suppressed]
    score = round(min(1.0, sum(s["weight"] for s in signals)), 2)
    askable = [s for s in signals if s.get("options")]
    needs = score > AMBIGUITY_THRESHOLD and bool(askable)
    top = max(askable, key=lambda s: s["weight"]) if needs else None
    ml_features = [
        float(bool(intent.get("measures"))),
        float(intent.get("needs_measure", False)),
        float(intent.get("kind") == "unknown"),
        float(bool(intent.get("ranking"))),
        float(not intent.get("measures") and not intent.get("list_columns") and not intent.get("filters")),
    ]
    ml_check = local_ml.predict_clarification(ml_features)
    return {
        "ambiguity_score": score,
        "threshold": AMBIGUITY_THRESHOLD,
        "signals": [{"type": s["type"], "weight": s["weight"], "detail": s["detail"]} for s in signals],
        "needs_clarification": needs,
        "signal": top["type"] if top else None,
        "question": top["question"] if top else None,
        "options": top["options"] if top else [],
        "ml_classifier": ml_check,
    }
