"""Data Quality Agent — plain-English warnings about how far a correct answer can be trusted.

A query can be perfectly correct and still mislead: an average over three rows, a value that
silently drops everyone with a missing measurement, a "top 10" cut in the middle of a tie,
a comparison that reverses once you control for another variable (Simpson's paradox), or a
value outside its documented range. Each warning is backed by a probe query.
"""
import math
from typing import Any, Dict, List, Optional

from app.agents.result_checks import probe_intent, run_probe
from app.datasets import Dataset
from app.semantic import Semantic

MIN_GROUP_SIZE = 10


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _warning(code: str, severity: str, title: str, detail: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "title": title, "detail": detail}


def _group_key(g: Dict[str, Any]) -> str:
    return f"{g['col']}_band" if g["bin"] else g["col"]


def small_groups(ds: Dataset, sem: Semantic, intent, scope, plan, result) -> List[Dict[str, str]]:
    if intent["kind"] not in ("aggregate", "ranking") or not intent["measures"] or intent["measures"][0]["agg"] in ("COUNT", "SUM"):
        return []
    if intent["kind"] == "ranking" and intent["ranking"]["target"] == "rows":
        return []
    count_col = sem.count_alias()
    if not intent["group_by"]:
        if not intent["filters"]:
            return []
        probe = run_probe(ds, sem, probe_intent(intent, measures=[{"col": None, "agg": "COUNT"}], ranking=None, kind="aggregate"), scope)
        n = probe["result"]["rows"][0][0] if probe["result"]["rows"] else 0
        if n < MIN_GROUP_SIZE:
            return [_warning("small_sample", "high" if n < 3 else "medium", f"Only {n} {ds.plural if n != 1 else ds.singular} match",
                             "A figure based on so few records can swing a lot.")]
        return []
    if count_col not in result["columns"]:
        return []
    key = _group_key(intent["group_by"][0])
    ki, ni = result["columns"].index(key), result["columns"].index(count_col)
    small = [(r[ki], r[ni]) for r in result["rows"] if r[ni] is not None and r[ni] < MIN_GROUP_SIZE]
    if not small:
        return []
    listed = ", ".join(f"{'(missing)' if g is None else g} ({n})" for g, n in small[:5])
    return [_warning("small_groups", "high" if any(n < 3 for _, n in small) else "medium", "Some groups are very small",
                     f"{listed} {'is' if len(small) == 1 else 'are'} based on fewer than {MIN_GROUP_SIZE} {ds.plural}, "
                     "so treat those values with caution.")]


def missing_values(ds: Dataset, sem: Semantic, intent, scope, plan, result) -> List[Dict[str, str]]:
    used = [m["col"] for m in intent["measures"] if m.get("col")] + [g["col"] for g in intent["group_by"]]
    if intent["kind"] == "drivers":
        return []
    warnings = []
    for name in dict.fromkeys(used):
        col = ds.columns[name]
        if col.nulls == 0:
            continue
        probe = run_probe(ds, sem, probe_intent(intent, kind="aggregate", measures=[{"col": None, "agg": "COUNT"}], group_by=[],
                                                ranking=None, share=False,
                                                filters=intent["filters"] + [{"col": name, "op": "is_null", "value": None, "label": ""}]),
                          scope)
        missing = probe["result"]["rows"][0][0] if probe["result"]["rows"] else 0
        if not missing:
            continue
        base = run_probe(ds, sem, probe_intent(intent, kind="aggregate", measures=[{"col": None, "agg": "COUNT"}], group_by=[],
                                               ranking=None, share=False), scope)
        total = base["result"]["rows"][0][0] or 1
        pct = 100 * missing / total
        severity = "high" if pct >= 30 else "medium" if pct >= 5 else "low"
        grouped = name in [g["col"] for g in intent["group_by"]]
        effect = "they appear as a separate “missing” group" if grouped else "they are left out of this figure"
        warnings.append(_warning("missing_values", severity, f"{col.label[:1].upper() + col.label[1:]} is missing for some {ds.plural}",
                                 f"{missing} of {total} {ds.plural} ({pct:.0f}%) have no recorded {col.label}; {effect}."))
    return warnings


def out_of_range(ds: Dataset, sem: Semantic, intent, scope, plan, result) -> List[Dict[str, str]]:
    warnings = []
    ranges = {c["name"]: c["valid_range"] for c in ds.manifest["columns"] if c.get("valid_range")}
    used = {m["col"] for m in intent["measures"] if m.get("col")} | {g["col"] for g in intent["group_by"]}
    for name in used & set(ranges):
        col = ds.columns[name]
        lo, hi = ranges[name]
        if col.minimum is not None and (col.minimum < lo or col.maximum > hi):
            probe = run_probe(ds, sem, probe_intent(intent, kind="aggregate", measures=[{"col": None, "agg": "COUNT"}], group_by=[],
                                                    ranking=None, share=False,
                                                    filters=intent["filters"] + [{"col": name, "op": ">", "value": hi, "label": ""}]), scope)
            n = probe["result"]["rows"][0][0] if probe["result"]["rows"] else 0
            if n:
                warnings.append(_warning("out_of_range", "low", "Possible data-entry error",
                                         f"{n} {ds.singular if n == 1 else ds.plural} {'has' if n == 1 else 'have'} "
                                         f"{'an' if col.label[0] in 'aeiou' else 'a'} {col.label} above "
                                         f"the documented maximum of {hi} (highest recorded: {_fmt(col.maximum)})."))
    return warnings


def ties_at_cutoff(ds: Dataset, sem: Semantic, intent, scope, plan, result) -> List[Dict[str, str]]:
    ranking = intent.get("ranking")
    alias = plan["spec"].get("metric_alias")
    if not ranking or alias not in result["columns"] or result["row_count"] < ranking["limit"]:
        return []
    probe = run_probe(ds, sem, intent, scope, limit=ranking["limit"] + 500)
    res = probe["result"]
    vi, limit = res["columns"].index(alias), ranking["limit"]
    if len(res["rows"]) <= limit or res["rows"][limit][vi] != res["rows"][limit - 1][vi]:
        return []
    cutoff = res["rows"][limit - 1][vi]
    tied = sum(1 for r in res["rows"] if r[vi] == cutoff)
    shown = sum(1 for r in result["rows"] if r[result["columns"].index(alias)] == cutoff)
    noun = ds.plural if ranking["target"] == "rows" else "groups"
    detail = (f"{tied} {noun} share the top value of {_fmt(cutoff)}; only one is shown." if limit == 1 else
              f"{tied} {noun} are tied at {_fmt(cutoff)} but only {shown} fit in the top {limit}, so which ones appear is arbitrary.")
    return [_warning("tie_at_cutoff", "low", "Tie at the cut-off", detail)]


def simpsons_paradox(ds: Dataset, sem: Semantic, intent, scope, plan, result) -> List[Dict[str, str]]:
    groups = intent["group_by"]
    alias = plan["spec"].get("metric_alias")
    if intent["kind"] != "aggregate" or len(groups) != 1 or groups[0]["bin"] or not intent["measures"] \
            or intent["measures"][0]["agg"] not in ("AVG", "RATE") or alias not in result["columns"] or result["row_count"] < 2:
        return []
    group_col = groups[0]["col"]
    used = {group_col} | {f["col"] for f in intent["filters"]}
    stratifier = next((s for s in ds.manifest.get("stratify_by", []) if s not in used), None)
    if not stratifier:
        return []
    gi, vi = result["columns"].index(group_col), result["columns"].index(alias)
    ranked = sorted((r for r in result["rows"] if r[gi] is not None and r[vi] is not None), key=lambda r: r[vi], reverse=True)
    if len(ranked) < 2 or ranked[0][vi] == ranked[1][vi]:
        return []
    (a, va), (b, vb) = (ranked[0][gi], ranked[0][vi]), (ranked[1][gi], ranked[1][vi])
    probe = run_probe(ds, sem, probe_intent(intent, group_by=groups + [{"col": stratifier, "bin": None}], ranking=None, share=False), scope)
    res = probe["result"]
    pgi, psi, pvi = res["columns"].index(group_col), res["columns"].index(stratifier), res["columns"].index(probe["alias"])
    by_stratum: Dict[Any, Dict[Any, float]] = {}
    for row in res["rows"]:
        if row[psi] is not None and row[pvi] is not None:
            by_stratum.setdefault(row[psi], {})[row[pgi]] = row[pvi]
    strata = [s for s, g in by_stratum.items() if a in g and b in g]
    flips = [s for s in strata if by_stratum[s][b] > by_stratum[s][a]]
    if len(strata) < 2 or len(flips) <= len(strata) / 2:
        return []
    label = sem.measure_label(intent["measures"][0])
    strat_label = ds.columns[stratifier].label
    return [_warning("simpsons_paradox", "medium", f"The comparison reverses within each {strat_label}",
                     f"{a} has a higher {label} than {b} overall ({_fmt(va)} vs {_fmt(vb)}), but {b} is higher within "
                     f"{len(flips)} of {len(strata)} {strat_label} groups ({', '.join(map(str, flips))}). The overall gap is largely "
                     f"due to different {strat_label} mixes (Simpson's paradox).")]


def thin_sample(ds: Dataset, sem: Semantic, intent, scope, plan, result) -> List[Dict[str, str]]:
    if intent["kind"] == "correlation" and result["row_count"] < 30:
        return [_warning("small_sample", "high", "Too few records for a reliable relationship",
                         f"Only {result['row_count']} {ds.plural} have both values, so any correlation is likely noise.")]
    return []


CHECKS = [missing_values, small_groups, out_of_range, ties_at_cutoff, simpsons_paradox, thin_sample]


def assess(ds: Dataset, sem: Semantic, intent: Dict[str, Any], scope: Dict[str, Any], plan: Dict[str, Any],
           result: Dict[str, Any]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    for check in CHECKS:
        try:
            warnings += check(ds, sem, intent, scope, plan, result)
        except Exception:
            continue  # warnings are advisory; a failing probe must never break the answer
    return warnings


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    n = len(pairs)
    if n < 3:
        return None
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)
