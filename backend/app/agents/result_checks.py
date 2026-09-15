"""Agent 8b — Result Verification.

The Anti-Hallucination Agent checks the SQL; this agent checks the *answer*. It runs small,
automatically derived probe queries and tests properties any correct answer must satisfy
(metamorphic testing):

* reference agreement   — an independently planned query returns the same result
* value bounds          — averages/min/max stay inside the column's real range, % in 0–100
* partition consistency — group counts add up to the total; an overall average or rate lies
                          between the group values; shares sum to 100%
* filter monotonicity   — removing a filter can never shrink a count
* unique groups         — each group appears once (duplicates signal a bad join or grouping)

Probe SQL is generated from the intent by the planner and only ever runs read-only.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from app.agents import execution, planning
from app.agents.anti_hallucination import Resolver
from app.datasets import Dataset
from app.semantic import Semantic

TOLERANCE = 0.051  # values are rounded to 1–2 decimals


def _normalize(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite")
    except sqlglot.errors.ParseError:
        return sql


def _close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


def _rows_equal(r1: List[List[Any]], r2: List[List[Any]], ordered: bool) -> bool:
    if len(r1) != len(r2):
        return False
    if not ordered:
        r1, r2 = sorted(r1, key=repr), sorted(r2, key=repr)
    return all(len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b)) for a, b in zip(r1, r2))


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def probe_intent(intent: Dict[str, Any], **changes: Any) -> Dict[str, Any]:
    probe = copy.deepcopy(intent)
    probe.update(changes)
    return probe


def run_probe(ds: Dataset, sem: Semantic, intent: Dict[str, Any], scope: Dict[str, Any], limit: Any = "keep") -> Dict[str, Any]:
    """Plans and executes an intent against the full schema (system-generated, never shown to an LLM)."""
    plan = planning.build_plan(ds, sem, intent, {"tables": sorted(ds.catalog.tables)}, scope)
    spec = plan["spec"]
    if limit != "keep" and spec["kind"] == "select":
        spec = dict(spec, limit=limit)
    sql = planning.render_sql(spec)
    return {"sql": sql, "result": execution.execute(ds.db_path, sql, max_rows=100_000), "alias": spec.get("metric_alias")}


def _check(name: str, title: str, applicable: bool, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "title": title, "applicable": applicable, "passed": passed or not applicable, "detail": detail}


def _value(result: Dict[str, Any], column: str) -> Optional[float]:
    if not result["rows"] or column not in result["columns"]:
        return None
    return result["rows"][0][result["columns"].index(column)]


# --------------------------------------------------------------------------- individual checks

def reference_agreement(ds: Dataset, plan: Dict[str, Any], final_sql: str, result: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    ref_sql = planning.render_sql(plan["spec"])
    title = "An independently planned query gives the same answer"
    if _normalize(ref_sql) == _normalize(final_sql):
        return _check("reference_agreement", title, False, True, "The executed query is identical to the planner's reference query."), None
    ref = execution.execute(ds.db_path, ref_sql)
    ordered = plan["spec"]["kind"] == "select" and bool(plan["spec"].get("order"))
    same = ref["columns"] == result["columns"] and _rows_equal(ref["rows"], result["rows"], ordered)
    detail = ("Both queries returned identical rows." if same else
              f"The reference query returned {ref['row_count']} rows that differ from the executed query's {result['row_count']}.")
    return _check("reference_agreement", title, True, same, detail), {"sql": ref_sql, "result": ref}


def value_bounds(ds: Dataset, final_sql: str, result: Dict[str, Any]) -> Dict[str, Any]:
    title = "Every value is within its possible range"
    tree = sqlglot.parse_one(final_sql, read="sqlite")
    first = tree if isinstance(tree, exp.Select) else next(tree.find_all(exp.Select))
    resolver = Resolver(tree, ds.catalog)
    tested, problems = [], []
    for idx, sel in enumerate(first.selects):
        name = sel.alias_or_name
        node = sel.this if isinstance(sel, exp.Alias) else sel
        if isinstance(node, exp.Round):
            node = node.this
        bounds: Optional[Tuple[float, float]] = None
        if name.endswith("_pct"):
            bounds = (0.0, 100.0)
        elif isinstance(node, (exp.Avg, exp.Min, exp.Max)) and isinstance(node.this, exp.Column):
            table, column, err = resolver.resolve(node.this)
            meta = ds.columns.get(column)
            if table == ds.fact_table and not err and meta and meta.minimum is not None:
                bounds = (meta.minimum, meta.maximum)
        elif isinstance(node, exp.Count):
            bounds = (0.0, float(ds.row_count))
        if bounds is None or idx >= len(result["columns"]):
            continue
        tested.append(name)
        for row in result["rows"]:
            v = row[idx]
            if isinstance(v, (int, float)) and not (bounds[0] - TOLERANCE <= v <= bounds[1] + TOLERANCE):
                problems.append(f"{name} = {_fmt(v)} is outside {_fmt(bounds[0])}–{_fmt(bounds[1])}")
                break
    if not tested:
        return _check("value_bounds", title, False, True, "No bounded values in this result.")
    return _check("value_bounds", title, True, not problems,
                  "; ".join(problems) if problems else f"Checked {', '.join(tested)} against the real column ranges.")


def _range_mode(measure: Dict[str, Any]) -> str:
    return {"COUNT": "sum", "SUM": "sum", "MAX": "max", "MIN": "min"}.get(measure["agg"], "between")


def _compare(mode: str, values: List[float], total: float) -> Tuple[bool, str]:
    if mode == "sum":
        return _close(sum(values), total, TOLERANCE), f"groups add up to {_fmt(sum(values))}; overall total {_fmt(total)}"
    if mode == "max":
        return _close(max(values), total, TOLERANCE), f"highest group {_fmt(max(values))}; overall maximum {_fmt(total)}"
    if mode == "min":
        return _close(min(values), total, TOLERANCE), f"lowest group {_fmt(min(values))}; overall minimum {_fmt(total)}"
    ok = min(values) - TOLERANCE <= total <= max(values) + TOLERANCE
    return ok, f"overall {_fmt(total)} lies {'within' if ok else 'outside'} the group range {_fmt(min(values))}–{_fmt(max(values))}"


def partition_consistency(ds: Dataset, sem: Semantic, intent: Dict[str, Any], scope: Dict[str, Any], plan: Dict[str, Any],
                          result: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = "The groups are consistent with the overall total"
    checks: List[Dict[str, Any]] = []
    if intent.get("share") and "share_pct" in result["columns"]:
        s_idx = result["columns"].index("share_pct")
        total_share = sum(r[s_idx] for r in result["rows"] if r[s_idx] is not None)
        checks.append(_check("shares_sum", "Shares add up to 100%", True, abs(total_share - 100) <= 0.1 * max(1, len(result["rows"])),
                             f"The shares add up to {total_share:.1f}%."))

    if intent["kind"] == "drivers":
        overall_probe = run_probe(ds, sem, probe_intent(intent, kind="aggregate", group_by=[], ranking=None, share=False), scope)
        total = _value(overall_probe["result"], overall_probe["alias"])
        alias = plan["spec"]["metric_alias"]
        fi, mi = result["columns"].index("factor"), result["columns"].index(alias)
        bad = []
        for factor in dict.fromkeys(r[fi] for r in result["rows"]):
            values = [r[mi] for r in result["rows"] if r[fi] == factor and isinstance(r[mi], (int, float))]
            if values and total is not None and not (min(values) - TOLERANCE <= total <= max(values) + TOLERANCE):
                bad.append(ds.columns[factor].label)
        detail = (f"For every factor the overall value {_fmt(total)} lies between its lowest and highest level." if not bad
                  else f"The overall value {_fmt(total)} falls outside the level range for: {', '.join(bad)}.")
        checks.append(_check("partition_consistency", title, total is not None, not bad, detail))
        return checks

    alias = plan["spec"].get("metric_alias")
    if intent["kind"] not in ("aggregate",) or len(intent["group_by"]) != 1 or result["row_count"] < 2 or alias not in result["columns"]:
        checks.append(_check("partition_consistency", title, False, True, "Only applies to results split into groups."))
        return checks
    measure = intent["measures"][0]
    idx = result["columns"].index(alias)
    values = [r[idx] for r in result["rows"] if isinstance(r[idx], (int, float))]
    probe = run_probe(ds, sem, probe_intent(intent, group_by=[], ranking=None, share=False), scope)
    total = _value(probe["result"], probe["alias"])
    if total is None or not values:
        checks.append(_check("partition_consistency", title, False, True, "No overall total to compare with."))
        return checks
    ok, detail = _compare(_range_mode(measure), values, total)
    checks.append(_check("partition_consistency", title, True, ok, detail[:1].upper() + detail[1:] + "."))
    return checks


def filter_monotonicity(ds: Dataset, sem: Semantic, intent: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    title = "Removing a filter never shrinks the count"
    removable = intent["filters"][:2]
    counting = intent["kind"] == "list" or any(m["agg"] == "COUNT" for m in intent["measures"])
    if not removable or not counting:
        return _check("filter_monotonicity", title, False, True, "Only applies to filtered counts and lists.")

    def measure(it: Dict[str, Any]) -> Optional[float]:
        if it["kind"] == "list":
            return run_probe(ds, sem, it, scope, limit=None)["result"]["row_count"]
        probe = run_probe(ds, sem, probe_intent(it, group_by=[], ranking=None, share=False), scope)
        return _value(probe["result"], probe["alias"]) or 0

    base = measure(intent)
    findings, ok = [], True
    for f in removable:
        relaxed = probe_intent(intent, filters=[g for g in intent["filters"] if g is not f])
        value = measure(relaxed)
        holds = value >= base
        ok = ok and holds
        findings.append(f"without “{f['label']}”: {_fmt(value)} {'≥' if holds else '<'} {_fmt(base)}")
    return _check("filter_monotonicity", title, True, ok, "; ".join(findings))


def unique_groups(intent: Dict[str, Any], plan: Dict[str, Any], result: Dict[str, Any], sem: Semantic) -> Dict[str, Any]:
    title = "Each group appears exactly once"
    kind = intent["kind"]
    if kind in ("list", "correlation") or result["row_count"] < 2 or (kind == "ranking" and intent["ranking"]["target"] == "rows"):
        return _check("unique_groups", title, False, True, "Only applies to grouped results.")
    if kind == "drivers":
        key_names = ["factor", "level"]
    elif kind == "distribution":
        key_names = ["bucket_start"]
    else:
        key_names = [f"{g['col']}_band" if g["bin"] else g["col"] for g in intent["group_by"]]
    key_idx = [result["columns"].index(k) for k in key_names if k in result["columns"]]
    if not key_idx:
        return _check("unique_groups", title, False, True, "No group columns.")
    keys = [tuple(r[i] for i in key_idx) for r in result["rows"]]
    dupes = len(keys) - len(set(keys))
    limit = (intent.get("ranking") or {}).get("limit")
    over_limit = limit is not None and result["row_count"] > limit
    ok = dupes == 0 and not over_limit
    detail = (f"{len(set(keys))} distinct groups." if ok else
              f"{dupes} duplicate group rows" if dupes else f"returned {result['row_count']} rows but the question asked for {limit}")
    return _check("unique_groups", title, True, ok, detail)


# --------------------------------------------------------------------------- agent entry

def verify_result(ds: Dataset, sem: Semantic, intent: Dict[str, Any], plan: Dict[str, Any], scope: Dict[str, Any],
                  final_sql: str, result: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    reference = None

    def guarded(name: str, fn, *args):
        try:
            out = fn(*args)
            return out if isinstance(out, list) else [out]
        except Exception as e:  # a probe that can't be planned is not evidence of an error
            return [_check(name, name.replace("_", " ").capitalize(), False, True, f"Skipped ({type(e).__name__}: {e}).")]

    try:
        ref_check, reference = reference_agreement(ds, plan, final_sql, result)
        checks.append(ref_check)
    except Exception as e:
        checks.append(_check("reference_agreement", "An independently planned query gives the same answer", False, True,
                             f"Skipped ({type(e).__name__})."))
    checks += guarded("value_bounds", value_bounds, ds, final_sql, result)
    checks += guarded("partition_consistency", partition_consistency, ds, sem, intent, scope, plan, result)
    checks += guarded("filter_monotonicity", filter_monotonicity, ds, sem, intent, scope)
    checks += guarded("unique_groups", unique_groups, intent, plan, result, sem)

    applicable = [c for c in checks if c["applicable"]]
    return {"passed": all(c["passed"] for c in applicable), "checks": checks, "applicable": len(applicable),
            "failed": [c["name"] for c in applicable if not c["passed"]], "reference": reference}
