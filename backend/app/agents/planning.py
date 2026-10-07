"""Agent 4 — Query Planning (dataset-agnostic).

Turns the clarified intent plus the retrieved schema context into (a) numbered plain-English
steps and (b) a structured query spec the SQL Generation Agent renders. Planning is limited
to the retrieved tables: if a needed table was not grounded, it fails loudly instead of
letting the generator invent one. Restricted columns block the plan before any SQL exists.
"""
from typing import Any, Dict, List, Optional

from app.agents import access_control
from app.datasets import ColumnMeta, Dataset
from app.semantic import Semantic

ROW_LIMIT = 200


class PlanningError(Exception):
    def __init__(self, message: str, status: str = "failed"):
        super().__init__(message)
        self.status = status


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def build_plan(ds: Dataset, sem: Semantic, intent: Dict[str, Any], schema_ctx: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
    kind = intent["kind"]
    if kind == "unknown" or intent.get("needs_measure"):
        raise PlanningError("I still couldn't tell what to measure — try naming a column, e.g. "
                            f"“average {ds.primary_measure.label} by …”.", "needs_clarification")

    blocked = [c for c in sem.referenced_columns(intent) if access_control.is_denied(scope, ds.fact_table, c)]
    if blocked:
        labels = ", ".join(ds.columns[c].label for c in blocked)
        raise PlanningError(f"Your role ({scope['role']}) isn't permitted to use {labels}.", "blocked")

    needed = sem.required_tables(intent)
    factors: List[ColumnMeta] = []
    if kind == "drivers":
        restricted = {c for c in ds.columns if access_control.is_denied(scope, ds.fact_table, c)}
        factors = sem.driver_factors(intent, restricted)
        needed = {ds.fact_table} | {c.lookup for c in factors if c.lookup}
    missing = needed - set(schema_ctx["tables"])
    if missing:
        raise PlanningError(f"Schema grounding did not retrieve {', '.join(sorted(missing))}; refusing to generate SQL over unseen tables.")

    if kind == "drivers":
        return _drivers_plan(ds, sem, intent, factors, scope)
    if kind == "correlation":
        return _correlation_plan(ds, sem, intent)
    if kind == "distribution":
        return _distribution_plan(ds, sem, intent)
    if kind == "list" or (kind == "ranking" and intent["ranking"]["target"] == "rows"):
        return _rows_plan(ds, sem, intent)
    return _aggregate_plan(ds, sem, intent)


# --------------------------------------------------------------------------- shared pieces

def _where(sem: Semantic, intent: Dict[str, Any], joins: List[str]) -> List[str]:
    clauses = []
    for f in intent["filters"]:
        col = sem.col(f["col"])
        join = sem.lookup_join(col)
        if join and join not in joins:
            joins.append(join)
        clauses.append(sem.filter_sql(f))
    return clauses


def _filter_steps(intent: Dict[str, Any]) -> List[str]:
    labels = [f["label"] for f in intent["filters"]]
    return ["Keep only the rows that match: " + "; ".join(labels) + "."] if labels else []


def _join_steps(sem: Semantic, joins: List[str]) -> List[str]:
    steps = []
    for j in joins:
        table = j.split()[1]
        col = next(c for c in sem.ds.columns.values() if c.lookup == table)
        steps.append(f"Join {table} to turn {col.label} codes into names ({sem.ds.fact_table}.{col.name} → {table}.code).")
    return steps


def _tables(ds: Dataset, joins: List[str]) -> List[str]:
    return sorted({ds.fact_table} | {j.split()[1] for j in joins})


def _spec(**parts: Any) -> Dict[str, Any]:
    base = {"kind": "select", "root": None, "joins": [], "select": [], "where": [], "group": [], "having": [],
            "order": [], "limit": None, "metric_alias": None}
    base.update(parts)
    return base


# --------------------------------------------------------------------------- plans

def _aggregate_plan(ds: Dataset, sem: Semantic, intent: Dict[str, Any]) -> Dict[str, Any]:
    joins: List[str] = []
    select: List[str] = []
    group: List[str] = []
    order: List[str] = []
    steps = [f"Start from the {ds.fact_table} table."]
    group_labels = []

    for g in intent["group_by"]:
        col = sem.col(g["col"])
        if g["bin"]:
            expr, alias = sem.band_expr(col, g["bin"]), f"{col.name}_band"
            group_labels.append(f"{col.label} bands of {_fmt(g['bin'])}{col.unit_suffix()}")
        elif col.role == "flag":
            expr, alias = sem.flag_label_expr(col), col.name
            group_labels.append(f"with / without {col.label}")
        else:
            join = sem.lookup_join(col)
            if join and join not in joins:
                joins.append(join)
            expr, alias = sem.display_expr(col), col.name
            group_labels.append(col.label)
        select.append(f"{expr} AS {alias}")
        group.append(expr)

    metric_aliases = []
    for m in intent["measures"]:
        expr, alias = sem.measure_expr(m)
        select.append(f"{expr} AS {alias}")
        metric_aliases.append(alias)
    count_alias = sem.count_alias()
    if intent["group_by"] and count_alias not in metric_aliases:
        select.append(f"COUNT(*) AS {count_alias}")
    if intent.get("share"):
        select.append("ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct")

    where = _where(sem, intent, joins)
    main_alias = metric_aliases[0]
    ranking = intent.get("ranking")
    limit = None
    if ranking:
        order.append(f"{main_alias} {ranking['direction']}")
        limit = ranking["limit"]
    elif intent["group_by"]:
        g = intent["group_by"][0]
        col = sem.col(g["col"])
        alias = f"{col.name}_band" if g["bin"] else col.name
        ordered = sem.order_expr(col, sem.display_expr(col)) if not g["bin"] and col.order and col.role == "category" else None
        null_last = f"{alias} IS NULL"
        order += [null_last, ordered or (alias if g["bin"] or col.role == "flag" else f"{main_alias} DESC")]

    steps += _join_steps(sem, joins) + _filter_steps(intent)
    if group_labels:
        steps.append("Group rows by " + ", ".join(group_labels) + ".")
    steps.append("Compute the " + " and ".join(sem.measure_label(m) for m in intent["measures"]) +
                 (f" for each group, plus how many {ds.plural} are in it." if intent["group_by"] else "."))
    for m in intent["measures"]:
        if m["agg"] == "RATE":
            col = sem.col(m["col"])
            steps.append(f"The {col.label} rate is the share of {ds.plural} where {col.name} = {sem.literal(col, col.positive)} "
                         "(missing values are left out).")
    if intent.get("share"):
        steps.append("Divide each group's count by the total to get its share (%).")
    if ranking:
        steps.append(f"Sort {'highest' if ranking['direction'] == 'DESC' else 'lowest'} first and keep the top {limit}.")

    spec = _spec(root=ds.fact_table, joins=joins, select=select, where=where, group=group, order=order, limit=limit,
                 metric_alias=main_alias)
    return {"steps": steps, "spec": spec, "tables": _tables(ds, joins)}


def _rows_plan(ds: Dataset, sem: Semantic, intent: Dict[str, Any]) -> Dict[str, Any]:
    joins: List[str] = []
    t = sem.t
    select = [f"{t}.{ds.id_column}"]
    shown: List[str] = []

    def show(col: ColumnMeta) -> None:
        if col.name in shown:
            return
        shown.append(col.name)
        if col.lookup:
            join = sem.lookup_join(col)
            if join not in joins:
                joins.append(join)
            select.append(f"{sem.display_expr(col)} AS {col.name}")
        else:
            select.append(f"{t}.{col.name}")

    ranking = intent.get("ranking")
    measure = sem.col(intent["measures"][0]["col"]) if intent["measures"] and intent["measures"][0].get("col") else None
    for name in [f["col"] for f in intent["filters"]] + intent["list_columns"]:
        show(sem.col(name))
    show(measure or ds.primary_measure)

    where = _where(sem, intent, joins)
    steps = [f"Start from the {ds.fact_table} table."] + _join_steps(sem, joins) + _filter_steps(intent)
    if ranking and measure:
        where.append(f"{t}.{measure.name} IS NOT NULL")
        order = [f"{t}.{measure.name} {ranking['direction']}"]
        limit = ranking["limit"]
        steps.append(f"Sort {ds.plural} by {measure.label}, {'highest' if ranking['direction'] == 'DESC' else 'lowest'} first, "
                     f"and keep the top {limit}.")
    else:
        order = [f"{t}.{ds.id_column}"]
        limit = ROW_LIMIT
        steps.append(f"Return the matching {ds.plural} (up to {ROW_LIMIT} rows).")
    spec = _spec(root=ds.fact_table, joins=joins, select=select, where=where, order=order, limit=limit,
                 metric_alias=measure.name if measure else None)
    return {"steps": steps, "spec": spec, "tables": _tables(ds, joins)}


def _distribution_plan(ds: Dataset, sem: Semantic, intent: Dict[str, Any]) -> Dict[str, Any]:
    col = sem.col(intent["measures"][0]["col"])
    width = col.bin_width or 10
    joins: List[str] = []
    expr = sem.band_expr(col, width)
    where = _where(sem, intent, joins) + [f"{sem.t}.{col.name} IS NOT NULL"]
    spec = _spec(root=ds.fact_table, joins=joins, select=[f"{expr} AS bucket_start", f"COUNT(*) AS {sem.count_alias()}"],
                 where=where, group=["bucket_start"], order=["bucket_start"], metric_alias=sem.count_alias())
    steps = [f"Start from the {ds.fact_table} table."] + _join_steps(sem, joins) + _filter_steps(intent) + [
        f"Leave out {ds.plural} with no recorded {col.label}.",
        f"Place each {ds.singular} into a {col.label} band {_fmt(width)}{col.unit_suffix()} wide.",
        f"Count the {ds.plural} in each band.",
    ]
    return {"steps": steps, "spec": spec, "tables": _tables(ds, joins),
            "bucket": {"width": width, "column": col.name, "unit": col.unit, "integer": col.type == "INTEGER"}}


def _correlation_plan(ds: Dataset, sem: Semantic, intent: Dict[str, Any]) -> Dict[str, Any]:
    x, y = (sem.col(m["col"]) for m in intent["measures"][:2])
    joins: List[str] = []
    where = _where(sem, intent, joins) + [f"{sem.t}.{x.name} IS NOT NULL", f"{sem.t}.{y.name} IS NOT NULL"]
    spec = _spec(root=ds.fact_table, joins=joins, select=[f"{sem.t}.{ds.id_column}", f"{sem.t}.{x.name}", f"{sem.t}.{y.name}"],
                 where=where, order=[f"{sem.t}.{ds.id_column}"], metric_alias=y.name)
    steps = [f"Start from the {ds.fact_table} table."] + _join_steps(sem, joins) + _filter_steps(intent) + [
        f"Keep {ds.plural} who have both {x.label} and {y.label} recorded.",
        f"Return one row per {ds.singular} so the relationship can be plotted and a correlation computed.",
    ]
    return {"steps": steps, "spec": spec, "tables": _tables(ds, joins), "axes": {"x": x.name, "y": y.name}}


def _drivers_plan(ds: Dataset, sem: Semantic, intent: Dict[str, Any], factors: List[ColumnMeta], scope: Dict[str, Any]) -> Dict[str, Any]:
    outcome = intent["measures"][0]
    metric_expr, metric_alias = sem.measure_expr(outcome)
    branches = []
    all_joins: List[str] = []
    for col in factors:
        joins: List[str] = []
        if col.role == "measure":
            level = f"CAST({sem.band_expr(col, col.bin_width or 10)} AS TEXT)"
        elif col.role == "flag":
            level = sem.flag_label_expr(col)
        else:
            join = sem.lookup_join(col)
            if join:
                joins.append(join)
            level = sem.display_expr(col)
        where = _where(sem, intent, joins) + [f"{sem.t}.{col.name} IS NOT NULL"]
        all_joins += joins
        branches.append(_spec(root=ds.fact_table, joins=joins,
                              select=[f"'{col.name}' AS factor", f"{level} AS level", f"{metric_expr} AS {metric_alias}", "COUNT(*) AS n"],
                              where=where, group=["level"], metric_alias=metric_alias))
    restricted = [ds.columns[c].label for c in ds.columns if access_control.is_denied(scope, ds.fact_table, c)]
    sparse = [c.label for c in ds.columns.values() if c.nulls > 0.5 * ds.row_count and c.name != outcome.get("col")]
    steps = [f"Start from the {ds.fact_table} table."] + _filter_steps(intent) + [
        f"For each of {len(factors)} factors, split {ds.plural} into its levels (numeric factors into bands).",
        f"Compute the {sem.measure_label(outcome)} and the number of {ds.plural} in every level.",
        "Stack all factors into one result so they can be compared side by side.",
        "Rank factors by the gap between their best and worst level, ignoring levels with very few rows.",
    ]
    notes = []
    if restricted:
        notes.append(f"Left out restricted columns: {', '.join(restricted)}.")
    if sparse:
        notes.append(f"Left out mostly-missing columns: {', '.join(sparse)}.")
    return {"steps": steps, "spec": {"kind": "union", "branches": branches, "metric_alias": metric_alias},
            "tables": _tables(ds, all_joins), "notes": notes, "factors": {c.name: c.label for c in factors}}


# --------------------------------------------------------------------------- rendering

def render_sql(spec: Dict[str, Any]) -> str:
    if spec["kind"] == "union":
        return "\nUNION ALL\n".join(render_sql(b) for b in spec["branches"])
    sql = "SELECT\n  " + ",\n  ".join(spec["select"])
    sql += f"\nFROM {spec['root']} t"
    for j in spec["joins"]:
        sql += "\n" + j
    if spec["where"]:
        sql += "\nWHERE " + "\n  AND ".join(spec["where"])
    if spec["group"]:
        sql += "\nGROUP BY " + ", ".join(spec["group"])
    if spec.get("having"):
        sql += "\nHAVING " + "\n  AND ".join(spec["having"])
    if spec["order"]:
        sql += "\nORDER BY " + ", ".join(spec["order"])
    if spec["limit"]:
        sql += f"\nLIMIT {spec['limit']}"
    return sql
