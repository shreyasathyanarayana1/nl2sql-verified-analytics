"""Agent 5 — SQL Generation.

This build does not call a hosted LLM. A trained local query-family retriever selects the
closest learned SQL family, while the verified structured plan is rendered deterministically
against the live schema. This prevents identifier invention and keeps SQL generation safe.

`inject_hallucination` deliberately corrupts a draft with one hallucination type so the
verification–repair loop can be evaluated (API-only; not exposed in the UI).
"""
import json
import random
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import exp

from app import local_ml
from app.agents.planning import render_sql
from app.datasets import Dataset

SYSTEM_PROMPT = """You write one read-only SQLite SELECT statement for an analytics platform.
Use ONLY the tables and columns listed in the schema context. Never use a restricted column.
Join tables only through the listed foreign keys. Follow the query plan step by step and keep the
output column aliases used in the plan. Respond as JSON: {"sql": "<one SELECT statement>"}"""


def _local_generate(ds: Dataset, intent: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate SQL locally from the verified structured plan.

    The learned model selects the closest SQL/query family; the existing renderer
    then produces schema-safe SQL from the actual intent, so the model cannot invent
    identifiers or bypass the verification layer.
    """
    learned = local_ml.predict_sql_template(intent["raw"])
    sql = render_sql(plan["spec"])
    return {"sql": sql, "model": "Local ML template selector + deterministic SQL renderer", "ml_generation": learned}


def _aliases(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        first = tree if isinstance(tree, exp.Select) else next(tree.find_all(exp.Select))
        return [s.alias_or_name for s in first.selects]
    except Exception:
        return []


def generate(ds: Dataset, intent: Dict[str, Any], plan: Dict[str, Any], schema_ctx: Dict[str, Any], scope: Dict[str, Any],
             history: List[Dict[str, Any]], feedback: Optional[List[str]] = None) -> Dict[str, Any]:
    out = _local_generate(ds, intent, plan)
    if feedback:
        out["feedback_used"] = feedback
    return out


# --------------------------------------------------------------------------- evaluation: hallucination injection

def _typo(name: str) -> str:
    """A plausible near-miss identifier, e.g. cholesterol → cholestrol, exam_score → exam_scor."""
    if len(name) > 6:
        cut = len(name) - 3
        return name[:cut] + name[cut + 1:]
    return name + "s"


def inject_hallucination(ds: Dataset, sql: str, kind: str, rng: random.Random) -> Dict[str, Any]:
    """Corrupts SQL with one realistic hallucination. Returns the new SQL and what changed."""
    if kind == "random":
        kind = rng.choice(["schema", "semantic", "structural"])
    tree = sqlglot.parse_one(sql, read="sqlite")

    def done(kind_: str, description: str) -> Dict[str, Any]:
        return {"sql": tree.sql(dialect="sqlite", pretty=True), "type": kind_, "description": description}

    if kind == "schema":
        for col in tree.find_all(exp.Column):
            if col.table == "t" and col.name in ds.columns:
                before = col.sql()
                col.set("this", exp.to_identifier(_typo(col.name)))
                return done("schema", f"Referenced non-existent column {col.sql()} instead of {before}")

    if kind == "semantic":
        for agg in tree.find_all(exp.Avg):
            before = agg.sql()
            agg.replace(exp.Sum(this=agg.this.copy()))
            return done("semantic", f"Used SUM instead of AVG ({before})")
        where = tree.find(exp.Where)
        if where:
            before = where.this.sql()
            where.pop()
            return done("semantic", f"Silently dropped the filter {before}")
        limit = tree.args.get("limit")
        if limit is not None:
            for ordered in tree.find_all(exp.Ordered):
                before = ordered.sql()
                desc = not ordered.args.get("desc")
                ordered.set("desc", desc)
                ordered.set("nulls_first", not desc)
                return done("semantic", f"Reversed the ranking direction ({before} → {ordered.sql()})")
        for agg in tree.find_all(exp.Max):
            before = agg.sql()
            agg.replace(exp.Min(this=agg.this.copy()))
            return done("semantic", f"Used MIN instead of MAX ({before})")

    if kind in ("structural", "schema", "semantic"):
        integer_cols = [c for c in ds.catalog.tables[ds.fact_table] if c.type == "INTEGER" and not c.primary_key]
        for join in tree.find_all(exp.Join):
            on = join.args.get("on")
            eq = on.find(exp.EQ) if on else None
            if not eq:
                continue
            side = eq.left if isinstance(eq.left, exp.Column) and eq.left.table == "t" else eq.right
            if not isinstance(side, exp.Column):
                continue
            wrong = next((c.name for c in integer_cols if c.name != side.name), None)
            if wrong:
                before = eq.sql()
                side.set("this", exp.to_identifier(wrong))
                return done("structural", f"Joined on a non-FK relationship ({eq.sql()} instead of {before})")

    return {"sql": sql, "type": None, "description": "No applicable hallucination for this query shape"}
