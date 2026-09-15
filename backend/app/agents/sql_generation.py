"""Agent 5 — SQL Generation.

With a working OpenAI key the SQL is written by GPT-4o from the question, the retrieved
schema chunks (never the full schema), the access scope and the plan. Offline — or if the
LLM call fails — the structured plan is rendered deterministically. A locally trained
text-to-SQL model can take the LLM's place behind the same `generate` function.

`inject_hallucination` deliberately corrupts a draft with one hallucination type so the
verification–repair loop can be evaluated (API-only; not exposed in the UI).
"""
import json
import random
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import exp

from app import llm
from app.agents.planning import render_sql
from app.datasets import Dataset

SYSTEM_PROMPT = """You write one read-only SQLite SELECT statement for an analytics platform.
Use ONLY the tables and columns listed in the schema context. Never use a restricted column.
Join tables only through the listed foreign keys. Follow the query plan step by step and keep the
output column aliases used in the plan. Respond as JSON: {"sql": "<one SELECT statement>"}"""


def _llm_generate(ds: Dataset, intent: Dict[str, Any], plan: Dict[str, Any], schema_ctx: Dict[str, Any],
                  scope: Dict[str, Any], history: List[Dict[str, Any]], feedback: Optional[List[str]]) -> str:
    context = "\n\n".join(c["content"] for c in schema_ctx["chunks"])
    restricted = ", ".join(f"{d['table']}.{d['column']}" for d in scope["denied"]) or "none"
    reference_shape = render_sql(plan["spec"])
    user = (
        f"Dataset: {ds.name} (one row per {ds.singular}).\nQuestion: {intent['raw']}\n\n"
        f"Intent: {json.dumps({k: intent[k] for k in ('kind', 'measures', 'group_by', 'filters', 'ranking', 'share')})}\n\n"
        f"Schema context (retrieved):\n{context}\n\nRestricted columns: {restricted}\n\n"
        "Query plan:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan["steps"])) +
        f"\n\nOutput columns expected (aliases): {', '.join(_aliases(reference_shape))}"
    )
    if history:
        user += "\n\nRecent turns:\n" + "\n".join(f"- {h['question']} -> {h['sql']}" for h in history)
    if feedback:
        user += "\n\nA verifier rejected the previous SQL for these reasons; fix all of them:\n" + "\n".join(f"- {f}" for f in feedback)
    return llm.chat_json(SYSTEM_PROMPT, user)["sql"].strip().rstrip(";")


def _aliases(sql: str) -> List[str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        first = tree if isinstance(tree, exp.Select) else next(tree.find_all(exp.Select))
        return [s.alias_or_name for s in first.selects]
    except Exception:
        return []


def generate(ds: Dataset, intent: Dict[str, Any], plan: Dict[str, Any], schema_ctx: Dict[str, Any], scope: Dict[str, Any],
             history: List[Dict[str, Any]], feedback: Optional[List[str]] = None) -> Dict[str, Any]:
    source = "rule-based renderer"
    sql = None
    if llm.enabled():
        try:
            sql = _llm_generate(ds, intent, plan, schema_ctx, scope, history, feedback)
            source = f"LLM ({llm.OPENAI_MODEL})"
        except Exception as e:  # network/API failure: degrade gracefully
            source = f"rule-based renderer (LLM unavailable: {type(e).__name__})"
    if sql is None:
        sql = render_sql(plan["spec"])
    return {"sql": sql, "model": source}


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
