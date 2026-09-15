"""Evidence drill-down — the rows behind any number in an answer.

Evidence SQL is derived from the *verified* query that produced the answer (same tables,
joins and filters) with aggregation removed and the clicked group pinned, so nothing is
re-guessed. It passes the same schema, access and join checks before it runs.
"""
from typing import Any, Dict, Optional

import sqlglot
from sqlglot import exp

from app.agents import anti_hallucination, execution
from app.datasets import Dataset

EVIDENCE_LIMIT = 500


class EvidenceError(Exception):
    pass


def _literal(value: Any) -> exp.Expression:
    if isinstance(value, bool):
        return exp.Literal.number(int(value))
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    text = str(value)
    try:
        float(text)
        return exp.Literal.number(text)
    except ValueError:
        return exp.Literal.string(text)


def build_sql(ds: Dataset, final_sql: str, key_column: Optional[str], key_value: Any) -> str:
    tree = sqlglot.parse_one(final_sql, read="sqlite")
    if not isinstance(tree, exp.Select) or not tree.find(exp.AggFunc):
        raise EvidenceError("Every row of this answer is already a single record.")
    select = tree.copy()
    aliases = {s.alias_or_name: (s.this if isinstance(s, exp.Alias) else s) for s in tree.selects}

    if key_column is not None:
        if key_column not in aliases:
            raise EvidenceError(f"`{key_column}` is not a column of this answer")
        expr = aliases[key_column].copy()
        select = select.where(exp.Is(this=expr, expression=exp.Null()) if key_value is None
                              else exp.EQ(this=expr, expression=_literal(key_value)))

    columns, seen = [exp.column(ds.id_column, table="t")], {f"t.{ds.id_column}"}

    def add(node: exp.Expression, alias: Optional[str] = None) -> None:
        key = node.sql()
        if key not in seen:
            seen.add(key)
            columns.append(exp.alias_(node, alias) if alias else node)

    for name, node in aliases.items():
        if not node.find(exp.AggFunc) and not node.find(exp.Window):
            add(node.copy(), name)
            for col in node.find_all(exp.Column):  # the raw value behind a band or label
                add(col.copy())
    for agg in tree.find_all(exp.AggFunc):
        for col in agg.find_all(exp.Column):
            add(col.copy())
    for cond in (tree.args.get("where").find_all(exp.Column) if tree.args.get("where") else []):
        add(cond.copy())

    select.set("expressions", columns)
    for arg in ("group", "having", "order", "limit"):
        select.set(arg, None)
    return select.limit(EVIDENCE_LIMIT).sql(dialect="sqlite", pretty=True)


def fetch(ds: Dataset, sem, scope: Dict[str, Any], turn: Dict[str, Any], key_column: Optional[str], key_value: Any) -> Dict[str, Any]:
    if turn["kind"] in ("correlation", "list", "drivers") or (turn["kind"] == "ranking" and turn.get("rows_ranking")):
        raise EvidenceError("Each point in this answer is already a single record.")
    sql = build_sql(ds, turn["sql"], key_column, key_value)
    tree = sqlglot.parse_one(sql, read="sqlite")
    schema = anti_hallucination.check_schema(tree, ds.catalog, scope)
    structure = anti_hallucination.check_structure(tree, ds.catalog)
    if not schema["passed"] or not structure["passed"]:
        raise EvidenceError("The evidence query failed verification: " + "; ".join(e["message"] for e in schema["errors"] + structure["errors"]))
    return {"sql": sql, **execution.execute(ds.db_path, sql), "column_labels": sem.column_labels()}
