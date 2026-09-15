"""Agent 7 — Query Repair.

Applies the cheapest correct fix for each error in the ValidationReport:

* schema errors      → fuzzy-match to the closest real identifier (≥ threshold similarity),
                       otherwise escalate to regeneration
* structural errors  → rebuild the join condition from the foreign-key graph
* semantic errors    → regenerate with the verifier's failure reasons appended

Repaired SQL always goes back through all three checks.
"""
import difflib
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot import exp

from app.agents.anti_hallucination import Resolver
from app.config import FUZZY_MATCH_THRESHOLD
from app.db.catalog import Catalog


def _closest(name: str, candidates: List[str]) -> Optional[Dict[str, Any]]:
    best = None
    for c in candidates:
        ratio = difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio()
        if best is None or ratio > best["similarity"]:
            best = {"match": c, "similarity": round(ratio, 3)}
    return best


def _fix_schema(tree: exp.Expression, errors: List[Dict[str, Any]], catalog: Catalog, changes: List[str]) -> List[str]:
    escalate: List[str] = []
    for err in errors:
        if err["code"] == "unknown_table":
            best = _closest(err["table"], list(catalog.tables))
            if best and best["similarity"] >= FUZZY_MATCH_THRESHOLD:
                for t in tree.find_all(exp.Table):
                    if t.name.lower() == err["table"]:
                        t.set("this", exp.to_identifier(best["match"]))
                changes.append(f"table `{err['table']}` → `{best['match']}` (similarity {best['similarity']:.2f})")
            else:
                escalate.append(err["message"])
        elif err["code"] == "unknown_column":
            resolver = Resolver(tree, catalog)
            if err.get("table"):
                candidates = catalog.column_names(err["table"])
            else:
                candidates = sorted({c for t in resolver.real_tables for c in catalog.column_names(t)})
            best = _closest(err["column"], candidates)
            if best and best["similarity"] >= FUZZY_MATCH_THRESHOLD:
                for col in tree.find_all(exp.Column):
                    if col.name.lower() == err["column"] and (col.table or None) == err.get("alias"):
                        col.set("this", exp.to_identifier(best["match"]))
                prefix = f"{err['alias']}." if err.get("alias") else ""
                changes.append(f"column `{prefix}{err['column']}` → `{prefix}{best['match']}` (similarity {best['similarity']:.2f})")
            else:
                sim = f"best match `{best['match']}` only {best['similarity']:.2f}" if best else "no candidates"
                changes.append(f"column `{err['column']}`: no confident match ({sim}) — escalating to regeneration")
                escalate.append(err["message"])
        elif err["code"] == "parse_error":
            escalate.append(err["message"])
    return escalate


def _fix_structure(tree: exp.Expression, errors: List[Dict[str, Any]], catalog: Catalog, changes: List[str]) -> List[str]:
    escalate: List[str] = []
    bad_aliases = {e.get("alias", "").lower() for e in errors if e.get("alias")}
    for select in tree.find_all(exp.Select):
        from_ = select.args.get("from") or select.args.get("from_")
        if not from_ or not isinstance(from_.this, exp.Table):
            continue
        joined = {from_.this.alias_or_name.lower(): from_.this.name.lower()}
        for join in select.args.get("joins") or []:
            target = join.this
            if not isinstance(target, exp.Table):
                continue
            alias, table = target.alias_or_name.lower(), target.name.lower()
            if alias in bad_aliases:
                fixed = False
                for prev_alias, prev_table in joined.items():
                    for fk in catalog.foreign_keys:
                        if {fk.from_table, fk.to_table} == {table, prev_table}:
                            a_from = alias if fk.from_table == table else prev_alias
                            a_to = prev_alias if fk.from_table == table else alias
                            if fk.from_table == fk.to_table:
                                continue
                            before = join.args.get("on").sql() if join.args.get("on") else "(none)"
                            cond = exp.EQ(this=exp.column(fk.from_column, table=a_from), expression=exp.column(fk.to_column, table=a_to))
                            join.set("on", cond)
                            changes.append(f"join on `{table}`: `{before}` → `{cond.sql()}` (FK {fk.from_table}.{fk.from_column} → {fk.to_table}.{fk.to_column})")
                            fixed = True
                            break
                    if fixed:
                        break
                if not fixed:
                    escalate.append(f"no foreign key connects `{table}` to the tables before it")
            joined[alias] = table
    return escalate


def repair(sql: str, report: Dict[str, Any], catalog: Catalog, regenerate, recheck) -> Dict[str, Any]:
    checks = report["checks"]
    changes: List[str] = []
    strategies: List[str] = []

    blocking = [e for c in checks.values() for e in c["errors"] if not e.get("repairable", True)]
    if blocking:
        return {"sql": sql, "blocked": True, "strategies": ["none — not repairable"], "changes": [e["message"] for e in blocking]}

    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        tree = None

    escalate: List[str] = []
    if tree is not None and checks["schema"]["errors"]:
        strategies.append("fuzzy schema match")
        escalate += _fix_schema(tree, checks["schema"]["errors"], catalog, changes)
    elif tree is None:
        escalate.append(checks["schema"]["errors"][0]["message"])
    if tree is not None and checks["structural"]["errors"]:
        strategies.append("FK-graph join rewrite")
        escalate += _fix_structure(tree, checks["structural"]["errors"], catalog, changes)

    semantic_errors = [e["message"] for e in checks["semantic"]["errors"]]
    if tree is not None and not escalate and changes and semantic_errors:
        # Semantic failures can be side effects of a bad identifier; re-check the patched SQL first.
        patched = tree.sql(dialect="sqlite", pretty=True)
        semantic_errors = [e["message"] for e in recheck(patched)["checks"]["semantic"]["errors"]]
        if not semantic_errors:
            changes.append("semantic check passes on the patched SQL — no regeneration needed")
    if escalate or semantic_errors:
        strategies.append("regenerate with verifier feedback")
        feedback = escalate + semantic_errors
        regenerated = regenerate(feedback)
        changes.append("regenerated SQL with feedback: " + "; ".join(feedback))
        return {"sql": regenerated["sql"], "blocked": False, "strategies": strategies, "changes": changes, "model": regenerated["model"]}

    return {"sql": tree.sql(dialect="sqlite", pretty=True), "blocked": False, "strategies": strategies, "changes": changes}
