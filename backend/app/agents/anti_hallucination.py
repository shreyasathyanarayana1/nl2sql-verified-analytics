"""Agent 6 — Anti-Hallucination verification.

Three independent checks run on every draft before it may reach the database:

1. Schema validation   — every table/column exists in the *live* catalog and is
                          permitted for the caller's role (deterministic).
2. Semantic verification — the SQL answers the stated intent, asked as specific
                          yes/no questions (deterministic intent-derived checks, plus
                          an independent LLM pass when a key is configured).
3. Structural validation — every join condition is a real foreign-key edge
                          (deterministic graph check).
"""
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from app import llm
from app.agents import access_control
from app.db.catalog import Catalog
from app.semantic import Semantic

OP_NODES = {"=": exp.EQ, ">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE, "in": exp.In}


class Resolver:
    """Maps SQL column references to (table, column) using the query's own aliases."""

    def __init__(self, tree: exp.Expression, catalog: Catalog):
        self.catalog = catalog
        self.alias_map: Dict[str, str] = {}
        self.derived: Dict[str, Set[str]] = {}
        self.derived_select: Dict[str, exp.Select] = {}
        for t in tree.find_all(exp.Table):
            self.alias_map[t.alias_or_name.lower()] = t.name.lower()
        for sq in tree.find_all(exp.Subquery):
            if sq.alias and isinstance(sq.this, exp.Select):
                self.derived[sq.alias.lower()] = {s.alias_or_name.lower() for s in sq.this.selects}
                self.derived_select[sq.alias.lower()] = sq.this
        self.select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)}
        self.real_tables = {v for v in self.alias_map.values() if catalog.has_table(v)}

    def lineage(self, col: exp.Column) -> Tuple[Optional[str], Optional[str]]:
        """Like resolve, but follows derived-table columns back to the base column they select."""
        qualifier = col.table.lower() if col.table else ""
        if qualifier in self.derived_select:
            for sel in self.derived_select[qualifier].selects:
                if sel.alias_or_name.lower() == col.name.lower():
                    inner = sel.this if isinstance(sel, exp.Alias) else sel
                    return self.resolve(inner)[:2] if isinstance(inner, exp.Column) else (None, None)
            return None, None
        return self.resolve(col)[:2]

    def resolve(self, col: exp.Column) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Returns (table, column, error). table is '' for select/derived aliases."""
        name = col.name.lower()
        qualifier = col.table.lower() if col.table else ""
        if qualifier:
            if qualifier in self.derived:
                return ("", name, None) if name in self.derived[qualifier] else (None, name, f"unknown column `{col.sql()}`")
            table = self.alias_map.get(qualifier)
            if table is None:
                return None, name, f"unknown table alias `{qualifier}`"
            if not self.catalog.has_table(table):
                return None, name, None  # reported as unknown table
            if name not in self.catalog.column_names(table):
                return table, name, f"unknown column `{col.sql()}`"
            return table, name, None
        if name in self.select_aliases or any(name in outs for outs in self.derived.values()):
            return "", name, None
        owners = [t for t in self.real_tables if name in self.catalog.column_names(t)]
        if owners:
            return owners[0], name, None
        return None, name, f"unknown column `{col.sql()}`"


def _literal(node: exp.Expression) -> Any:
    if isinstance(node, exp.Literal):
        return node.this if node.is_string else float(node.this)
    if isinstance(node, exp.Paren):
        return _literal(node.this)
    return None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(a).lower() == str(b).lower()


# --------------------------------------------------------------------------- check 1

def check_schema(tree: exp.Expression, catalog: Catalog, scope: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    resolver = Resolver(tree, catalog)
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}

    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name not in cte_names and not catalog.has_table(name):
            errors.append({"code": "unknown_table", "table": name, "alias": t.alias_or_name,
                           "message": f"unknown table `{name}`", "repairable": True})

    seen = set()
    referenced: List[str] = []
    for col in tree.find_all(exp.Column):
        table, name, error = resolver.resolve(col)
        key = (col.table, col.name)
        if error and key not in seen:
            errors.append({"code": "unknown_column", "table": table, "column": name, "alias": col.table or None,
                           "message": error, "repairable": True})
        seen.add(key)
        if table and not error:
            referenced.append(f"{table}.{name}")
            if access_control.is_denied(scope, table, name):
                msg = f"access to `{table}.{name}` is not permitted for role {scope['role']}"
                if not any(e["message"] == msg for e in errors):
                    errors.append({"code": "access_denied", "table": table, "column": name, "message": msg, "repairable": False})

    for select in tree.find_all(exp.Select):
        if any(isinstance(s, exp.Star) or (isinstance(s, exp.Column) and isinstance(s.this, exp.Star)) for s in select.selects):
            for table in resolver.real_tables:
                if any(d["table"] == table for d in scope["denied"]):
                    errors.append({"code": "access_denied", "table": table, "message": f"SELECT * would expose restricted columns of `{table}`", "repairable": False})

    return {"passed": not errors, "errors": errors, "referenced": sorted(set(referenced))}


# --------------------------------------------------------------------------- check 3

def check_structure(tree: exp.Expression, catalog: Catalog) -> Dict[str, Any]:
    resolver = Resolver(tree, catalog)
    errors: List[Dict[str, Any]] = []
    joins: List[Dict[str, Any]] = []

    def validate_pair(eq: exp.EQ, join_alias: Optional[str]) -> bool:
        lt, lc = resolver.lineage(eq.left)
        rt, rc = resolver.lineage(eq.right)
        if not lt or not rt or lt == rt:
            return False
        fk = catalog.fk_between(lt, lc, rt, rc)
        joins.append({"condition": eq.sql(), "valid": bool(fk),
                      "foreign_key": f"{fk.from_table}.{fk.from_column} → {fk.to_table}.{fk.to_column}" if fk else None})
        if not fk:
            errors.append({"code": "invalid_join", "alias": join_alias, "condition": eq.sql(),
                           "message": f"`{eq.sql()}` joins {lt}.{lc} to {rt}.{rc}, which is not a foreign-key relationship",
                           "repairable": True})
        return True

    for join in tree.find_all(exp.Join):
        target = join.this
        if isinstance(target, exp.Subquery):
            on = join.args.get("on")
            conds = [e for e in (on.find_all(exp.EQ) if on else []) if isinstance(e.left, exp.Column) and isinstance(e.right, exp.Column)]
            if not conds:
                errors.append({"code": "cross_join", "alias": target.alias, "message": f"derived table `{target.alias}` is joined without a key condition", "repairable": True})
            for eq in conds:
                if not validate_pair(eq, target.alias):
                    errors.append({"code": "invalid_join", "alias": target.alias,
                                   "message": f"`{eq.sql()}` does not trace back to a foreign-key relationship", "repairable": True})
            continue
        if not isinstance(target, exp.Table):
            continue
        alias = target.alias_or_name
        on = join.args.get("on")
        conditions = [e for e in (on.find_all(exp.EQ) if on else []) if isinstance(e.left, exp.Column) and isinstance(e.right, exp.Column)]
        if not conditions:
            select = join.find_ancestor(exp.Select)
            where = select.args.get("where") if select else None
            conditions = [e for e in (where.find_all(exp.EQ) if where else [])
                          if isinstance(e.left, exp.Column) and isinstance(e.right, exp.Column)
                          and alias.lower() in (e.left.table.lower(), e.right.table.lower())]
        if not conditions:
            errors.append({"code": "cross_join", "alias": alias, "message": f"`{target.name}` is joined without any key condition (cross join)", "repairable": True})
            continue
        touched = False
        for eq in conditions:
            if alias.lower() in (eq.left.table.lower(), eq.right.table.lower()):
                touched = validate_pair(eq, alias) or touched
        if not touched:
            errors.append({"code": "invalid_join", "alias": alias, "message": f"join condition for `{target.name}` does not reference it", "repairable": True})

    return {"passed": not errors, "errors": errors, "joins": joins}


# --------------------------------------------------------------------------- check 2

def _find_filter(tree: exp.Expression, resolver: Resolver, table: str, column: str, f: Dict[str, Any]) -> bool:
    node_type = OP_NODES[f["op"]]
    for where in tree.find_all(exp.Where):
        for node in where.find_all(node_type):
            if node_type is exp.In:
                col = node.this
                if isinstance(col, exp.Column) and resolver.resolve(col)[:2] == (table, column):
                    values = sorted(str(_literal(v)).lower().rstrip("0").rstrip(".") for v in node.expressions)
                    wanted = sorted(str(float(v) if isinstance(v, (int, float)) else v).lower().rstrip("0").rstrip(".") for v in f["value"])
                    if values == wanted:
                        return True
                continue
            left, right = node.left, node.right
            if isinstance(left, exp.Column) and resolver.resolve(left)[:2] == (table, column) and _same(_literal(right), f["value"]):
                return True
    return False


def _grouped_columns(tree: exp.Expression, resolver: Resolver) -> Set[Tuple[Optional[str], Optional[str]]]:
    """Columns that drive grouping, following GROUP BY aliases back to their select expressions."""
    out: Set[Tuple[Optional[str], Optional[str]]] = set()
    for select in tree.find_all(exp.Select):
        group = select.args.get("group")
        if not group:
            continue
        aliases = {s.alias.lower(): s.this for s in select.selects if isinstance(s, exp.Alias)}
        for g in group.expressions:
            if isinstance(g, exp.Column) and not g.table and g.name.lower() in aliases:
                g = aliases[g.name.lower()]
            for c in ([g] if isinstance(g, exp.Column) else list(g.find_all(exp.Column))):
                out.add(resolver.resolve(c)[:2])
    return out


def check_semantics(tree: exp.Expression, intent: Dict[str, Any], sem: Semantic, scope: Dict[str, Any], sql: str,
                    use_llm: bool) -> Dict[str, Any]:
    catalog = sem.ds.catalog
    resolver = Resolver(tree, catalog)
    expect = sem.expectations(intent)
    questions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    fact = sem.ds.fact_table

    def ask(question: str, ok: bool, detail: str) -> None:
        questions.append({"question": question, "answer": ok, "detail": detail, "source": "intent-derived check"})

    present = {type(a).__name__ for a in tree.find_all(exp.AggFunc)}
    expected = expect["agg_funcs"]
    names = lambda s: ", ".join(sorted(x.upper() for x in s)) or "none"
    ask("Does the SQL use the aggregation the question asks for?", present == expected,
        f"expected {names(expected)}; SQL uses {names(present)}")

    resolved = {resolver.resolve(c)[:2] for c in tree.find_all(exp.Column)}
    for fn, (table, column) in expect["wrapped"]:
        label = sem.ds.columns[column].label if column in sem.ds.columns else column
        if fn is None:
            ok = (table, column) in resolved
            ask(f"Does the SQL return {label}?", ok, "found" if ok else f"{table}.{column} is never referenced")
            continue
        wrappers = {type(a).__name__ for a in tree.find_all(exp.AggFunc)
                    if any(resolver.resolve(c)[:2] == (table, column) for c in a.find_all(exp.Column))}
        found = ", ".join(sorted(w.upper() for w in wrappers)) or ("no aggregate" if (table, column) in resolved else "never referenced")
        ask(f"Is {label} aggregated with {fn.upper()}?", fn in wrappers, f"found {found}")

    grouped = _grouped_columns(tree, resolver)
    for table, column in expect["group_columns"]:
        label = next((c.label for c in sem.ds.columns.values() if (c.lookup, "name") == (table, column) or (fact, c.name) == (table, column)), column)
        ask(f"Does the SQL group by {label}?", (table, column) in grouped,
            "found in GROUP BY" if (table, column) in grouped else "missing from GROUP BY")
    if intent["kind"] == "drivers":
        restricted = {c for c in sem.ds.columns if access_control.is_denied(scope, fact, c)}
        missing = [c.label for c in sem.driver_factors(intent, restricted)
                   if (sem.display_column(c) if c.lookup else (fact, c.name)) not in grouped]
        ask("Does the SQL break the outcome down by every eligible factor?", not missing,
            "all factors grouped" if not missing else f"missing: {', '.join(missing)}")

    for (table, column), f in expect["filters"]:
        found = _find_filter(tree, resolver, table, column, f)
        ask(f"Does the SQL filter on {f['label']}?", found,
            "condition present" if found else f"no WHERE condition {table}.{column} {f['op']} {f['value']}")

    for col, m in expect["rates"]:
        found = False
        for case in tree.find_all(exp.Case):
            for eq in case.find_all(exp.EQ):
                if isinstance(eq.left, exp.Column) and resolver.resolve(eq.left)[:2] == (fact, col.name) and _same(_literal(eq.right), col.positive):
                    found = True
        ask(f"Does the SQL count {col.label} = {col.positive} as the positive outcome?", found,
            "condition present" if found else f"no CASE WHEN {col.name} = {col.positive}")

    ranking = expect["ranking"]
    if ranking and isinstance(tree, exp.Select):
        order = tree.args.get("order")
        first = order.expressions[0] if order and order.expressions else None
        direction_ok = bool(first) and bool(first.args.get("desc")) == (ranking["direction"] == "DESC")
        limit_node = tree.args.get("limit")
        limit_val = _literal(limit_node.expression) if limit_node is not None else None
        ask(f"Does the SQL sort {'highest' if ranking['direction'] == 'DESC' else 'lowest'} first?", direction_ok,
            f"ORDER BY {first.sql() if first else 'missing'}")
        ask(f"Does the SQL return the top {ranking['limit']}?", _same(limit_val, float(ranking["limit"])),
            f"LIMIT {int(limit_val) if limit_val is not None else 'missing'}")

    if expect["share"]:
        ok = any(isinstance(w.this, exp.Sum) for w in tree.find_all(exp.Window))
        ask("Does the SQL divide each group by the overall total to get its share?", ok,
            "SUM(...) OVER () present" if ok else "no windowed total found")

    if use_llm and llm.enabled():
        try:
            reply = llm.chat_json(
                "You audit SQL against a user's question. Answer each yes/no question strictly. "
                'Respond as JSON: {"answers": [{"question": str, "answer": true|false, "reason": str}]}',
                f"Question: {intent['raw']}\n\nSQL:\n{sql}\n\nAnswer these:\n"
                + "\n".join(f"- {q['question']}" for q in questions)
                + "\n- Does this SQL, taken as a whole, correctly answer the user's question?",
            )
            for a in reply.get("answers", []):
                questions.append({"question": a["question"], "answer": bool(a["answer"]), "detail": a.get("reason", ""),
                                  "source": "independent LLM pass"})
        except Exception as e:
            warnings.append(f"LLM semantic pass unavailable ({type(e).__name__}); deterministic checks only.")

    failed = [q for q in questions if not q["answer"]]
    errors = [{"code": "semantic_mismatch", "message": f"{q['question']} — No: {q['detail']}", "repairable": True} for q in failed]
    return {"passed": not failed, "questions": questions, "errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------- agent entry

def verify(sql: str, intent: Dict[str, Any], sem: Semantic, scope: Dict[str, Any]) -> Dict[str, Any]:
    catalog = sem.ds.catalog
    timings = {}
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        err = {"code": "parse_error", "message": f"SQL does not parse: {str(e).splitlines()[0]}", "repairable": True}
        return {"passed": False, "checks": {
            "schema": {"passed": False, "errors": [err], "referenced": []},
            "semantic": {"passed": False, "questions": [], "errors": [], "warnings": [], "skipped": True},
            "structural": {"passed": False, "errors": [], "joins": [], "skipped": True},
        }, "hallucination_types": ["schema"], "timings_ms": {}}

    if not isinstance(tree, (exp.Select, exp.Union)):
        err = {"code": "not_read_only", "message": "only SELECT statements may be executed", "repairable": False}
        return {"passed": False, "checks": {
            "schema": {"passed": False, "errors": [err], "referenced": []},
            "semantic": {"passed": False, "questions": [], "errors": [], "warnings": [], "skipped": True},
            "structural": {"passed": False, "errors": [], "joins": [], "skipped": True},
        }, "hallucination_types": ["schema"], "timings_ms": {}}

    t0 = time.perf_counter()
    schema = check_schema(tree, catalog, scope)
    timings["schema"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    semantic = check_semantics(tree, intent, sem, scope, sql, use_llm=schema["passed"])
    timings["semantic"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    structural = check_structure(tree, catalog)
    timings["structural"] = round((time.perf_counter() - t0) * 1000, 2)

    checks = {"schema": schema, "semantic": semantic, "structural": structural}
    return {
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "hallucination_types": [name for name, c in checks.items() if not c["passed"]],
        "timings_ms": timings,
    }
