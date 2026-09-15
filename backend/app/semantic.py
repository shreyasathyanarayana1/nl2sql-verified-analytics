"""Semantic layer: turns dataset metadata into SQL building blocks and verifier expectations.

The planner (to build SQL) and the Anti-Hallucination Agent (to derive what a correct query
*must* contain) both read from here. Expectations come from the intent alone, never from the
generated SQL, so the verifier stays independent of the generator.
"""
from typing import Any, Dict, List, Optional, Set, Tuple

from app.datasets import ColumnMeta, Dataset

AGG_FUNC = {"AVG": "Avg", "SUM": "Sum", "MIN": "Min", "MAX": "Max", "COUNT": "Count"}
AGG_WORD = {"AVG": "average", "SUM": "total", "MIN": "lowest", "MAX": "highest", "COUNT": "number of", "RATE": ""}


class Semantic:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.t = ds.alias(ds.fact_table)

    # ------------------------------------------------------------------ columns

    def col(self, name: str) -> ColumnMeta:
        return self.ds.columns[name]

    def literal(self, col: ColumnMeta, value: Any) -> str:
        if isinstance(value, list):
            return "(" + ", ".join(self.literal(col, v) for v in value) + ")"
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            return str(int(value)) if float(value).is_integer() else str(value)
        return "'" + str(value).replace("'", "''") + "'"

    def lookup_join(self, col: ColumnMeta) -> Optional[str]:
        if not col.lookup:
            return None
        alias = self.ds.alias(col.lookup)
        return f"JOIN {col.lookup} {alias} ON {self.t}.{col.name} = {alias}.code"

    def display_expr(self, col: ColumnMeta) -> str:
        """Human-readable value of a column: the lookup name for coded columns."""
        return f"{self.ds.alias(col.lookup)}.name" if col.lookup else f"{self.t}.{col.name}"

    def display_column(self, col: ColumnMeta) -> Tuple[str, str]:
        """(table, column) the display expression reads — used by the verifier."""
        return (col.lookup, "name") if col.lookup else (self.ds.fact_table, col.name)

    def band_expr(self, col: ColumnMeta, width: float) -> str:
        w = int(width) if float(width).is_integer() else width
        return f"CAST({self.t}.{col.name} / {w} AS INTEGER) * {w}"

    def flag_label_expr(self, col: ColumnMeta) -> str:
        return (f"CASE WHEN {self.t}.{col.name} = {self.literal(col, col.positive)} THEN 'Yes' "
                f"WHEN {self.t}.{col.name} = {self.literal(col, col.negative)} THEN 'No' END")

    def order_expr(self, col: ColumnMeta, expr: str) -> Optional[str]:
        if not col.order:
            return None
        quote = lambda v: "'" + str(v).replace("'", "''") + "'"
        cases = " ".join(f"WHEN {quote(v)} THEN {i}" for i, v in enumerate(col.order))
        return f"CASE {expr} {cases} ELSE {len(col.order)} END"

    # ------------------------------------------------------------------ measures

    def count_alias(self) -> str:
        return f"n_{self.ds.plural}"

    def measure_expr(self, measure: Dict[str, Any]) -> Tuple[str, str]:
        agg, name = measure["agg"], measure.get("col")
        if agg == "COUNT":
            return "COUNT(*)", self.count_alias()
        col = self.col(name)
        ref = f"{self.t}.{name}"
        if agg == "RATE":
            return (f"ROUND(100.0 * SUM(CASE WHEN {ref} = {self.literal(col, col.positive)} THEN 1 ELSE 0 END) / COUNT({ref}), 1)",
                    f"{name}_rate_pct")
        if agg == "AVG":
            return f"ROUND(AVG({ref}), 2)", f"avg_{name}"
        if agg == "SUM":
            return f"SUM({ref})", f"total_{name}"
        if agg == "MAX":
            return f"MAX({ref})", f"highest_{name}"
        if agg == "MIN":
            return f"MIN({ref})", f"lowest_{name}"
        if agg is None:
            return ref, name
        raise ValueError(f"unknown aggregation {agg}")

    def measure_label(self, measure: Dict[str, Any], short: bool = False) -> str:
        agg, name = measure["agg"], measure.get("col")
        if agg == "COUNT":
            return f"number of {self.ds.plural}"
        col = self.col(name)
        if agg == "RATE":
            return f"{col.label} rate"
        if agg is None:
            return col.label
        return f"{AGG_WORD[agg]} {col.label}"

    # ------------------------------------------------------------------ filters

    def filter_sql(self, f: Dict[str, Any]) -> str:
        col = self.col(f["col"])
        if f["op"] == "is_null":
            return f"{self.t}.{col.name} IS NULL"
        expr = self.display_expr(col) if col.lookup else f"{self.t}.{col.name}"
        op = "IN" if f["op"] == "in" else f["op"]
        return f"{expr} {op} {self.literal(col, f['value'])}"

    def filter_label(self, f: Dict[str, Any]) -> str:
        return f["label"]

    # ------------------------------------------------------------------ tables & labels

    def required_tables(self, intent: Dict[str, Any]) -> Set[str]:
        tables = {self.ds.fact_table}
        names = [m["col"] for m in intent.get("measures", []) if m.get("col")]
        names += [g["col"] for g in intent.get("group_by", [])]
        names += [f["col"] for f in intent.get("filters", [])]
        names += intent.get("list_columns", [])
        if intent.get("kind") == "drivers":
            names += [c.name for c in self.driver_factors(intent, restricted=set())]
        for n in names:
            if n in self.ds.columns and self.ds.columns[n].lookup:
                tables.add(self.ds.columns[n].lookup)
        return tables

    def referenced_columns(self, intent: Dict[str, Any]) -> List[str]:
        names = [m["col"] for m in intent.get("measures", []) if m.get("col")]
        names += [g["col"] for g in intent.get("group_by", [])] + [f["col"] for f in intent.get("filters", [])]
        names += intent.get("list_columns", [])
        return list(dict.fromkeys(n for n in names if n in self.ds.columns))

    def driver_factors(self, intent: Dict[str, Any], restricted: Set[str]) -> List[ColumnMeta]:
        outcome = intent["measures"][0]["col"] if intent.get("measures") else None
        filtered = {f["col"] for f in intent.get("filters", []) if f["op"] in ("=", "in")}
        out = []
        for c in self.ds.columns.values():
            if c.name in (outcome, *filtered) or c.name in restricted:
                continue
            if c.nulls > 0.5 * self.ds.row_count:
                continue
            out.append(c)
        return out

    def column_labels(self) -> Dict[str, str]:
        labels = {self.ds.id_column: self.ds.entity["id_label"], self.count_alias(): self.ds.plural.capitalize(),
                  "bucket_start": "Band", "share_pct": "Share %", "factor": "Factor", "level": "Level", "n": self.ds.plural.capitalize()}
        for c in self.ds.columns.values():
            unit = f" ({c.unit})" if c.unit and c.unit not in ("points",) else ""
            labels[c.name] = c.label[:1].upper() + c.label[1:] + unit
            labels[f"{c.name}_band"] = labels[c.name]
            labels[f"avg_{c.name}"] = f"Avg {c.label}"
            labels[f"total_{c.name}"] = f"Total {c.label}"
            labels[f"highest_{c.name}"] = f"Highest {c.label}"
            labels[f"lowest_{c.name}"] = f"Lowest {c.label}"
            labels[f"{c.name}_rate_pct"] = f"{c.label[:1].upper() + c.label[1:]} rate %"
        return labels

    # ------------------------------------------------------------------ verifier expectations

    def expectations(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        kind = intent["kind"]
        measures = intent.get("measures", [])
        funcs: Set[str] = set()
        wrapped: List[Tuple[Optional[str], Tuple[str, str]]] = []
        for m in measures:
            agg = m["agg"]
            if kind in ("correlation", "list") or (kind == "ranking" and intent["ranking"]["target"] == "rows"):
                agg = None
            if agg == "COUNT":
                funcs.add("Count")
            elif agg == "RATE":
                funcs |= {"Sum", "Count"}
                wrapped.append(("Sum", (self.ds.fact_table, m["col"])))
            elif agg:
                funcs.add(AGG_FUNC[agg])
                wrapped.append((AGG_FUNC[agg], (self.ds.fact_table, m["col"])))
            elif m.get("col"):
                wrapped.append((None, (self.ds.fact_table, m["col"])))
        if intent.get("group_by") or kind in ("distribution", "drivers"):
            funcs.add("Count")
        if intent.get("share"):
            funcs.add("Sum")

        groups = []
        for g in intent.get("group_by", []):
            col = self.col(g["col"])
            groups.append((self.ds.fact_table, col.name) if g.get("bin") or col.role == "flag" else self.display_column(col))

        filters = []
        for f in intent.get("filters", []):
            col = self.col(f["col"])
            filters.append((self.display_column(col) if col.lookup else (self.ds.fact_table, col.name), f))

        rate_conditions = [(self.col(m["col"]), m) for m in measures if m["agg"] == "RATE"]
        return {"agg_funcs": funcs, "wrapped": wrapped, "group_columns": groups, "filters": filters,
                "ranking": intent.get("ranking"), "share": bool(intent.get("share")), "rates": rate_conditions}
