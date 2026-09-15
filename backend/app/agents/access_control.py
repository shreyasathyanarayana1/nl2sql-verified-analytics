"""RBAC Access Control Agent — resolves which tables and columns the caller may touch.

Runs alongside schema retrieval; its scope is enforced three times: restricted columns are
hidden from the generator's prompt, plans that need them are blocked, and the
Anti-Hallucination schema check rejects any SQL that references them anyway.
"""
from typing import Any, Dict

from app import store
from app.datasets import Dataset


def resolve_scope(ds: Dataset, role: str) -> Dict[str, Any]:
    denied = [] if role == "admin" else store.get_denials(ds.id, role)
    denied_tables = {d["table"] for d in denied if d["column"] == "*"}
    denied_cols = {(d["table"], d["column"]) for d in denied if d["column"] != "*"}
    allowed = {t: [c.name for c in cols if (t, c.name) not in denied_cols]
               for t, cols in ds.catalog.tables.items() if t not in denied_tables}
    return {"role": role, "denied": denied, "allowed": allowed}


def is_denied(scope: Dict[str, Any], table: str, column: str) -> bool:
    return any(d["table"] == table and d["column"] in ("*", column) for d in scope["denied"])
