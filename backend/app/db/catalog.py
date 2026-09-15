"""Live schema introspection for a dataset's SQLite database.

The Anti-Hallucination Agent validates against this catalog, never against what the
LLM was shown, so a hallucinated identifier cannot validate itself.
"""
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pathlib import Path


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool


@dataclass
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (self.from_table, self.from_column, self.to_table, self.to_column)


@dataclass
class Catalog:
    tables: Dict[str, List[Column]] = field(default_factory=dict)
    foreign_keys: List[ForeignKey] = field(default_factory=list)

    def has_table(self, table: str) -> bool:
        return table.lower() in self.tables

    def column_names(self, table: str) -> List[str]:
        return [c.name for c in self.tables.get(table.lower(), [])]

    def tables_with_column(self, column: str) -> List[str]:
        return [t for t, cols in self.tables.items() if any(c.name == column.lower() for c in cols)]

    def fk_between(self, t1: str, c1: str, t2: str, c2: str) -> Optional[ForeignKey]:
        for fk in self.foreign_keys:
            if (fk.from_table, fk.from_column, fk.to_table, fk.to_column) in ((t1, c1, t2, c2), (t2, c2, t1, c1)):
                return fk
        return None

    def neighbours(self, table: str) -> List[Tuple[str, ForeignKey]]:
        out = []
        for fk in self.foreign_keys:
            if fk.from_table == table:
                out.append((fk.to_table, fk))
            elif fk.to_table == table:
                out.append((fk.from_table, fk))
        return out

    def join_path(self, start: str, goal: str, priority: Optional[List[str]] = None,
                  allowed: Optional[List[str]] = None) -> Optional[List[ForeignKey]]:
        """Shortest foreign-key path between two tables (BFS over the FK graph).

        `priority` breaks ties between equally short paths; `allowed` restricts which
        tables the path may pass through.
        """
        if start == goal:
            return []
        rank = {t: i for i, t in enumerate(priority or [])}
        frontier = [(start, [])]
        seen = {start}
        while frontier:
            table, path = frontier.pop(0)
            for nxt, fk in sorted(self.neighbours(table), key=lambda n: rank.get(n[0], 99)):
                if allowed is not None and nxt not in allowed:
                    continue
                if nxt in seen:
                    continue
                if nxt == goal:
                    return path + [fk]
                seen.add(nxt)
                frontier.append((nxt, path + [fk]))
        return None


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn


def load_catalog(db_path: Path) -> Catalog:
    catalog = Catalog()
    conn = connect_readonly(db_path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        for table in names:
            cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            catalog.tables[table.lower()] = [Column(c[1].lower(), c[2], bool(c[5])) for c in cols]
            for fk in conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall():
                catalog.foreign_keys.append(ForeignKey(table.lower(), fk[3].lower(), fk[2].lower(), fk[4].lower()))
    finally:
        conn.close()
    return catalog
