"""Dataset registry: imports CSV datasets into SQLite and exposes their semantic layer.

Each folder under backend/datasets/ holds a source CSV and a manifest.json describing the
columns in business terms (label, role, unit, synonyms, category order, coded values).
The manifest is the only domain knowledge in the system; every agent reads from it, so
a new dataset needs a manifest, not code changes.

Roles:
  measure   numeric column that can be averaged, summed, binned or correlated
  category  a column you group or filter by (optionally coded, joined to a lookup table)
  flag      a yes/no column; its "rate" is the share of rows with the positive value
"""
import csv
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import BASE_DIR, DATA_DIR
from app.db.catalog import Catalog, connect_readonly, load_catalog

DATASETS_DIR = BASE_DIR / "datasets"


@dataclass
class ColumnMeta:
    name: str
    table: str
    label: str
    role: str
    type: str
    description: str = ""
    unit: str = ""
    synonyms: List[str] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    value_synonyms: Dict[str, List[str]] = field(default_factory=dict)
    positive: Any = None
    negative: Any = None
    lookup: Optional[str] = None
    bin_width: Optional[float] = None
    outcome: bool = False
    # profiled from the imported data
    values: List[Any] = field(default_factory=list)
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    nulls: int = 0

    @property
    def is_numeric(self) -> bool:
        return self.role == "measure"

    def unit_suffix(self) -> str:
        return "%" if self.unit == "%" else (f" {self.unit}" if self.unit else "")


@dataclass
class Dataset:
    id: str
    name: str
    description: str
    source: str
    entity: Dict[str, str]
    fact_table: str
    id_column: str
    db_path: Path
    manifest: Dict[str, Any]
    columns: Dict[str, ColumnMeta]
    lookups: Dict[str, Dict[str, Any]]
    row_count: int
    catalog: Catalog

    @property
    def singular(self) -> str:
        return self.entity["singular"]

    @property
    def plural(self) -> str:
        return self.entity["plural"]

    @property
    def primary_measure(self) -> ColumnMeta:
        return self.columns[self.manifest["primary_measure"]]

    def alias(self, table: str) -> str:
        if table == self.fact_table:
            return "t"
        return "".join(part[0] for part in table.split("_")) + "x"

    def measures(self) -> List[ColumnMeta]:
        return [c for c in self.columns.values() if c.role == "measure"]

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description, "source": self.source,
            "entity": self.entity, "rows": self.row_count, "columns": len(self.columns),
            "tables": sorted(self.catalog.tables), "primary_measure": self.primary_measure.label,
        }


# --------------------------------------------------------------------------- import

def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _convert(raw: str, col: Dict[str, Any], markers: List[str]) -> Any:
    value = raw.strip()
    if value in markers:
        return None
    if "value_map" in col:
        return col["value_map"].get(value.split(".")[0], value)
    if col["type"] in ("INTEGER", "REAL"):
        number = float(value)
        return int(number) if col["type"] == "INTEGER" and number.is_integer() else number
    return value


def _needs_import(folder: Path, db_path: Path) -> bool:
    if not db_path.exists():
        return True
    newest = max((folder / "manifest.json").stat().st_mtime, (folder / "source.csv").stat().st_mtime)
    return newest > db_path.stat().st_mtime


def import_dataset(folder: Path, force: bool = False) -> Path:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    db_path = DATA_DIR / f"{manifest['id']}.db"
    if not force and not _needs_import(folder, db_path):
        return db_path

    fact = manifest["fact_table"]
    id_column = f"{manifest['entity']['singular']}_id"
    lookups = {lk["table"]: lk for lk in manifest.get("lookups", [])}
    markers = manifest.get("missing_markers", ["", "?"])

    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    try:
        for table, lk in lookups.items():
            conn.execute(f"CREATE TABLE {table} (code INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
            conn.executemany(f"INSERT INTO {table} VALUES (?, ?)", [(int(k), v) for k, v in lk["values"].items()])

        defs = [f"{id_column} INTEGER PRIMARY KEY"]
        fks = []
        for col in manifest["columns"]:
            sql_type = "TEXT" if "value_map" in col else col["type"]
            defs.append(f"{col['name']} {sql_type}")
            if col.get("lookup"):
                fks.append(f"FOREIGN KEY ({col['name']}) REFERENCES {col['lookup']}(code)")
        conn.execute(f"CREATE TABLE {fact} ({', '.join(defs + fks)})")

        with open(folder / manifest.get("csv", "source.csv"), newline="") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            index = {h: i for i, h in enumerate(header)}
            missing = [c["source"] for c in manifest["columns"] if c["source"] not in index]
            if missing:
                raise ValueError(f"{manifest['id']}: CSV is missing columns {missing}")
            rows = []
            for n, record in enumerate(reader, start=1):
                if not any(cell.strip() for cell in record):
                    continue
                values = [n] + [_convert(record[index[c["source"]]] if index[c["source"]] < len(record) else "", c, markers)
                                for c in manifest["columns"]]
                rows.append(values)
        placeholders = ", ".join("?" for _ in range(len(manifest["columns"]) + 1))
        conn.executemany(f"INSERT INTO {fact} VALUES ({placeholders})", rows)
        conn.commit()
    finally:
        conn.close()
    tmp.replace(db_path)
    return db_path


# --------------------------------------------------------------------------- load

def _profile(db_path: Path, fact: str, columns: Dict[str, ColumnMeta], lookups: Dict[str, Dict[str, Any]]) -> int:
    conn = connect_readonly(db_path)
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM {fact}").fetchone()[0]
        for col in columns.values():
            col.nulls = conn.execute(f"SELECT COUNT(*) - COUNT({col.name}) FROM {fact}").fetchone()[0]
            if col.role == "measure":
                col.minimum, col.maximum = conn.execute(f"SELECT MIN({col.name}), MAX({col.name}) FROM {fact}").fetchone()
            elif col.lookup:
                col.values = [r[0] for r in conn.execute(f"SELECT name FROM {col.lookup} ORDER BY code")]
            else:
                present = [r[0] for r in conn.execute(f"SELECT DISTINCT {col.name} FROM {fact} WHERE {col.name} IS NOT NULL")]
                col.values = [v for v in col.order if v in present] + sorted((v for v in present if v not in col.order), key=str)
        return total
    finally:
        conn.close()


def load_dataset(folder: Path) -> Dataset:
    db_path = import_dataset(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    fact = manifest["fact_table"]
    columns: Dict[str, ColumnMeta] = {}
    for c in manifest["columns"]:
        columns[c["name"]] = ColumnMeta(
            name=c["name"], table=fact, label=c.get("label", c["name"].replace("_", " ")), role=c["role"], type=c["type"],
            description=c.get("description", ""), unit=c.get("unit", ""), synonyms=c.get("synonyms", []),
            order=c.get("order", []), value_synonyms=c.get("value_synonyms", {}), positive=c.get("positive"),
            negative=c.get("negative"), lookup=c.get("lookup"), bin_width=c.get("bin_width"), outcome=c.get("outcome", False),
        )
    lookups = {lk["table"]: lk for lk in manifest.get("lookups", [])}
    rows = _profile(db_path, fact, columns, lookups)
    return Dataset(
        id=manifest["id"], name=manifest["name"], description=manifest["description"], source=manifest["source"],
        entity=manifest["entity"], fact_table=fact, id_column=f"{manifest['entity']['singular']}_id", db_path=db_path,
        manifest=manifest, columns=columns, lookups=lookups, row_count=rows, catalog=load_catalog(db_path),
    )


def load_all() -> Dict[str, Dataset]:
    datasets: Dict[str, Dataset] = {}
    for folder in sorted(p for p in DATASETS_DIR.iterdir() if (p / "manifest.json").exists()):
        ds = load_dataset(folder)
        datasets[ds.id] = ds
    return datasets


if __name__ == "__main__":
    for ds in load_all().values():
        print(f"{ds.id}: {ds.row_count} rows → {ds.db_path}")
