"""Platform metadata store, scoped per dataset: schema annotations, access policies,
vector index, audit log. Session memory is kept in-process; the design's Redis cache can
replace `SessionStore` without touching the agents.
"""
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app.config import APP_DB_PATH, HISTORY_WINDOW

_lock = threading.Lock()


@contextmanager
def _conn():
    conn = sqlite3.connect(APP_DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_store(datasets: Dict[str, Any]) -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                dataset_id TEXT NOT NULL, table_name TEXT NOT NULL, column_name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL, PRIMARY KEY (dataset_id, table_name, column_name));
            CREATE TABLE IF NOT EXISTS access_denials (
                dataset_id TEXT NOT NULL, role TEXT NOT NULL, table_name TEXT NOT NULL, column_name TEXT NOT NULL,
                PRIMARY KEY (dataset_id, role, table_name, column_name));
            CREATE TABLE IF NOT EXISTS schema_chunks (
                dataset_id TEXT NOT NULL, table_name TEXT NOT NULL, content TEXT NOT NULL,
                embedding TEXT NOT NULL, indexed_at REAL NOT NULL, PRIMARY KEY (dataset_id, table_name));
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, dataset_id TEXT, username TEXT, role TEXT,
                session_id TEXT, question TEXT, final_sql TEXT, status TEXT, row_count INTEGER,
                latency_ms INTEGER, repair_attempts INTEGER, hallucinations TEXT, confidence INTEGER);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        for ds in datasets.values():
            key = f"seeded:{ds.id}"
            if c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone():
                continue
            c.executemany(
                "INSERT OR IGNORE INTO access_denials VALUES (?,?,?,?)",
                [(ds.id, "insights_user", ds.fact_table, col) for col in ds.manifest.get("restricted_for_insights_user", [])],
            )
            c.execute("INSERT INTO meta VALUES (?, '1')", (key,))


def get_meta(key: str) -> Optional[str]:
    with _conn() as c:
        row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))


# ---- annotations (admin overrides of manifest descriptions) ----------------

def get_annotations(dataset_id: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with _conn() as c:
        for r in c.execute("SELECT table_name, column_name, description FROM annotations WHERE dataset_id=?", (dataset_id,)):
            out.setdefault(r["table_name"], {})[r["column_name"]] = r["description"]
    return out


def set_annotation(dataset_id: str, table: str, column: str, description: str) -> None:
    with _conn() as c:
        if description.strip():
            c.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?,?)", (dataset_id, table, column, description.strip()))
        else:
            c.execute("DELETE FROM annotations WHERE dataset_id=? AND table_name=? AND column_name=?", (dataset_id, table, column))


# ---- access policies -------------------------------------------------------

def get_denials(dataset_id: str, role: str) -> List[Dict[str, str]]:
    with _conn() as c:
        rows = c.execute("SELECT table_name, column_name FROM access_denials WHERE dataset_id=? AND role=?",
                         (dataset_id, role)).fetchall()
    return [{"table": r["table_name"], "column": r["column_name"]} for r in rows]


def set_denials(dataset_id: str, role: str, denials: List[Dict[str, str]]) -> None:
    with _conn() as c:
        c.execute("DELETE FROM access_denials WHERE dataset_id=? AND role=?", (dataset_id, role))
        c.executemany("INSERT OR IGNORE INTO access_denials VALUES (?,?,?,?)",
                      [(dataset_id, role, d["table"], d["column"]) for d in denials])


# ---- vector index ----------------------------------------------------------

def replace_chunks(dataset_id: str, chunks: List[Dict[str, Any]]) -> None:
    now = time.time()
    with _conn() as c:
        c.execute("DELETE FROM schema_chunks WHERE dataset_id=?", (dataset_id,))
        c.executemany("INSERT INTO schema_chunks VALUES (?,?,?,?,?)",
                      [(dataset_id, ch["table"], ch["content"], json.dumps(ch["embedding"]), now) for ch in chunks])


def load_chunks(dataset_id: str) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT table_name, content, embedding, indexed_at FROM schema_chunks WHERE dataset_id=?",
                         (dataset_id,)).fetchall()
    return [{"table": r["table_name"], "content": r["content"], "embedding": json.loads(r["embedding"]), "indexed_at": r["indexed_at"]}
            for r in rows]


# ---- audit log -------------------------------------------------------------

def write_audit(entry: Dict[str, Any]) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO audit_log (ts, dataset_id, username, role, session_id, question, final_sql, status, row_count,"
            " latency_ms, repair_attempts, hallucinations, confidence) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), entry.get("dataset_id"), entry.get("username"), entry.get("role"), entry.get("session_id"),
             entry.get("question"), entry.get("final_sql"), entry.get("status"), entry.get("row_count"), entry.get("latency_ms"),
             entry.get("repair_attempts"), json.dumps(entry.get("hallucinations", [])), entry.get("confidence")),
        )


def read_audit(dataset_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    with _conn() as c:
        if dataset_id:
            rows = c.execute("SELECT * FROM audit_log WHERE dataset_id=? ORDER BY id DESC LIMIT ?", (dataset_id, limit)).fetchall()
        else:
            rows = c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["hallucinations"] = json.loads(d["hallucinations"] or "[]")
        out.append(d)
    return out


# ---- session memory --------------------------------------------------------

class SessionStore:
    """Per-session conversation memory, isolated by session id, owner and dataset."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str], username: str, dataset_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id or "")
            if s and s["owner"] == username and s["dataset_id"] == dataset_id:
                return s
            sid = uuid.uuid4().hex
            self._sessions[sid] = {"id": sid, "owner": username, "dataset_id": dataset_id, "history": [], "pending": None, "turns": {}}
            return self._sessions[sid]

    def get(self, session_id: str, username: str) -> Optional[Dict[str, Any]]:
        s = self._sessions.get(session_id)
        return s if s and s["owner"] == username else None

    @staticmethod
    def remember(session: Dict[str, Any], turn: Dict[str, Any]) -> None:
        session["history"].append(turn)
        # Sliding window: keep only the most recent query-result pairs in context.
        del session["history"][:-HISTORY_WINDOW]


sessions = SessionStore()
