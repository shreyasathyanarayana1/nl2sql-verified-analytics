"""Agent 8 — Execution. Runs verified SQL on a read-only connection with a time budget."""
import time
from pathlib import Path
from typing import Any, Dict

from app.config import MAX_RESULT_ROWS
from app.db.catalog import connect_readonly

TIME_BUDGET_S = 5.0


def execute(db_path: Path, sql: str, max_rows: int = MAX_RESULT_ROWS) -> Dict[str, Any]:
    conn = connect_readonly(db_path)
    started = time.perf_counter()
    conn.set_progress_handler(lambda: 1 if time.perf_counter() - started > TIME_BUDGET_S else 0, 10_000)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = [list(r) for r in rows[:max_rows]]
        return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated,
                "exec_ms": round((time.perf_counter() - started) * 1000, 2)}
    finally:
        conn.close()
