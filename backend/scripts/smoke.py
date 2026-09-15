"""Runs questions through each dataset's pipeline (no HTTP) and prints outcomes.

Usage: PYTHONPATH=. .venv/bin/python scripts/smoke.py [inject-mode] [role]
"""
import sys

from app import store
from app.agents import schema_reasoning
from app.datasets import load_all
from app.orchestrator import Pipeline

inject = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "none" else None
role = sys.argv[2] if len(sys.argv) > 2 else "insights_user"
datasets = load_all()
store.init_store(datasets)
for ds in datasets.values():
    schema_reasoning.ensure_index(ds)
    pipeline = Pipeline(ds)
    print(f"\n==================== {ds.name}")
    for group in ds.manifest["suggestions"]:
        for q in group["questions"]:
            session = store.sessions.get_or_create(None, "smoke", ds.id)
            r = pipeline.run(q, {"username": "smoke", "role": role}, session, inject=inject)
            conf = r["confidence"]["score"] if r["confidence"] else "-"
            print(f"\n### {q}\n  status={r['status']} attempts={len(r['attempts'])} confidence={conf} {r['error'] or ''}")
            if r["injected"]:
                print("  injected:", r["injected"]["description"])
            for a in r["attempts"]:
                v = a["validation"]
                if v and not v["passed"]:
                    print("   rejected", a["attempt"], [e["message"] for c in v["checks"].values() for e in c["errors"]][:3])
                if a["repair"]:
                    print("   repair", a["repair"]["strategies"])
            if r["clarification"]:
                print("  ASK:", r["clarification"]["question"], [(o["label"], o.get("preview")) for o in r["clarification"]["options"]])
            if r["result_checks"]:
                failed = [c["name"] for c in r["result_checks"]["checks"] if c["applicable"] and not c["passed"]]
                print(f"  result checks: {r['result_checks']['applicable']} applicable, failed={failed}")
            for w in r["data_warnings"]:
                print("  WARN:", w["severity"], w["title"], "—", w["detail"][:140])
            if r["visualization"]:
                print("  VIZ:", r["visualization"]["chart_type"], "|", r["visualization"]["explanation"][:260])
            if r["final_sql"]:
                print("  SQL:", " ".join(r["final_sql"].split())[:220])
            print("  next:", r["followups"])
