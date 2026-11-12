"""Orchestration layer — an explicit state machine over the agents, one pipeline per dataset.

Mirrors the LangGraph design: named nodes, one shared state object, and conditional edges
implementing the two feedback loops (clarification before generation, verification ⇄ repair
after it) plus the result-level repair after execution. Each node returns the next node.
"""
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from app import store
from app.agents import (access_control, anti_hallucination, clarification, confidence, data_quality, execution,
                        followups, intent as intent_agent, narrative, planning, repair, result_checks, schema_reasoning,
                        sql_generation, visualization)
from app.config import MAX_REPAIR_ATTEMPTS
from app.datasets import Dataset
from app.semantic import Semantic

END = "__end__"

AGENTS = [
    {"key": "intent", "step": "1", "name": "Intent Understanding"},
    {"key": "clarification", "step": "2", "name": "Query Clarification"},
    {"key": "schema", "step": "3", "name": "Schema Reasoning (RAG)"},
    {"key": "access", "step": "3b", "name": "RBAC Access Control"},
    {"key": "planning", "step": "4", "name": "Query Planning"},
    {"key": "generation", "step": "5", "name": "SQL Generation"},
    {"key": "verification", "step": "6", "name": "Anti-Hallucination"},
    {"key": "repair", "step": "7", "name": "Query Repair"},
    {"key": "execution", "step": "8", "name": "Execution"},
    {"key": "result_checks", "step": "8b", "name": "Result Verification"},
    {"key": "quality", "step": "8c", "name": "Data Quality & Confidence"},
    {"key": "visualization", "step": "9", "name": "Visualization"},
    {"key": "explanation", "step": "10", "name": "Explanation"},
]
AGENT_NAMES = {a["key"]: a["name"] for a in AGENTS}


def describe_intent(ds: Dataset, sem: Semantic, i: Dict[str, Any]) -> str:
    bits = [f"kind={i['kind']}"]
    if i["measures"]:
        bits.append("measure: " + ", ".join(sem.measure_label(m) if m["agg"] else ds.columns[m["col"]].label for m in i["measures"]))
    if i["group_by"]:
        bits.append("by " + ", ".join(ds.columns[g["col"]].label + (f" (bands of {g['bin']:g})" if g["bin"] else "") for g in i["group_by"]))
    if i["filters"]:
        bits.append("filters: " + "; ".join(f["label"] for f in i["filters"]))
    if i["ranking"]:
        bits.append(f"{'top' if i['ranking']['direction'] == 'DESC' else 'bottom'} {i['ranking']['limit']} {i['ranking']['target']}")
    if i["follow_up"]:
        bits.append("follow-up")
    return " · ".join(bits)


def _trim(result: Optional[Dict[str, Any]], limit: int = 500) -> Optional[Dict[str, Any]]:
    """Large results (e.g. every point of a correlation) are analysed in full but only a page is sent to the browser."""
    if not result or result["row_count"] <= limit:
        return result
    return {**result, "rows": result["rows"][:limit], "row_count": result["row_count"], "truncated": True, "shown": limit}


class Pipeline:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.sem = Semantic(ds)
        self.nodes: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "intent": self.node_intent, "clarification": self.node_clarification, "ground": self.node_ground,
            "planning": self.node_planning, "generation": self.node_generation, "verification": self.node_verification,
            "repair": self.node_repair, "execution": self.node_execution, "result_checks": self.node_result_checks,
            "quality": self.node_quality, "visualization": self.node_visualization,
        }

    # ------------------------------------------------------------------ runner

    def run(self, question: str, user: Dict[str, Any], session: Dict[str, Any], inject: Optional[str] = None,
            suppressed: Optional[List[str]] = None, clarification_rounds: int = 0) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "question": question, "user": user, "session": session, "inject": inject if inject not in (None, "", "none") else None,
            "suppressed": suppressed or [], "clarification_rounds": clarification_rounds,
            "status": None, "error": None, "trace": [], "intent": None, "clarification": None,
            "schema_context": None, "access_scope": access_control.resolve_scope(self.ds, user["role"]), "plan": None,
            "attempts": [], "sql": None, "final_sql": None, "result": None, "visualization": None, "warnings": [],
            "injected": None, "result_checks": None, "result_repaired": False, "data_warnings": [], "confidence": None,
            "followups": [], "auto_resolved": None, "narrative": None, "started": time.perf_counter(),
        }
        node = "intent"
        while node != END:
            node = self.nodes[node](state)
        return self.finish(state)

    def record(self, state, key: str, status: str, started: float, summary: str, detail: Any = None, attempt: Optional[int] = None):
        state["trace"].append({"key": key, "name": AGENT_NAMES[key], "status": status, "summary": summary, "detail": detail,
                               "attempt": attempt, "duration_ms": round((time.perf_counter() - started) * 1000, 2)})

    def denied_columns(self, state) -> set:
        return {c for c in self.ds.columns if access_control.is_denied(state["access_scope"], self.ds.fact_table, c)}

    # ------------------------------------------------------------------ nodes

    def node_intent(self, state) -> str:
        t0 = time.perf_counter()
        history = state["session"]["history"]
        previous = history[-1]["intent"] if history else None
        state["intent"] = intent_agent.extract_intent(self.ds, state["question"], previous)
        self.record(state, "intent", "done", t0, describe_intent(self.ds, self.sem, state["intent"]), state["intent"])
        return "clarification"

    def node_clarification(self, state) -> str:
        t0 = time.perf_counter()
        suppressed = list(state["suppressed"])
        if state["clarification_rounds"] >= 2:
            suppressed += ["missing_measure", "unknown_measure", "ambiguous_value", "unsupported_intent", "unmatched_terms"]
        assessment = clarification.assess(self.ds, state["intent"], suppressed, self.denied_columns(state))
        state["clarification"] = assessment
        if assessment["needs_clarification"] and assessment["signal"] != "unsupported_intent":
            comparison = self.compare_interpretations(state, assessment)
            if comparison["agree"]:
                chosen = assessment["options"][0]
                state["auto_resolved"] = {"original": state["question"], "chosen": chosen["label"],
                                          "compared": [o["label"] for o in assessment["options"]], "detail": comparison["detail"]}
                self.record(state, "clarification", "done", t0,
                            f"ambiguity {assessment['ambiguity_score']} but every interpretation gives the same answer — not asking",
                            {**assessment, "comparison": comparison})
                state["question"] = chosen["question"]
                state["suppressed"] = suppressed + [assessment["signal"]]
                return "intent"
            for option, preview in zip(assessment["options"], comparison["previews"]):
                option["preview"] = preview
        if assessment["needs_clarification"]:
            self.record(state, "clarification", "waiting", t0,
                        f"ambiguity {assessment['ambiguity_score']} > {assessment['threshold']} — asking the user", assessment)
            state["status"] = "needs_clarification"
            state["session"]["pending"] = {"question": state["question"], "signal": assessment["signal"], "options": assessment["options"],
                                           "suppressed": suppressed + [assessment["signal"]], "rounds": state["clarification_rounds"] + 1,
                                           "inject": state["inject"]}
            return END
        self.record(state, "clarification", "done", t0,
                    f"ambiguity {assessment['ambiguity_score']} ≤ {assessment['threshold']} — proceeding", assessment)
        return "ground"

    def node_ground(self, state) -> str:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            schema_future = pool.submit(schema_reasoning.retrieve, self.ds, self.sem, state["intent"])
            scope_future = pool.submit(access_control.resolve_scope, self.ds, state["user"]["role"])
            ctx, scope = schema_future.result(), scope_future.result()
        state["schema_context"], state["access_scope"] = ctx, scope
        self.record(state, "schema", "done", t0, f"retrieved {len(ctx['tables'])} of {ctx['total_tables']} tables: {', '.join(ctx['tables'])}", ctx)
        denied = ", ".join(f"{d['table']}.{d['column']}" for d in scope["denied"]) or "none"
        self.record(state, "access", "done", t0, f"role {scope['role']} · restricted: {denied}", scope)
        return "planning"

    def node_planning(self, state) -> str:
        t0 = time.perf_counter()
        try:
            state["plan"] = planning.build_plan(self.ds, self.sem, state["intent"], state["schema_context"], state["access_scope"])
        except planning.PlanningError as e:
            state["status"] = "blocked" if e.status == "blocked" else "failed"
            state["error"] = str(e)
            self.record(state, "planning", "failed", t0, state["error"])
            return END
        self.record(state, "planning", "done", t0, f"{len(state['plan']['steps'])} steps", state["plan"]["steps"])
        return "generation"

    def history(self, state) -> List[Dict[str, Any]]:
        return [{"question": h["question"], "sql": h["sql"]} for h in state["session"]["history"]]

    def node_generation(self, state) -> str:
        t0 = time.perf_counter()
        out = sql_generation.generate(self.ds, state["intent"], state["plan"], state["schema_context"], state["access_scope"], self.history(state))
        sql = out["sql"]
        summary = f"draft SQL via {out['model']}"
        if state["inject"]:
            injected = sql_generation.inject_hallucination(self.ds, sql, state["inject"], random.Random(state["question"]))
            if injected["type"]:
                sql = injected["sql"]
                state["injected"] = injected
                summary += f" · evaluation: injected {injected['type']} hallucination"
        state["sql"] = sql
        state["attempts"].append({"attempt": 1, "sql": sql, "source": out["model"], "validation": None, "repair": None})
        self.record(state, "generation", "done", t0, summary,
                    {"sql": sql, "injected": state["injected"], "template": out.get("ml_generation")})
        return "verification"

    def node_verification(self, state) -> str:
        t0 = time.perf_counter()
        attempt = state["attempts"][-1]
        report = anti_hallucination.verify(state["sql"], state["intent"], self.sem, state["access_scope"])
        attempt["validation"] = report
        for w in report["checks"]["semantic"].get("warnings", []):
            if w not in state["warnings"]:
                state["warnings"].append(w)
        summary = "all 3 checks passed" if report["passed"] else "failed: " + ", ".join(report["hallucination_types"])
        self.record(state, "verification", "done" if report["passed"] else "failed", t0, f"attempt {attempt['attempt']}: {summary}",
                    {"failed": report["hallucination_types"]}, attempt=attempt["attempt"])
        if report["passed"]:
            state["final_sql"] = state["sql"]
            return "execution"
        if len(state["attempts"]) > MAX_REPAIR_ATTEMPTS:
            state["status"] = "failed"
            errors = [e["message"] for c in report["checks"].values() for e in c["errors"]]
            state["error"] = (f"The query could not be verified after {MAX_REPAIR_ATTEMPTS} repair attempts, so it was not run. "
                              f"Remaining issues: {'; '.join(errors[:3])}")
            return END
        return "repair"

    def node_repair(self, state) -> str:
        t0 = time.perf_counter()
        attempt = state["attempts"][-1]

        def regenerate(feedback: List[str]) -> Dict[str, Any]:
            return sql_generation.generate(self.ds, state["intent"], state["plan"], state["schema_context"], state["access_scope"],
                                           self.history(state), feedback=feedback)

        def recheck(sql: str) -> Dict[str, Any]:
            return anti_hallucination.verify(sql, state["intent"], self.sem, state["access_scope"])

        fix = repair.repair(state["sql"], attempt["validation"], self.ds.catalog, regenerate, recheck)
        attempt["repair"] = fix
        if fix["blocked"]:
            state["status"] = "blocked"
            state["error"] = "This query was blocked: " + "; ".join(fix["changes"])
            self.record(state, "repair", "failed", t0, "not repairable — access policy violation", fix, attempt=attempt["attempt"])
            return END
        state["sql"] = fix["sql"]
        state["attempts"].append({"attempt": attempt["attempt"] + 1, "sql": fix["sql"], "source": "repair", "validation": None, "repair": None})
        self.record(state, "repair", "done", t0, " + ".join(fix["strategies"]) + f" → re-verify (attempt {attempt['attempt'] + 1})",
                    fix, attempt=attempt["attempt"])
        return "verification"

    def node_execution(self, state) -> str:
        t0 = time.perf_counter()
        try:
            max_rows = 50_000 if state["intent"]["kind"] == "correlation" else 500
            state["result"] = execution.execute(self.ds.db_path, state["final_sql"], max_rows=max_rows)
        except Exception as e:
            state["status"] = "failed"
            state["error"] = f"Database error: {e}"
            self.record(state, "execution", "failed", t0, state["error"])
            return END
        r = state["result"]
        self.record(state, "execution", "done", t0, f"{r['row_count']} rows in {r['exec_ms']} ms (read-only)")
        return "result_checks"

    def node_result_checks(self, state) -> str:
        t0 = time.perf_counter()
        report = result_checks.verify_result(self.ds, self.sem, state["intent"], state["plan"], state["access_scope"],
                                             state["final_sql"], state["result"])
        ref = report.pop("reference")
        if "reference_agreement" in report["failed"] and ref and not state["result_repaired"]:
            ref_report = anti_hallucination.verify(ref["sql"], state["intent"], self.sem, state["access_scope"])
            if ref_report["passed"]:
                last = state["attempts"][-1]
                last["repair"] = {"sql": ref["sql"], "blocked": False, "strategies": ["result mismatch → independently planned query"],
                                  "changes": [next(c["detail"] for c in report["checks"] if c["name"] == "reference_agreement"),
                                              "Replaced the answer with the reference query, which passed all three SQL checks."]}
                state["attempts"].append({"attempt": last["attempt"] + 1, "sql": ref["sql"], "source": "result-level repair",
                                          "validation": ref_report, "repair": None})
                state["final_sql"], state["result"], state["result_repaired"] = ref["sql"], ref["result"], True
                self.record(state, "result_checks", "failed", t0, "answer disagreed with the reference query — replaced and re-checking", report)
                return "result_checks"
        state["result_checks"] = report
        summary = (f"{report['applicable']} check{'s' if report['applicable'] != 1 else ''} passed" if report["passed"]
                   else "failed: " + ", ".join(report["failed"]))
        self.record(state, "result_checks", "done" if report["passed"] else "failed", t0, summary, report)
        return "quality"

    def node_quality(self, state) -> str:
        t0 = time.perf_counter()
        state["data_warnings"] = data_quality.assess(self.ds, self.sem, state["intent"], state["access_scope"], state["plan"], state["result"])
        state["intent"]["notes"] += state["plan"].get("notes", [])
        state["confidence"] = confidence.compute(state)
        c = state["confidence"]
        n = len(state["data_warnings"])
        self.record(state, "quality", "done", t0, f"confidence {c['score']} ({c['level']}) · {n} data warning{'s' if n != 1 else ''}",
                    {"warnings": state["data_warnings"], "confidence": c})
        return "visualization"

    def node_visualization(self, state) -> str:
        t0 = time.perf_counter()
        state["visualization"] = visualization.build(self.ds, self.sem, state["intent"], state["result"], state["plan"])
        state["followups"] = followups.suggest(self.ds, state["intent"], state["result"], state["visualization"],
                                               state["access_scope"], state["question"])
        self.record(state, "visualization", "done", t0, f"chart: {state['visualization']['chart_type']}")
        t0 = time.perf_counter()
        state["narrative"] = narrative.build(self.ds, self.sem, state)
        state["status"] = "completed"
        sections = state["narrative"]["sections"]
        self.record(state, "explanation", "done", t0,
                    f"{len(sections)} sections · {sum(len(s['paragraphs']) + len(s['bullets']) for s in sections)} statements, all computed from the result")
        return END

    # ------------------------------------------------------------------ interpretations

    def preview(self, question: str, user: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs one interpretation end to end (verified, read-only) without recording it."""
        it = intent_agent.extract_intent(self.ds, question, previous)
        scope = access_control.resolve_scope(self.ds, user["role"])
        plan = planning.build_plan(self.ds, self.sem, it, {"tables": sorted(self.ds.catalog.tables)}, scope)
        sql = planning.render_sql(plan["spec"])
        if not anti_hallucination.verify(sql, it, self.sem, scope)["passed"]:
            raise ValueError("interpretation failed verification")
        return {"intent": it, "result": execution.execute(self.ds.db_path, sql), "alias": plan["spec"].get("metric_alias")}

    def compare_interpretations(self, state, assessment) -> Dict[str, Any]:
        history = state["session"]["history"]
        previous = history[-1]["intent"] if history else None
        previews, keys = [], []
        for option in assessment["options"]:
            try:
                run = self.preview(option["question"], state["user"], previous)
            except Exception:
                previews.append(None)
                keys.append(None)
                continue
            res, it = run["result"], run["intent"]
            if not res["rows"] or run["alias"] not in res["columns"]:
                previews.append("no matching rows" if not res["rows"] else None)
                keys.append(None)
                continue
            vi = res["columns"].index(run["alias"])
            unit = "%" if it["measures"] and it["measures"][0]["agg"] == "RATE" else ""
            if it["kind"] == "ranking" and it["ranking"]["target"] == "rows":
                ii = res["columns"].index(self.ds.id_column)
                previews.append(f"#1 would be {self.ds.entity['id_label']} {res['rows'][0][ii]} ({visualization._fmt(res['rows'][0][vi])})")
                keys.append([r[ii] for r in res["rows"]])
            elif it["group_by"]:
                g = it["group_by"][0]
                key = f"{g['col']}_band" if g["bin"] else g["col"]
                gi = res["columns"].index(key)
                ranked = sorted(res["rows"], key=lambda r: (r[vi] is None, -(r[vi] or 0)))
                previews.append(f"Highest: {ranked[0][gi]} ({visualization._fmt(ranked[0][vi])}{unit})")
                keys.append([r[gi] for r in ranked])
            else:
                previews.append(f"= {visualization._fmt(res['rows'][0][vi])}{unit}")
                keys.append(None)
        valid = [k for k in keys if k]
        agree = (len(valid) == len(keys) >= 2 and all(k[0] == valid[0][0] for k in valid)
                 and all(len(set(k[:10]) & set(valid[0][:10])) / max(len(set(k[:10]) | set(valid[0][:10])), 1) >= 0.7 for k in valid))
        labels = [o["label"] for o in assessment["options"]]
        detail = (f"{', '.join(labels[:-1])} and {labels[-1]} all give the same top result, so I used {labels[0]}"
                  if agree else "The interpretations give different answers, so it's worth choosing")
        return {"agree": agree, "previews": previews, "detail": detail}

    # ------------------------------------------------------------------ output

    def finish(self, state) -> Dict[str, Any]:
        session = state["session"]
        latency = round((time.perf_counter() - state["started"]) * 1000)
        if state["status"] != "needs_clarification":
            session["pending"] = None
        if state["status"] == "completed":
            store.SessionStore.remember(session, {"question": state["question"], "intent": state["intent"], "sql": state["final_sql"]})

        detected = [{"attempt": a["attempt"], "types": a["validation"]["hallucination_types"]}
                    for a in state["attempts"] if a["validation"] and not a["validation"]["passed"]]
        if state["status"] != "needs_clarification":
            store.write_audit({
                "dataset_id": self.ds.id, "username": state["user"]["username"], "role": state["user"]["role"],
                "session_id": session["id"], "question": state["question"], "final_sql": state["final_sql"] or state["sql"],
                "status": state["status"], "row_count": state["result"]["row_count"] if state["result"] else None,
                "latency_ms": latency, "repair_attempts": max(0, len(state["attempts"]) - 1), "hallucinations": detected,
                "confidence": state["confidence"]["score"] if state["confidence"] else None,
            })

        response_id = uuid.uuid4().hex
        intent = state["intent"] or {}
        if state["status"] == "completed":
            session["turns"][response_id] = {
                "sql": state["final_sql"], "kind": intent["kind"], "key_column": state["visualization"]["key_column"],
                "rows_ranking": bool(intent.get("ranking") and intent["ranking"]["target"] == "rows"), "question": state["question"],
            }
            for old in list(session["turns"])[:-20]:
                del session["turns"][old]
        return {
            "response_id": response_id, "session_id": session["id"], "dataset_id": self.ds.id, "status": state["status"],
            "question": state["question"], "error": state["error"], "notes": list(intent.get("notes", [])), "warnings": state["warnings"],
            "clarification": state["clarification"] if state["status"] == "needs_clarification" else None,
            "ambiguity": state["clarification"], "intent": intent, "schema_context": state["schema_context"],
            "access_scope": {"role": state["access_scope"]["role"], "denied": state["access_scope"]["denied"]},
            "plan": state["plan"]["steps"] if state["plan"] else None, "attempts": state["attempts"], "injected": state["injected"],
            "final_sql": state["final_sql"], "result": _trim(state["result"]), "visualization": state["visualization"],
            "column_labels": self.sem.column_labels(), "result_checks": state["result_checks"], "data_warnings": state["data_warnings"],
            "confidence": state["confidence"], "followups": state["followups"], "auto_resolved": state["auto_resolved"],
            "narrative": state["narrative"], "trace": state["trace"], "latency_ms": latency,
        }
