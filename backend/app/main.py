"""FastAPI entry point: auth, dataset-scoped query/clarify/evidence endpoints, and the administrator API."""
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import auth, store
from app.agents import access_control, evidence, schema_reasoning
from app.config import AMBIGUITY_THRESHOLD, CORS_ORIGINS, MAX_REPAIR_ATTEMPTS, SCHEMA_TOP_K
from app.datasets import Dataset, load_all
from app.orchestrator import AGENTS, Pipeline
from app.semantic import Semantic

runtime: Dict[str, Any] = {"datasets": {}, "pipelines": {}}


def load_datasets() -> None:
    datasets = load_all()
    store.init_store(datasets)
    for ds in datasets.values():
        schema_reasoning.ensure_index(ds)
    runtime["datasets"] = datasets
    runtime["pipelines"] = {ds_id: Pipeline(ds) for ds_id, ds in datasets.items()}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    llm.check_available()
    load_datasets()
    yield


app = FastAPI(title="Hallucination-Resistant NL2SQL Platform", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

INJECTION_TYPES = {"none", "schema", "semantic", "structural", "random"}


def dataset(dataset_id: Optional[str]) -> Dataset:
    datasets = runtime["datasets"]
    if not dataset_id:
        return next(iter(datasets.values()))
    if dataset_id not in datasets:
        raise HTTPException(404, f"Unknown dataset “{dataset_id}”")
    return datasets[dataset_id]


# --------------------------------------------------------------------------- models

class LoginRequest(BaseModel):
    username: str
    password: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    dataset_id: Optional[str] = None
    session_id: Optional[str] = None
    inject: Optional[str] = "none"


class ClarifyRequest(BaseModel):
    session_id: str
    option_id: Optional[str] = None
    text: Optional[str] = Field(default=None, max_length=300)


class EvidenceRequest(BaseModel):
    session_id: str
    response_id: str
    key: Optional[Any] = None


class AnnotationRequest(BaseModel):
    dataset_id: str
    table: str
    column: str = ""
    description: str = Field(max_length=500)


class Denial(BaseModel):
    table: str
    column: str


class PolicyRequest(BaseModel):
    dataset_id: str
    role: str
    denied: List[Denial]


def sanitize(text: str) -> str:
    """Strip control characters and prompt-injection style directives before any prompt is built."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"(?i)\b(ignore (all|previous|the above) instructions|system prompt|you are now)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- general

@app.get("/api/health")
def health():
    return {"status": "ok", "llm": "disabled — local ML pipeline",
            "llm_status": {"mode": "local"}, "embedder": "local-tfidf-v1", "agents": AGENTS,
            "datasets": list(runtime["datasets"]),
            "config": {"top_k": SCHEMA_TOP_K, "ambiguity_threshold": AMBIGUITY_THRESHOLD, "max_repair_attempts": MAX_REPAIR_ATTEMPTS}}


@app.post("/api/auth/login")
def login(body: LoginRequest):
    user = auth.authenticate(body.username, body.password)
    return {"token": auth.issue_token(body.username), "user": user}


@app.get("/api/auth/me")
def me(user=Depends(auth.current_user)):
    return user


@app.get("/api/datasets")
def list_datasets(_user=Depends(auth.current_user)):
    return [ds.summary() for ds in runtime["datasets"].values()]


# --------------------------------------------------------------------------- insights

@app.post("/api/query")
def query(body: QueryRequest, user=Depends(auth.current_user)):
    if body.inject and body.inject not in INJECTION_TYPES:
        raise HTTPException(400, f"inject must be one of {sorted(INJECTION_TYPES)}")
    ds = dataset(body.dataset_id)
    question = sanitize(body.question)
    if not question:
        raise HTTPException(400, "Question is empty")
    session = store.sessions.get_or_create(body.session_id, user["username"], ds.id)
    session["pending"] = None
    return runtime["pipelines"][ds.id].run(question, user, session, inject=body.inject)


@app.post("/api/query/clarify")
def clarify(body: ClarifyRequest, user=Depends(auth.current_user)):
    session = store.sessions.get(body.session_id, user["username"])
    if not session or not session.get("pending"):
        raise HTTPException(409, "There is no pending clarification for this session")
    pending = session["pending"]
    option = next((o for o in pending["options"] if o["id"] == body.option_id), None)
    if option:
        question = option["question"]
    elif body.text and sanitize(body.text):
        question = f"{pending['question']} {sanitize(body.text)}"
    else:
        raise HTTPException(400, "Provide option_id or text")
    return runtime["pipelines"][session["dataset_id"]].run(question, user, session, inject=pending["inject"],
                                                           suppressed=pending["suppressed"], clarification_rounds=pending["rounds"])


@app.post("/api/query/evidence")
def query_evidence(body: EvidenceRequest, user=Depends(auth.current_user)):
    session = store.sessions.get(body.session_id, user["username"])
    turn = (session or {}).get("turns", {}).get(body.response_id)
    if not turn:
        raise HTTPException(404, "That answer is no longer available — ask the question again")
    ds = dataset(session["dataset_id"])
    scope = access_control.resolve_scope(ds, user["role"])
    key_column = turn["key_column"] if body.key is not None else None
    try:
        return evidence.fetch(ds, Semantic(ds), scope, turn, key_column, body.key)
    except evidence.EvidenceError as e:
        raise HTTPException(422, str(e))


@app.get("/api/suggestions")
def suggestions(dataset_id: Optional[str] = None, _user=Depends(auth.current_user)):
    return dataset(dataset_id).manifest.get("suggestions", [])


# --------------------------------------------------------------------------- admin

@app.get("/api/admin/schema")
def admin_schema(dataset_id: Optional[str] = None, _admin=Depends(auth.require_admin)):
    ds = dataset(dataset_id)
    annotations = store.get_annotations(ds.id)
    chunks = {c["table"]: c for c in store.load_chunks(ds.id)}
    tables = []
    for t, cols in sorted(ds.catalog.tables.items(), key=lambda kv: (kv[0] != ds.fact_table, kv[0])):
        notes = annotations.get(t, {})
        meta = ds.columns if t == ds.fact_table else {}
        default_desc = ds.manifest.get("table_description", "") if t == ds.fact_table else ds.lookups[t].get("description", "")
        tables.append({
            "name": t,
            "description": notes.get("", default_desc),
            "columns": [{"name": c.name, "type": c.type, "primary_key": c.primary_key,
                         "role": meta[c.name].role if c.name in meta else ("identifier" if c.primary_key else "label"),
                         "nulls": meta[c.name].nulls if c.name in meta else 0,
                         "description": notes.get(c.name, meta[c.name].description if c.name in meta else "")} for c in cols],
            "foreign_keys": [{"column": fk.from_column, "references": f"{fk.to_table}.{fk.to_column}"}
                             for fk in ds.catalog.foreign_keys if fk.from_table == t],
            "indexed_at": chunks.get(t, {}).get("indexed_at"),
            "chunk": chunks.get(t, {}).get("content"),
        })
    return {"dataset": ds.summary(), "tables": tables, "embedder": llm.embedder_name()}


@app.put("/api/admin/schema/annotations")
def admin_annotate(body: AnnotationRequest, _admin=Depends(auth.require_admin)):
    ds = dataset(body.dataset_id)
    if not ds.catalog.has_table(body.table) or (body.column and body.column not in ds.catalog.column_names(body.table)):
        raise HTTPException(404, "Unknown table or column")
    store.set_annotation(ds.id, body.table, body.column, body.description)
    return {"ok": True, "reindex_required": True}


@app.post("/api/admin/reindex")
def admin_reindex(dataset_id: Optional[str] = None, _admin=Depends(auth.require_admin)):
    return schema_reasoning.build_index(dataset(dataset_id))


@app.get("/api/admin/policies")
def admin_policies(dataset_id: Optional[str] = None, _admin=Depends(auth.require_admin)):
    return {"roles": {"insights_user": store.get_denials(dataset(dataset_id).id, "insights_user")}}


@app.put("/api/admin/policies")
def admin_set_policies(body: PolicyRequest, _admin=Depends(auth.require_admin)):
    ds = dataset(body.dataset_id)
    if body.role != "insights_user":
        raise HTTPException(400, "Only the insights_user role has configurable policies")
    for d in body.denied:
        if not ds.catalog.has_table(d.table) or (d.column != "*" and d.column not in ds.catalog.column_names(d.table)):
            raise HTTPException(404, f"Unknown table or column {d.table}.{d.column}")
    store.set_denials(ds.id, body.role, [d.dict() for d in body.denied])
    return {"ok": True}


@app.get("/api/admin/audit")
def admin_audit(dataset_id: Optional[str] = None, limit: int = 200, _admin=Depends(auth.require_admin)):
    rows = store.read_audit(dataset_id, min(max(limit, 1), 1000))
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    return {
        "entries": rows,
        "stats": {
            "total": len(rows), "completed": sum(1 for r in rows if r["status"] == "completed"),
            "hallucinations_detected": sum(1 for r in rows if r["hallucinations"]),
            "repaired": sum(1 for r in rows if r["hallucinations"] and r["status"] == "completed"),
            "blocked": sum(1 for r in rows if r["status"] == "blocked"),
            "p80_latency_ms": latencies[int(0.8 * (len(latencies) - 1))] if latencies else None,
        },
    }
