"""Agent 3 — Schema Reasoning (retrieval-augmented schema grounding).

Offline: one chunk per table (columns with business descriptions, synonyms, units, coded
values, keys) is embedded into the dataset's vector index. Online: the clarified intent is
embedded, the top-k chunks retrieved, and the agent then reasons over the result — issuing
targeted follow-up retrievals for tables the intent needs but the first pass missed, and
adding FK bridge tables needed to connect what it found. Only these tables are ever shown
to the SQL generator.
"""
from typing import Any, Dict, List

from app import llm, store
from app.config import SCHEMA_TOP_K
from app.datasets import Dataset
from app.semantic import Semantic

MIN_SCORE = 0.05


def chunk_text(ds: Dataset, table: str) -> str:
    notes = store.get_annotations(ds.id).get(table, {})
    if table == ds.fact_table:
        lines = [f"Table {table}: {notes.get('', ds.manifest.get('table_description', ''))} One row per {ds.singular}."]
        lines.append(f"- {ds.id_column} (INTEGER primary key): identifier of the {ds.singular}")
        for c in ds.columns.values():
            desc = notes.get(c.name, c.description)
            extra = []
            if c.unit:
                extra.append(f"unit {c.unit}")
            if c.role == "category" and c.values:
                extra.append("values " + ", ".join(map(str, c.values)))
            if c.role == "flag":
                extra.append(f"yes = {c.positive}, no = {c.negative}")
            if c.synonyms:
                extra.append("also called " + ", ".join(c.synonyms[:6]))
            ref = f" references {c.lookup}(code)" if c.lookup else ""
            lines.append(f"- {c.name} ({c.type}, {c.role}{ref}): {desc}" + (f" [{'; '.join(extra)}]" if extra else ""))
    else:
        lk = ds.lookups[table]
        col = next(c for c in ds.columns.values() if c.lookup == table)
        lines = [f"Table {table}: {notes.get('', lk.get('description', ''))} Names for {col.label} codes.",
                 "- code (INTEGER primary key)", "- name (TEXT): " + ", ".join(f"{k} = {v}" for k, v in lk["values"].items()),
                 f"- joined from {ds.fact_table}.{col.name}; also called " + ", ".join(col.synonyms[:5])]
    return "\n".join(lines)


def build_index(ds: Dataset) -> Dict[str, Any]:
    tables = sorted(ds.catalog.tables)
    texts = [chunk_text(ds, t) for t in tables]
    try:
        vectors = llm.embed(texts)
    except Exception:
        vectors = llm.embed(texts)  # the failed call switched llm to offline mode
    store.replace_chunks(ds.id, [{"table": t, "content": c, "embedding": v} for t, c, v in zip(tables, texts, vectors)])
    store.set_meta(f"embedder:{ds.id}", llm.embedder_name())
    return {"chunks": len(tables), "embedder": llm.embedder_name()}


def ensure_index(ds: Dataset) -> None:
    if not store.load_chunks(ds.id) or store.get_meta(f"embedder:{ds.id}") != llm.embedder_name():
        build_index(ds)


def _embed(ds: Dataset, texts: List[str]) -> List[List[float]]:
    """Embeds with the active embedder; if OpenAI fails mid-session, rebuild the index offline and retry."""
    try:
        return llm.embed(texts)
    except Exception:
        build_index(ds)
        return llm.embed(texts)


def intent_text(ds: Dataset, intent: Dict[str, Any]) -> str:
    parts = [intent["raw"]]
    for name in Semantic(ds).referenced_columns(intent):
        c = ds.columns[name]
        parts += [c.label, name]
    return " ".join(parts)


def retrieve(ds: Dataset, sem: Semantic, intent: Dict[str, Any], k: int = SCHEMA_TOP_K) -> Dict[str, Any]:
    ensure_index(ds)
    query_vec = _embed(ds, [intent_text(ds, intent)])[0]
    chunks = store.load_chunks(ds.id)
    scored = sorted(
        ({"table": ch["table"], "content": ch["content"], "score": round(llm.cosine(query_vec, ch["embedding"]), 4)} for ch in chunks),
        key=lambda c: c["score"], reverse=True,
    )
    selected = [dict(c, reason="top-k similarity") for c in scored[:k] if c["score"] >= MIN_SCORE]
    by_table = {c["table"]: c for c in scored}
    actions: List[str] = []

    # Follow-up retrieval: every table the intent depends on must be grounded.
    needed = sem.required_tables(intent)
    for table in sorted(needed):
        if table in {c["table"] for c in selected} or table not in by_table:
            continue
        probe_terms = table.replace("_", " ")
        if table != ds.fact_table:
            col = next((c for c in ds.columns.values() if c.lookup == table), None)
            probe_terms += " " + (col.label if col else "")
        probe = _embed(ds, [probe_terms.strip()])[0]
        unselected = [ch for ch in chunks if ch["table"] not in {c["table"] for c in selected}]
        best = max(unselected, key=lambda ch: llm.cosine(probe, ch["embedding"]))
        selected.append(dict(by_table[table], reason=f"follow-up retrieval for “{probe_terms.strip()}”"))
        if best["table"] == table:
            actions.append(f"Initial top-{k} missed {table}; a targeted retrieval for “{probe_terms.strip()}” found it.")
        else:
            actions.append(f"Initial top-{k} missed {table}; the targeted retrieval ranked {best['table']} first, "
                           f"so {table} was added because the question's columns require it.")

    # Drop weakly related chunks nothing in the intent depends on (keeps the prompt small).
    top_score = scored[0]["score"] if scored else 0
    pruned = [c for c in selected if c["table"] in needed or c["score"] >= 0.9 * top_score]
    for c in selected:
        if c not in pruned:
            actions.append(f"Pruned {c['table']} (score {c['score']}) — retrieved but not needed for this question.")

    # Bridge expansion: the kept tables must be connected through real foreign keys.
    names = [c["table"] for c in pruned]
    if ds.fact_table in names:
        for table in names:
            for fk in ds.catalog.join_path(ds.fact_table, table) or []:
                for t in (fk.from_table, fk.to_table):
                    if t not in {c["table"] for c in pruned}:
                        pruned.append(dict(by_table[t], reason=f"FK bridge to {table}"))
                        actions.append(f"Added {t} as a join bridge ({fk.from_table}.{fk.from_column} → {fk.to_table}.{fk.to_column}).")

    return {
        "k": k, "embedder": llm.embedder_name(),
        "candidates": [{"table": c["table"], "score": c["score"]} for c in scored],
        "chunks": pruned, "tables": [c["table"] for c in pruned], "actions": actions,
        "total_tables": len(ds.catalog.tables),
    }
