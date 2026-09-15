"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Shell from "@/components/Shell";
import { api } from "@/lib/api";
import type { AuditResponse, SchemaTable } from "@/lib/types";
import { useAuth } from "@/lib/useAuth";
import { useDatasets } from "@/lib/useDatasets";

type Tab = "schema" | "policies" | "audit";

export default function AdminPage() {
  const user = useAuth("admin");
  const [tab, setTab] = useState<Tab>("schema");
  const { datasets, selected, select } = useDatasets(Boolean(user));

  if (!user) return null;
  return (
    <Shell
      user={user}
      title="Administration"
      actions={
        <select className="select" style={{ width: 260 }} value={selected?.id ?? ""} onChange={(e) => select(e.target.value)} aria-label="Dataset">
          {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
      }
    >
      <div className="page">
        <div className="tabs" style={{ marginBottom: 18 }}>
          <button className={`tab ${tab === "schema" ? "active" : ""}`} onClick={() => setTab("schema")}>Schema &amp; annotations</button>
          <button className={`tab ${tab === "policies" ? "active" : ""}`} onClick={() => setTab("policies")}>Access policies</button>
          <button className={`tab ${tab === "audit" ? "active" : ""}`} onClick={() => setTab("audit")}>Audit log</button>
        </div>
        {selected && tab === "schema" && <SchemaTab key={selected.id} datasetId={selected.id} />}
        {selected && tab === "policies" && <PoliciesTab key={selected.id} datasetId={selected.id} />}
        {selected && tab === "audit" && <AuditTab key={selected.id} datasetId={selected.id} />}
      </div>
    </Shell>
  );
}

// ---------------------------------------------------------------- schema

function SchemaTab({ datasetId }: { datasetId: string }) {
  const [tables, setTables] = useState<SchemaTable[]>([]);
  const [embedder, setEmbedder] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.schema(datasetId);
      setTables(data.tables);
      setEmbedder(data.embedder);
      setSelected((s) => s ?? data.tables[0]?.name ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load schema");
    }
  }, [datasetId]);

  useEffect(() => {
    load();
  }, [load]);

  const table = tables.find((t) => t.name === selected);

  const save = async (column: string, description: string, previous: string) => {
    if (!table || description === previous) return;
    try {
      await api.annotate(datasetId, table.name, column, description);
      setTables((ts) =>
        ts.map((t) =>
          t.name !== table.name
            ? t
            : column === ""
              ? { ...t, description }
              : { ...t, columns: t.columns.map((c) => (c.name === column ? { ...c, description } : c)) },
        ),
      );
      setDirty(true);
      setStatus(`Saved description for ${table.name}${column ? "." + column : ""}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  const reindex = async () => {
    setStatus("Re-indexing…");
    try {
      const r = await api.reindex(datasetId);
      setDirty(false);
      setStatus(`Re-indexed ${r.chunks} schema chunks with ${r.embedder}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Re-index failed");
    }
  };

  return (
    <>
      <div className="row" style={{ marginBottom: 14, flexWrap: "wrap" }}>
        <p className="muted" style={{ margin: 0, flex: "1 1 400px" }}>
          Business descriptions are embedded with the schema, so better annotations mean better retrieval by the
          Schema Reasoning Agent. Re-index after editing.
        </p>
        {dirty && <span className="badge badge-warning">Index out of date</span>}
        <button className="btn btn-primary" onClick={reindex}>Re-index schema</button>
      </div>
      {status && <div className="notice notice-info" style={{ margin: "0 0 12px" }}>{status}</div>}
      {error && <div className="notice notice-danger" style={{ margin: "0 0 12px" }}>{error}</div>}

      <div className="schema-grid">
        <div className="card table-list">
          {tables.map((t) => (
            <button key={t.name} className={t.name === selected ? "active" : ""} onClick={() => setSelected(t.name)}>
              <span className="mono">{t.name}</span>
              <span className="subtle" style={{ fontSize: 12 }}>{t.columns.length} cols</span>
            </button>
          ))}
          <div className="subtle" style={{ fontSize: 12, padding: "8px 10px" }}>Embedder: {embedder}</div>
        </div>

        {table && (
          <div className="card" key={table.name}>
            <div className="section-head">
              <h3 className="mono">{table.name}</h3>
              <span className="subtle" style={{ fontSize: 12.5 }}>
                {table.indexed_at ? `indexed ${new Date(table.indexed_at * 1000).toLocaleString()}` : "not indexed"}
              </span>
            </div>
            <div style={{ padding: 16 }}>
              <label className="label">Table description</label>
              <textarea
                className="textarea"
                rows={2}
                defaultValue={table.description}
                onBlur={(e) => save("", e.target.value, table.description)}
              />
              <div className="table-wrap" style={{ marginTop: 14, maxHeight: "none" }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Type</th>
                      <th>Role</th>
                      <th>Keys</th>
                      <th style={{ width: "50%" }}>Business description (click to edit)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {table.columns.map((c) => {
                      const fk = table.foreign_keys.find((f) => f.column === c.name);
                      return (
                        <tr key={c.name}>
                          <td className="mono">{c.name}</td>
                          <td className="subtle">{c.type}</td>
                          <td>
                            <span className="badge">{c.role}</span>
                            {c.nulls > 0 && <div className="subtle" style={{ fontSize: 11.5 }}>{c.nulls} missing</div>}
                          </td>
                          <td>
                            {c.primary_key && <span className="badge badge-primary">PK</span>}{" "}
                            {fk && <span className="badge badge-info">FK → {fk.references}</span>}
                          </td>
                          <td style={{ whiteSpace: "normal" }}>
                            <input
                              className="annot-input"
                              defaultValue={c.description}
                              placeholder="Add a description…"
                              onBlur={(e) => save(c.name, e.target.value, c.description)}
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {table.chunk && (
                <details className="more" style={{ marginTop: 12 }}>
                  <summary>Embedded chunk (what retrieval sees)</summary>
                  <pre className="code" style={{ marginTop: 8, whiteSpace: "pre-wrap" }}>{table.chunk}</pre>
                </details>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

// ---------------------------------------------------------------- policies

function PoliciesTab({ datasetId }: { datasetId: string }) {
  const [tables, setTables] = useState<SchemaTable[]>([]);
  const [denied, setDenied] = useState<Set<string>>(new Set());
  const [saved, setSaved] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.schema(datasetId), api.policies(datasetId)])
      .then(([schema, policies]) => {
        setTables(schema.tables);
        const set = new Set((policies.roles.insights_user ?? []).map((d) => `${d.table}.${d.column}`));
        setDenied(set);
        setSaved(new Set(set));
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load policies"));
  }, [datasetId]);

  const toggle = (key: string) =>
    setDenied((s) => {
      const next = new Set(s);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const changed = useMemo(() => denied.size !== saved.size || Array.from(denied).some((k) => !saved.has(k)), [denied, saved]);

  const saveAll = async () => {
    try {
      const list = Array.from(denied).map((k) => {
        const [table, column] = k.split(".");
        return { table, column };
      });
      await api.setPolicies(datasetId, "insights_user", list);
      setSaved(new Set(denied));
      setStatus("Access policy saved. It applies to the next query.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    }
  };

  return (
    <>
      <div className="row" style={{ marginBottom: 14, flexWrap: "wrap" }}>
        <p className="muted" style={{ margin: 0, flex: "1 1 400px" }}>
          Restricted columns are hidden from the SQL generator for the <strong>Insights User</strong> role, and any SQL that
          references them is rejected by the schema check. Administrators are unrestricted.
        </p>
        <button className="btn btn-primary" onClick={saveAll} disabled={!changed}>Save policy</button>
      </div>
      {status && <div className="notice notice-info" style={{ margin: "0 0 12px" }}>{status}</div>}
      {error && <div className="notice notice-danger" style={{ margin: "0 0 12px" }}>{error}</div>}
      <div className="stack" style={{ gap: 14 }}>
        {tables.map((t) => {
          const whole = denied.has(`${t.name}.*`);
          return (
            <div className="card" key={t.name}>
              <div className="section-head">
                <h3 className="mono">{t.name}</h3>
                <div className="spacer" />
                <label className="row" style={{ cursor: "pointer", fontSize: 13 }}>
                  <input type="checkbox" checked={whole} onChange={() => toggle(`${t.name}.*`)} style={{ accentColor: "var(--danger)" }} />
                  Restrict entire table
                </label>
              </div>
              <div style={{ padding: "10px 16px", display: "flex", flexWrap: "wrap", gap: 8 }}>
                {t.columns.map((c) => {
                  const key = `${t.name}.${c.name}`;
                  const on = whole || denied.has(key);
                  return (
                    <label
                      key={key}
                      className={`badge ${on ? "badge-danger" : ""}`}
                      style={{ cursor: whole ? "not-allowed" : "pointer", height: 28, padding: "0 10px" }}
                    >
                      <input type="checkbox" checked={on} disabled={whole} onChange={() => toggle(key)} style={{ accentColor: "var(--danger)" }} />
                      <span className="mono">{c.name}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

// ---------------------------------------------------------------- audit

const STATUS_BADGE: Record<string, string> = {
  completed: "badge-success",
  blocked: "badge-danger",
  failed: "badge-danger",
};

function AuditTab({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<AuditResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "hallucinations" | "problems">("all");

  const load = useCallback(() => {
    api.audit(datasetId).then(setData).catch((e) => setError(e instanceof Error ? e.message : "Failed to load audit log"));
  }, [datasetId]);
  useEffect(load, [load]);

  if (error) return <div className="notice notice-danger" style={{ margin: 0 }}>{error}</div>;
  if (!data) return <div className="row muted"><span className="spinner" /> Loading…</div>;

  const s = data.stats;
  const entries = data.entries.filter((e) =>
    filter === "all" ? true : filter === "hallucinations" ? e.hallucinations.length > 0 : e.status !== "completed",
  );

  return (
    <>
      <div className="stats">
        <Stat label="Queries logged" value={s.total} />
        <Stat label="Completed" value={s.completed} />
        <Stat label="Hallucinations caught" value={s.hallucinations_detected} tone="warning" />
        <Stat label="Repaired & answered" value={s.repaired} tone="success" />
        <Stat label="Blocked by policy" value={s.blocked} tone="danger" />
        <Stat label="p80 latency" value={s.p80_latency_ms !== null ? `${s.p80_latency_ms} ms` : "—"} />
      </div>
      <div className="card">
        <div className="section-head">
          <h3>Executed and attempted queries</h3>
          <div className="spacer" />
          <select className="select" style={{ width: 200 }} value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
            <option value="all">All entries</option>
            <option value="hallucinations">With hallucinations</option>
            <option value="problems">Blocked or failed</option>
          </select>
          <button className="btn btn-sm" onClick={load}>Refresh</button>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="data">
            <thead>
              <tr>
                <th>Time</th>
                <th>User</th>
                <th>Question &amp; SQL</th>
                <th>Status</th>
                <th>Hallucinations</th>
                <th style={{ textAlign: "right" }}>Confidence</th>
                <th style={{ textAlign: "right" }}>Rows</th>
                <th style={{ textAlign: "right" }}>Latency</th>
              </tr>
            </thead>
            <tbody>
              {entries.length === 0 && (
                <tr><td colSpan={8} className="muted" style={{ textAlign: "center", padding: 24 }}>No entries yet — ask a question first.</td></tr>
              )}
              {entries.map((e) => (
                <tr key={e.id} style={{ verticalAlign: "top" }}>
                  <td className="subtle">{new Date(e.ts * 1000).toLocaleString()}</td>
                  <td>{e.username}</td>
                  <td style={{ whiteSpace: "normal", minWidth: 280 }}>
                    {e.question}
                    {e.final_sql && (
                      <details className="more">
                        <summary>SQL</summary>
                        <pre className="code" style={{ marginTop: 6 }}>{e.final_sql}</pre>
                      </details>
                    )}
                  </td>
                  <td><span className={`badge ${STATUS_BADGE[e.status] ?? ""}`}>{e.status}</span></td>
                  <td style={{ whiteSpace: "normal" }}>
                    {e.hallucinations.length === 0 ? (
                      <span className="subtle">none</span>
                    ) : (
                      e.hallucinations.map((h) => (
                        <div key={h.attempt} style={{ fontSize: 12.5 }}>
                          attempt {h.attempt}: {h.types.join(", ")}
                        </div>
                      ))
                    )}
                  </td>
                  <td className="num">{e.confidence ?? "—"}</td>
                  <td className="num">{e.row_count ?? "—"}</td>
                  <td className="num">{e.latency_ms} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function Stat({ label, value, tone }: { label: string; value: number | string; tone?: "success" | "warning" | "danger" }) {
  const color = tone ? `var(--${tone})` : undefined;
  return (
    <div className="card stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color }}>{value}</div>
    </div>
  );
}
