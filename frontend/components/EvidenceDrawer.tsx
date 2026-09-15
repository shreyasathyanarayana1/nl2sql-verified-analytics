"use client";

import { useEffect } from "react";
import type { Evidence } from "@/lib/types";
import { DataTable, downloadCsv } from "./ResultView";

export interface EvidenceState {
  title: string;
  loading: boolean;
  error: string | null;
  data: Evidence | null;
}

export default function EvidenceDrawer({ state, onClose }: { state: EvidenceState | null; onClose: () => void }) {
  useEffect(() => {
    if (!state) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state, onClose]);

  if (!state) return null;
  const { data } = state;

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label="Rows behind this number" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <div style={{ minWidth: 0 }}>
            <div className="eyebrow">Evidence</div>
            <h3>{state.title}</h3>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className="drawer-body">
          {state.loading && <div className="row muted"><span className="spinner" /> Fetching the underlying rows…</div>}
          {state.error && <div className="notice notice-danger" style={{ margin: 0 }}>{state.error}</div>}
          {data && (
            <>
              <p className="muted" style={{ marginTop: 0 }}>
                {data.row_count === 1 ? "This is the record" : `These are the ${data.row_count}${data.truncated ? "+" : ""} records`} that produced this number. They come from the same
                verified query, with the grouping removed, so nothing is re-guessed.
              </p>
              <DataTable columns={data.columns} rows={data.rows} labels={data.column_labels} maxHeight="calc(100vh - 290px)" />
              <div className="row" style={{ marginTop: 12, flexWrap: "wrap" }}>
                <span className="subtle" style={{ fontSize: 12.5 }}>
                  {data.row_count} row{data.row_count === 1 ? "" : "s"}{data.truncated ? " (first 500)" : ""} · {data.exec_ms} ms
                </span>
                <div className="spacer" />
                <button className="btn btn-sm" onClick={() => downloadCsv(data.columns, data.rows, state.title)}>Download CSV</button>
              </div>
              <details className="more" style={{ marginTop: 10 }}>
                <summary>Evidence query</summary>
                <pre className="code" style={{ marginTop: 8 }}>{data.sql}</pre>
              </details>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
