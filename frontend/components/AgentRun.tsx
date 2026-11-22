"use client";

import { useEffect, useRef, useState } from "react";
import type { PipelineResponse, TraceStep } from "@/lib/types";

/** Every agent in pipeline order, with the model or technique it runs. `ml` marks a trained model. */
export const AGENTS: { key: string; no: string; name: string; model: string; ml: boolean }[] = [
  { key: "intent", no: "01", name: "Intent Understanding", model: "rule parser + TF-IDF / logistic-regression intent classifier", ml: true },
  { key: "clarification", no: "02", name: "Query Clarification", model: "ambiguity scorer + clarification classifier", ml: true },
  { key: "schema", no: "03", name: "Schema Reasoning (RAG)", model: "TF-IDF schema embedder, top-k retrieval", ml: true },
  { key: "access", no: "3b", name: "RBAC Access Control", model: "role-based column policy", ml: false },
  { key: "planning", no: "04", name: "Query Planning", model: "semantic-layer planner", ml: false },
  { key: "generation", no: "05", name: "SQL Generation", model: "nearest-neighbour SQL template retriever + renderer", ml: true },
  { key: "verification", no: "06", name: "Anti-Hallucination", model: "schema · semantic · join checks + ML hallucination classifier", ml: true },
  { key: "repair", no: "07", name: "Query Repair", model: "repair-type classifier + SQL AST patcher", ml: true },
  { key: "execution", no: "08", name: "Execution", model: "SQLite, read-only connection", ml: false },
  { key: "result_checks", no: "8b", name: "Result Verification", model: "independent probe queries", ml: false },
  { key: "quality", no: "8c", name: "Data Quality & Confidence", model: "data-quality rules + confidence regressor", ml: true },
  { key: "visualization", no: "09", name: "Visualization", model: "chart selector", ml: false },
  { key: "explanation", no: "10", name: "Explanation", model: "explanation writer — every figure computed from the result", ml: false },
];
const BY_KEY = Object.fromEntries(AGENTS.map((a) => [a.key, a]));

const pct = (v: number | undefined) => (v === undefined ? "" : `${Math.round(v * 100)}%`);

/** What the agent's model concluded for this particular question, when the response carries it. */
function modelOutput(step: TraceStep, r: PipelineResponse): string | null {
  switch (step.key) {
    case "intent": {
      const m = r.intent?.ml_intent;
      return m?.enabled ? `classifier → “${m.prediction}” at ${pct(m.confidence)}` : null;
    }
    case "clarification": {
      const m = r.ambiguity?.ml_classifier;
      if (!m?.enabled) return null;
      return `classifier → ${Number(m.prediction) === 1 ? "needs clarification" : "clear question"} at ${pct(m.confidence)}`;
    }
    case "schema":
      return r.schema_context ? `${r.schema_context.embedder} · kept ${r.schema_context.tables.length} of ${r.schema_context.total_tables} tables` : null;
    case "generation": {
      const t = (step.detail as { template?: { enabled?: boolean; template?: string; similarity?: number } } | null)?.template;
      return t?.enabled ? `nearest template “${t.template}” · similarity ${t.similarity?.toFixed(2)}` : null;
    }
    case "verification": {
      const v = r.attempts.find((a) => a.attempt === step.attempt)?.validation?.checks.ml_classifier;
      if (!v || v.skipped || v.confidence === undefined) return null;
      return `ML screen → “${v.prediction}” at ${pct(v.confidence)} (flags only above ${pct(v.threshold)})`;
    }
    case "quality": {
      const reason = r.confidence?.reasons.find((x) => x.text.startsWith("Local ML confidence model"));
      return reason ? `regressor → ${reason.text.split(": ")[1]}` : null;
    }
    default:
      return null;
  }
}

/** Shown while the request is in flight: every agent on a track, with a pulse moving along it. */
export function LiveAgents() {
  const [tick, setTick] = useState(0);
  const started = useRef(Date.now());
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 140);
    return () => clearInterval(t);
  }, []);
  const active = tick % AGENTS.length;
  const agent = AGENTS[active];
  return (
    <div className="live" aria-live="polite" aria-label="Agents working">
      <div className="live-head">
        <span className="eyebrow" style={{ margin: 0 }}>Agents working</span>
        <span className="live-clock">{((Date.now() - started.current) / 1000).toFixed(1)} s</span>
      </div>
      <ol className="track">
        {AGENTS.map((a, i) => (
          <li key={a.key} className={i === active ? "on" : i < active ? "past" : ""} title={a.name}>
            <span>{a.no}</span>
          </li>
        ))}
      </ol>
      <div className="live-now">
        <strong>{agent.name}</strong>
        <span className="live-model">{agent.model}</span>
      </div>
    </div>
  );
}

const REPLAY_MS = 120;

/** The recorded trace of this answer, replayed agent by agent, then kept as a compact record. */
export default function AgentRun({ response, onDone }: { response: PipelineResponse; onDone?: () => void }) {
  const steps = response.trace;
  const [shown, setShown] = useState(0);
  const [open, setOpen] = useState(true);
  const done = shown >= steps.length;
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useEffect(() => {
    if (shown >= steps.length) {
      doneRef.current?.();
      return;
    }
    const t = setTimeout(() => setShown((n) => n + 1), REPLAY_MS);
    return () => clearTimeout(t);
  }, [shown, steps.length]);

  const agentsUsed = new Set(steps.map((s) => s.key)).size;
  const modelsUsed = new Set(steps.filter((s) => BY_KEY[s.key]?.ml).map((s) => s.key)).size;

  return (
    <section className="agent-run">
      <header className="agent-run-head">
        <span className="eyebrow" style={{ margin: 0 }}>Agent run</span>
        <span className="agent-run-stats">
          {agentsUsed} agents · {modelsUsed} trained models · {(response.latency_ms / 1000).toFixed(2)} s
        </span>
        <span className="spacer" />
        {done && (
          <button className="link-btn" onClick={() => setOpen((o) => !o)}>{open ? "Collapse" : "Show every agent"}</button>
        )}
      </header>
      {open && (
        <ol className="agent-steps">
          {steps.slice(0, Math.min(shown + 1, steps.length)).map((s, i) => {
            const meta = BY_KEY[s.key];
            const running = i === shown;
            const out = modelOutput(s, response);
            return (
              <li key={i} className={`agent-step ${running ? "running" : s.status}`}>
                <span className="agent-no">{meta?.no ?? "··"}</span>
                <div className="agent-body">
                  <div className="agent-title">
                    <strong>{s.name}</strong>
                    {meta?.ml && <span className="ml-tag">ML</span>}
                    {s.attempt && <span className="subtle mono" style={{ fontSize: 11 }}>draft {s.attempt}</span>}
                  </div>
                  <div className="agent-model">{meta?.model}</div>
                  {!running && <div className="agent-summary">{s.summary}</div>}
                  {!running && out && <div className="agent-output">{out}</div>}
                </div>
                <span className="agent-ms">{running ? "…" : `${s.duration_ms} ms`}</span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
