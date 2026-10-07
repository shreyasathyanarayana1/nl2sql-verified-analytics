"use client";

import { useEffect, useRef, useState } from "react";
import type { PipelineResponse } from "@/lib/types";
import AgentRun from "./AgentRun";
import Narrative from "./Narrative";
import PipelineTrace from "./PipelineTrace";
import { DataTable, downloadCsv, fmt, ResultChart } from "./ResultView";
import { TrustBadge, TrustPanel } from "./TrustPanel";
import VerificationPanel from "./VerificationPanel";

type TechTab = "checks" | "pipeline";

interface Props {
  response: PipelineResponse;
  onClarify: (optionId: string | null, text: string | null, label: string) => void;
  clarifyDisabled: boolean;
  onAsk: (question: string) => void;
  onEvidence: (response: PipelineResponse, key: string | number | null, title: string) => void;
  busy: boolean;
}

const SEVERITY_TAG = { high: "Caution", medium: "Caveat", low: "Note" } as const;

function Clarification({ response, onClarify, disabled }: { response: PipelineResponse; onClarify: Props["onClarify"]; disabled: boolean }) {
  const c = response.clarification!;
  const [freeText, setFreeText] = useState("");
  const hasPreviews = c.options.some((o) => o.preview);
  return (
    <div className="card answer clarify">
      <div className="answer-body">
        <div className="eyebrow" style={{ color: "var(--warning)" }}>Ambiguous — clarification needed</div>
        <div className="clarify-q">{c.question}</div>
        {hasPreviews && (
          <p className="muted" style={{ margin: "4px 0 0", fontSize: 13 }}>
            Each reading was run against the data. They give different answers, so the choice matters:
          </p>
        )}
        <div className={hasPreviews ? "option-grid" : "chip-row"}>
          {c.options.map((o) =>
            hasPreviews ? (
              <button key={o.id} className="option-card" disabled={disabled} onClick={() => onClarify(o.id, null, o.label)}>
                <strong>{o.label}</strong>
                {o.preview && <span className="subtle">{o.preview}</span>}
              </button>
            ) : (
              <button key={o.id} className="chip" disabled={disabled} onClick={() => onClarify(o.id, null, o.label)}>
                {o.label}
              </button>
            ),
          )}
        </div>
        {!disabled && (
          <form
            className="row"
            style={{ marginTop: 12, maxWidth: 480 }}
            onSubmit={(e) => {
              e.preventDefault();
              if (freeText.trim()) onClarify(null, freeText.trim(), freeText.trim());
            }}
          >
            <input className="input" placeholder="…or answer in your own words" value={freeText} onChange={(e) => setFreeText(e.target.value)} />
            <button className="btn" type="submit" disabled={!freeText.trim()}>Answer</button>
          </form>
        )}
      </div>
    </div>
  );
}

function Technical({ response }: { response: PipelineResponse }) {
  const [tab, setTab] = useState<TechTab>("checks");
  return (
    <div className="technical">
      <div className="tabs">
        <button className={`tab ${tab === "checks" ? "active" : ""}`} onClick={() => setTab("checks")}>
          SQL drafts &amp; checks{response.attempts.length > 1 ? ` (${response.attempts.length} drafts)` : ""}
        </button>
        <button className={`tab ${tab === "pipeline" ? "active" : ""}`} onClick={() => setTab("pipeline")}>Pipeline log</button>
      </div>
      <div style={{ paddingTop: 14 }}>
        {tab === "checks" && <VerificationPanel response={response} />}
        {tab === "pipeline" && <PipelineTrace response={response} />}
      </div>
    </div>
  );
}

/** Replays the agent run first, then reveals the answer under it. */
export default function AnswerCard(props: Props) {
  const [revealed, setRevealed] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (revealed) ref.current?.closest(".entry")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [revealed]);
  return (
    <div ref={ref} className="answer-stack">
      <AgentRun response={props.response} onDone={() => setRevealed(true)} />
      {revealed && <div className="reveal"><AnswerBody {...props} /></div>}
    </div>
  );
}

function AnswerBody({ response: r, onClarify, clarifyDisabled, onAsk, onEvidence, busy }: Props) {
  const [trustOpen, setTrustOpen] = useState(false);
  const [showAnyway, setShowAnyway] = useState(false);
  const [tableOpen, setTableOpen] = useState(false);
  const [techOpen, setTechOpen] = useState(false);

  if (r.status === "needs_clarification" && r.clarification) {
    return <Clarification response={r} onClarify={onClarify} disabled={clarifyDisabled} />;
  }

  if (r.status !== "completed" || !r.result || !r.visualization) {
    return (
      <div className="card answer">
        <div className="answer-body">
          <div className="eyebrow" style={{ color: "var(--danger)" }}>{r.status === "blocked" ? "Not allowed" : "Couldn't answer this"}</div>
          <div className="answer-explanation">{r.error}</div>
          {r.warnings.map((w, i) => <div key={i} className="warn-card warn-medium"><span className="note-tag">Caveat</span><div>{w}</div></div>)}
          <div className="answer-actions">
            <button className="link-btn" onClick={() => setTechOpen((o) => !o)}>{techOpen ? "Hide details" : "What happened?"}</button>
          </div>
          {techOpen && <Technical response={r} />}
        </div>
      </div>
    );
  }

  const viz = r.visualization;
  const result = r.result;
  const c = r.confidence;
  const withheld = Boolean(c?.abstain) && !showAnyway;
  const keyIdx = viz.key_column ? result.columns.indexOf(viz.key_column) : -1;
  const labelIdx = viz.label_column ? result.columns.indexOf(viz.label_column) : -1;
  const canDrill = keyIdx >= 0 && !["correlation", "drivers", "list"].includes(r.intent?.kind ?? "");
  const hasChart = !["table", "empty"].includes(viz.chart_type);
  const showTable = !hasChart || tableOpen;
  const repairs = r.attempts.length - 1;

  const drill = (rowIndex: number) => {
    if (!canDrill) return;
    const row = result.rows[rowIndex];
    const label = labelIdx >= 0 ? row[labelIdx] : row[keyIdx];
    const title = viz.chart_type === "histogram" ? `Students in the ${viz.labels[rowIndex]} band` : `${fmt(label)} — ${viz.value_label ?? "details"}`;
    onEvidence(r, row[keyIdx], title);
  };

  const aggregateKinds = ["aggregate", "ranking", "count_having"];
  const canSeeAll = r.intent && aggregateKinds.includes(r.intent.kind) && viz.chart_type === "kpi";

  return (
    <div className="card answer">
      <div className="answer-body">
        <div className="answer-top">
          <div className="answer-explanation">{viz.explanation}</div>
          {c && <TrustBadge confidence={c} open={trustOpen} onClick={() => setTrustOpen((o) => !o)} />}
        </div>
        {trustOpen && <TrustPanel response={r} />}

        {r.auto_resolved && (
          <div className="info-line">
            <span className="note-tag">Resolved</span>
            <span>“{r.auto_resolved.original}” could mean several things — {r.auto_resolved.detail}.</span>
          </div>
        )}
        {repairs > 0 && !trustOpen && (
          <div className="info-line repaired">
            <span className="note-tag">Repaired</span>
            <span>
              Draft 1 of the SQL was rejected by the verifier and repaired before it ran.{" "}
              <button className="link-btn" onClick={() => setTechOpen(true)}>See what changed</button>
            </span>
          </div>
        )}

        {r.data_warnings.map((w, i) => (
          <div key={i} className={`warn-card warn-${w.severity}`}>
            <span className="note-tag">{SEVERITY_TAG[w.severity]}</span>
            <div><strong>{w.title}.</strong> {w.detail}</div>
          </div>
        ))}
        {r.warnings.map((w, i) => (
          <div key={`w${i}`} className="warn-card warn-medium"><span className="note-tag">Caveat</span><div>{w}</div></div>
        ))}

        {withheld ? (
          <div className="abstain">
            <strong>Withheld: confidence is below the threshold for showing this result by default.</strong>
            <ul>
              {c!.reasons.filter((x) => x.impact < 0).slice(0, 3).map((x, i) => <li key={i}>{x.text}</li>)}
            </ul>
            <div className="row" style={{ flexWrap: "wrap" }}>
              <button className="btn btn-sm" onClick={() => setShowAnyway(true)}>Show the result anyway</button>
              <span className="subtle" style={{ fontSize: 12.5 }}>or rephrase the question more specifically</span>
            </div>
          </div>
        ) : (
          <>
            {hasChart && (
              <div className="chart-wrap">
                <ResultChart viz={viz} result={result} onSelect={canDrill ? drill : undefined} />
                {canDrill && <div className="chart-hint">Click any {viz.chart_type === "pie" ? "slice" : viz.chart_type === "line" ? "point" : "bar"} to see the rows behind it</div>}
              </div>
            )}
            {showTable && result.row_count > 0 && (
              <DataTable columns={result.columns} rows={result.rows} labels={r.column_labels} onRowClick={canDrill ? drill : undefined} />
            )}
          </>
        )}

        {r.narrative && <Narrative sections={r.narrative.sections} hideFindings={withheld} />}

        {r.notes.length > 0 && (
          <div className="assumptions">
            {r.notes.map((n, i) => <span key={i}>{n}</span>)}
          </div>
        )}

        <div className="answer-actions">
          {hasChart && result.row_count > 0 && !withheld && (
            <button className="link-btn" onClick={() => setTableOpen((o) => !o)}>{tableOpen ? "Hide table" : `Show table (${result.row_count} rows)`}</button>
          )}
          {canSeeAll && <button className="link-btn" onClick={() => onEvidence(r, null, viz.explanation)}>See the rows behind this</button>}
          {result.row_count > 0 && !withheld && (
            <button className="link-btn" onClick={() => downloadCsv(result.columns, result.rows, r.question)}>Download CSV</button>
          )}
          <button className="link-btn" onClick={() => setTechOpen((o) => !o)}>{techOpen ? "Hide verification record" : "Verification record"}</button>
          <span className="spacer" />
          <span className="latency">{(r.latency_ms / 1000).toFixed(2)} s · {r.attempts.length} draft{r.attempts.length === 1 ? "" : "s"}</span>
        </div>
        {techOpen && <Technical response={r} />}

        {r.followups.length > 0 && (
          <div className="followups">
            <span className="eyebrow" style={{ margin: 0 }}>Next</span>
            {r.followups.map((q) => (
              <button key={q} className="chip chip-sm" disabled={busy} onClick={() => onAsk(q)}>{q}</button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
