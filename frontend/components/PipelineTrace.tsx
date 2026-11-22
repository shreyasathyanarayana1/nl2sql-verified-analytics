import type { Ambiguity, Intent, PipelineResponse, SchemaContext, TraceStep } from "@/lib/types";

const STEP_NUMBER: Record<string, string> = {
  intent: "1", clarification: "2", schema: "3", access: "3b", planning: "4", generation: "5",
  verification: "6", repair: "7", execution: "8", result_checks: "8b", quality: "8c", visualization: "9", explanation: "10",
};

function IntentDetail({ intent }: { intent: Intent }) {
  const rows: [string, string][] = [
    ["Query type", intent.kind],
    ["Measures", intent.measures.map((m) => (m.agg ? `${m.agg} ${m.col ?? "rows"}` : m.col)).join(", ") || "—"],
    ["Group by", intent.group_by.map((g) => (g.bin ? `${g.col} (bands of ${g.bin})` : g.col)).join(", ") || "—"],
    ["Filters", intent.filters.map((f) => f.label).join("; ") || "—"],
    ["Ranking", intent.ranking ? `${intent.ranking.direction === "DESC" ? "highest" : "lowest"} ${intent.ranking.limit} ${intent.ranking.target}` : "—"],
    ["Unmatched words", intent.unknown_terms.join(", ") || "—"],
  ];
  return (
    <table className="data" style={{ marginTop: 6, maxWidth: 520 }}>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td className="muted" style={{ width: 120 }}>{k}</td>
            <td>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function AmbiguityDetail({ a }: { a: Ambiguity }) {
  return (
    <div style={{ marginTop: 6 }}>
      <div className="row" style={{ gap: 10 }}>
        <div className="score-bar" style={{ width: 160 }}>
          <span style={{ width: `${a.ambiguity_score * 100}%`, background: a.ambiguity_score > a.threshold ? "var(--warning)" : "var(--success)" }} />
        </div>
        <span className="subtle">score {a.ambiguity_score} · threshold {a.threshold}</span>
      </div>
      {a.signals.length > 0 ? (
        <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 13 }}>
          {a.signals.map((s) => (
            <li key={s.type}><code>{s.type}</code> (+{s.weight}) — {s.detail}</li>
          ))}
        </ul>
      ) : (
        <div className="subtle" style={{ fontSize: 13 }}>No ambiguity signals.</div>
      )}
    </div>
  );
}

function SchemaDetail({ ctx }: { ctx: SchemaContext }) {
  const max = Math.max(...ctx.candidates.map((c) => c.score), 0.01);
  const kept = new Set(ctx.tables);
  return (
    <div>
      <div className="subtle" style={{ fontSize: 12.5, marginTop: 6 }}>
        Embedder <code>{ctx.embedder}</code> · top-k = {ctx.k} · the SQL generator only sees the kept tables
      </div>
      <table className="data" style={{ marginTop: 6, maxWidth: 520 }}>
        <tbody>
          {ctx.candidates.map((c) => (
            <tr key={c.table}>
              <td style={{ width: 130 }}><code>{c.table}</code></td>
              <td>
                <div className="score-bar"><span style={{ width: `${(Math.max(c.score, 0) / max) * 100}%`, background: kept.has(c.table) ? "var(--primary)" : "var(--border-strong)" }} /></div>
              </td>
              <td className="num">{c.score.toFixed(3)}</td>
              <td>{kept.has(c.table) ? <span className="badge badge-primary">in context</span> : <span className="badge">excluded</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {ctx.actions.length > 0 && (
        <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 13 }}>
          {ctx.actions.map((a, i) => <li key={i}>{a}</li>)}
        </ul>
      )}
      {ctx.chunks.map((c) => (
        <div className="chunk" key={c.table}>
          <div className="row"><code>{c.table}</code><span className="subtle" style={{ fontSize: 12 }}>{c.reason}</span></div>
          <pre>{c.content}</pre>
        </div>
      ))}
    </div>
  );
}

function StepDetail({ step, response }: { step: TraceStep; response: PipelineResponse }) {
  switch (step.key) {
    case "intent":
      return response.intent ? <IntentDetail intent={response.intent} /> : null;
    case "clarification":
      return response.ambiguity ? <AmbiguityDetail a={response.ambiguity} /> : null;
    case "schema":
      return response.schema_context ? <SchemaDetail ctx={response.schema_context} /> : null;
    case "planning":
      return response.plan ? (
        <ol style={{ margin: "6px 0 0", paddingLeft: 20, fontSize: 13 }}>
          {response.plan.map((s, i) => <li key={i}>{s}</li>)}
        </ol>
      ) : null;
    default:
      return null;
  }
}

export default function PipelineTrace({ response }: { response: PipelineResponse }) {
  return (
    <ul className="trace">
      {response.trace.map((step, i) => {
        const hasDetail = ["intent", "clarification", "schema", "planning"].includes(step.key);
        return (
          <li key={i}>
            <div className={`step-icon ${step.status}`}>{STEP_NUMBER[step.key]}</div>
            <div style={{ minWidth: 0 }}>
              <div className="row" style={{ gap: 6 }}>
                <span className="step-name">{step.name}</span>
                {step.attempt && <span className="badge">attempt {step.attempt}</span>}
                {step.status === "failed" && <span className="badge badge-danger">failed</span>}
                {step.status === "waiting" && <span className="badge badge-warning">waiting for user</span>}
              </div>
              <div className="step-summary">{step.summary}</div>
              {hasDetail && (
                <details className="more">
                  <summary>Details</summary>
                  <StepDetail step={step} response={response} />
                </details>
              )}
            </div>
            <div className="step-time">{step.duration_ms} ms</div>
          </li>
        );
      })}
    </ul>
  );
}
