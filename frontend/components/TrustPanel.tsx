import type { Confidence, PipelineResponse } from "@/lib/types";

const LEVEL_CLASS = { high: "trust-high", medium: "trust-medium", low: "trust-low" } as const;

export function TrustBadge({ confidence, onClick, open }: { confidence: Confidence; onClick: () => void; open: boolean }) {
  const icon = confidence.level === "high" ? "✓" : confidence.level === "medium" ? "!" : "✕";
  return (
    <button className={`trust-badge ${LEVEL_CLASS[confidence.level]}`} onClick={onClick} aria-expanded={open}>
      <span className="trust-icon">{icon}</span>
      <span>{confidence.label}</span>
      <span className="trust-score">{confidence.score}</span>
      <span className="trust-caret">{open ? "▴" : "▾"}</span>
    </button>
  );
}

export function TrustPanel({ response }: { response: PipelineResponse }) {
  const c = response.confidence;
  if (!c) return null;
  const sqlAttempts = response.attempts.length;
  const firstPassed = response.attempts[0]?.validation?.passed;
  const resultChecks = (response.result_checks?.checks ?? []).filter((x) => x.applicable);
  const negatives = c.reasons.filter((r) => r.impact < 0);

  return (
    <div className="trust-panel">
      <div className="trust-meter">
        <div className="trust-meter-bar"><span className={LEVEL_CLASS[c.level]} style={{ width: `${c.score}%` }} /></div>
        <span className="subtle" style={{ fontSize: 12.5 }}>{c.score}/100</span>
      </div>

      <div className="trust-grid">
        <section>
          <h4>Query checks</h4>
          <ul className="check-list">
            <li className={firstPassed ? "ok" : "warn"}>
              {firstPassed
                ? "Tables, columns, meaning and joins all verified on the first draft"
                : `Mistakes were caught in the draft SQL and fixed (${sqlAttempts - 1} repair${sqlAttempts > 2 ? "s" : ""})`}
            </li>
            <li className="ok">Ran read-only against the live database</li>
          </ul>
        </section>
        <section>
          <h4>Answer checks</h4>
          {resultChecks.length === 0 ? (
            <p className="subtle" style={{ margin: 0, fontSize: 13 }}>No cross-checks apply to this kind of answer.</p>
          ) : (
            <ul className="check-list">
              {resultChecks.map((rc) => (
                <li key={rc.name} className={rc.passed ? "ok" : "bad"} title={rc.detail}>
                  {rc.title}
                  <span className="subtle"> — {rc.detail}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      {negatives.length > 0 && (
        <section style={{ marginTop: 12 }}>
          <h4>What lowered the score</h4>
          <ul className="reason-list">
            {negatives.map((r, i) => (
              <li key={i}><span className="impact">{r.impact}</span>{r.text}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
