import type { Attempt, PipelineResponse, ValidationReport } from "@/lib/types";

const CHECKS: { key: keyof ValidationReport["checks"]; title: string; hint: string }[] = [
  { key: "schema", title: "Schema validation", hint: "Tables & columns exist, access permitted" },
  { key: "semantic", title: "Semantic verification", hint: "SQL answers the stated intent" },
  { key: "structural", title: "Structural validation", hint: "Joins follow real foreign keys" },
  { key: "ml_classifier", title: "ML hallucination detection", hint: "Learned classifier independently screens the draft SQL" },
];

function CheckCard({ report, name }: { report: ValidationReport; name: (typeof CHECKS)[number] }) {
  const check = report.checks[name.key];
  const skipped = "skipped" in check && check.skipped;
  const state = skipped ? "skip" : check.passed ? "pass" : "fail";
  return (
    <div className={`check ${state}`}>
      <h4>
        <span style={{ color: state === "pass" ? "var(--success)" : state === "fail" ? "var(--danger)" : "var(--text-3)" }}>
          {state === "pass" ? "✓" : state === "fail" ? "✕" : "–"}
        </span>
        {name.title}
        {report.timings_ms[name.key] !== undefined && (
          <span className="subtle" style={{ fontWeight: 400, marginLeft: "auto", fontSize: 12 }}>{report.timings_ms[name.key]} ms</span>
        )}
      </h4>
      <div className="subtle" style={{ fontSize: 12 }}>{skipped ? "Skipped — SQL did not parse" : name.hint}</div>
      {check.errors.length > 0 && (
        <ul>
          {check.errors.map((e, i) => (
            <li key={i} className="error-text">{e.message}</li>
          ))}
        </ul>
      )}
      {name.key === "ml_classifier" && !skipped && (
        <div className="subtle" style={{ fontSize: 12, marginTop: 8 }}>
          Prediction: <strong>{report.checks.ml_classifier.prediction}</strong>
          {report.checks.ml_classifier.confidence !== undefined &&
            <> · confidence {(report.checks.ml_classifier.confidence * 100).toFixed(1)}% (threshold {(report.checks.ml_classifier.threshold ?? 0) * 100}%)</>}
          {report.checks.ml_classifier.informational_only && <> · informational because a deterministic check already failed</>}
        </div>
      )}
      {name.key === "structural" && report.checks.structural.joins.length > 0 && check.errors.length === 0 && (
        <ul>
          {report.checks.structural.joins.map((j, i) => (
            <li key={i}><code>{j.condition}</code></li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AttemptBlock({ attempt, isFinal, injected }: { attempt: Attempt; isFinal: boolean; injected: PipelineResponse["injected"] }) {
  const v = attempt.validation;
  return (
    <div className="attempt">
      <div className="attempt-head">
        <strong>Attempt {attempt.attempt}</strong>
        <span className="badge">{attempt.attempt === 1 ? attempt.source : "after repair"}</span>
        {v && (v.passed ? <span className="badge badge-success">Verified</span> : <span className="badge badge-danger">Rejected: {v.hallucination_types.join(", ")}</span>)}
        {isFinal && v?.passed && <span className="badge badge-primary">Executed</span>}
      </div>
      {attempt.attempt === 1 && injected?.type && (
        <div className="notice notice-warning" style={{ margin: "0 0 10px" }}>
          Demo mode injected a {injected.type} hallucination: {injected.description}
        </div>
      )}
      <pre className="code">{attempt.sql}</pre>
      {v && (
        <>
          <div className="check-grid" style={{ marginTop: 10 }}>
            {CHECKS.map((c) => (
              <CheckCard key={c.key} report={v} name={c} />
            ))}
          </div>
          {v.checks.semantic.questions.length > 0 && (
            <details className="more">
              <summary>Semantic yes/no questions ({v.checks.semantic.questions.filter((q) => q.answer).length}/{v.checks.semantic.questions.length} yes)</summary>
              <div className="qa">
                {v.checks.semantic.questions.map((q, i) => (
                  <div key={i} style={{ display: "contents" }}>
                    <span className={q.answer ? "yes" : "no"}>{q.answer ? "✓" : "✕"}</span>
                    <span>
                      {q.question} <span className="subtle">— {q.detail}</span>
                      {q.source !== "intent-derived check" && <span className="badge badge-info" style={{ marginLeft: 6 }}>{q.source}</span>}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </>
      )}
      {attempt.repair && (
        <div className="repair-box">
          <strong>Query Repair Agent:</strong> {attempt.repair.strategies.join(" → ")}
          <ul>
            {attempt.repair.changes.map((c, i) => (
              <li key={i}>{c}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function VerificationPanel({ response }: { response: PipelineResponse }) {
  if (response.attempts.length === 0) {
    return <p className="muted">No SQL was generated for this question{response.error ? `: ${response.error}` : "."}</p>;
  }
  const resultChecks = response.result_checks?.checks ?? [];
  return (
    <div>
      <h4 className="section-title">1 · Checking the SQL before it runs</h4>
      {response.attempts.map((a, i) => (
        <AttemptBlock key={a.attempt} attempt={a} isFinal={i === response.attempts.length - 1} injected={response.injected} />
      ))}
      {resultChecks.length > 0 && (
        <>
          <h4 className="section-title">2 · Checking the answer after it runs</h4>
          <p className="subtle" style={{ margin: "0 0 8px", fontSize: 13 }}>
            Independent probe queries test properties any correct answer must satisfy.
          </p>
          <div className="result-check-list">
            {resultChecks.map((c) => (
              <div key={c.name} className={`check ${!c.applicable ? "skip" : c.passed ? "pass" : "fail"}`}>
                <h4>
                  <span style={{ color: !c.applicable ? "var(--text-3)" : c.passed ? "var(--success)" : "var(--danger)" }}>
                    {!c.applicable ? "–" : c.passed ? "✓" : "✕"}
                  </span>
                  {c.title}
                </h4>
                <div className="subtle" style={{ fontSize: 12.5 }}>{c.detail}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
