"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";
import { api } from "@/lib/api";

const PERSONAS = [
  { username: "analyst", password: "analyst123", title: "Insights User", hint: "Ask questions, view results" },
  { username: "admin", password: "admin123", title: "Administrator", hint: "Schema, policies, audit log" },
];

const AGENTS = [
  "Intent", "Clarification", "Schema RAG", "Planning", "SQL Generation",
  "Anti-Hallucination", "Repair", "Execution", "Visualization",
];

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("analyst");
  const [password, setPassword] = useState("analyst123");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await api.login(username, password);
      router.replace("/chat");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <section className="login-hero">
        <div>
          <div className="eyebrow">Verified Analytics</div>
        </div>
        <div>
          <h1>Ask the database in plain English. Get an answer you can check.</h1>
          <p>
            A generated query is never trusted on its own. Each draft is verified against the live schema, the stated
            intent and the real foreign keys, repaired if it fails, and the result is cross-checked after it runs.
          </p>
          <ol className="pipeline-mini">
            {AGENTS.map((a, i) => (
              <li key={a} className={a === "Anti-Hallucination" || a === "Repair" ? "hl" : ""}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                <span>{a}</span>
              </li>
            ))}
          </ol>
        </div>
        <p className="login-foot">B.M.S. College of Engineering · Dept. of CSE · Major Project</p>
      </section>

      <div className="login-form-wrap">
        <form className="card login-card" onSubmit={submit}>
          <h2>Sign in</h2>
          <p className="muted" style={{ margin: "4px 0 0" }}>Choose a demo persona or enter credentials.</p>

          <div className="persona">
            {PERSONAS.map((p) => (
              <button
                type="button"
                key={p.username}
                className={username === p.username ? "active" : ""}
                onClick={() => {
                  setUsername(p.username);
                  setPassword(p.password);
                }}
              >
                <strong>{p.title}</strong>
                <small>{p.hint}</small>
              </button>
            ))}
          </div>

          <div className="stack" style={{ gap: 12 }}>
            <div>
              <label className="label" htmlFor="username">Username</label>
              <input id="username" className="input" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" />
            </div>
            <div>
              <label className="label" htmlFor="password">Password</label>
              <input id="password" type="password" className="input" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" />
            </div>
            {error && <div className="notice notice-danger" style={{ margin: 0 }}>{error}</div>}
            <button className="btn btn-primary" type="submit" disabled={loading} style={{ height: 40 }}>
              {loading ? <span className="spinner" /> : "Sign in"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
