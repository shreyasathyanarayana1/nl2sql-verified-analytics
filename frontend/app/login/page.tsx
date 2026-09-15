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
        <div className="row">
          <div className="brand-mark">NL</div>
          <strong style={{ color: "#fff" }}>Verified Analytics</strong>
        </div>
        <div>
          <h1>Ask your database in plain English — and trust the answer.</h1>
          <p>
            Every generated SQL query is grounded in retrieved schema, checked for schema, semantic and join
            hallucinations, and repaired before it ever touches the database.
          </p>
          <div className="pipeline-mini">
            {AGENTS.map((a, i) => (
              <span key={a} className={a === "Anti-Hallucination" || a === "Repair" ? "hl" : ""}>
                {i + 1}. {a}
              </span>
            ))}
          </div>
        </div>
        <p style={{ fontSize: 12 }}>B.M.S. College of Engineering · Dept. of CSE · Major Project</p>
      </section>

      <div className="login-form-wrap">
        <form className="card login-card" onSubmit={submit}>
          <h2 style={{ fontSize: 20 }}>Sign in</h2>
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
