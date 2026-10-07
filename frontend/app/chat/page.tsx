"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnswerCard from "@/components/AnswerCard";
import { LiveAgents } from "@/components/AgentRun";
import EvidenceDrawer, { EvidenceState } from "@/components/EvidenceDrawer";
import Shell from "@/components/Shell";
import { api } from "@/lib/api";
import type { PipelineResponse, SuggestionGroup } from "@/lib/types";
import { useAuth } from "@/lib/useAuth";
import { useDatasets } from "@/lib/useDatasets";

interface Turn {
  id: number;
  question: string;
  isReply: boolean;
  at: Date;
  response: PipelineResponse | null;
  error: string | null;
}

const entryNo = (n: number) => `Q.${String(n).padStart(2, "0")}`;
const clock = (d: Date) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export default function ChatPage() {
  const user = useAuth();
  const { datasets, selected: dataset, select: selectDataset, error: datasetError } = useDatasets(Boolean(user));
  const [turns, setTurns] = useState<Turn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [suggestions, setSuggestions] = useState<SuggestionGroup[]>([]);
  const [evidence, setEvidence] = useState<EvidenceState | null>(null);
  const [aiStatus, setAiStatus] = useState<{ mode: "local"; reason: string | null } | null>(null);
  const nextId = useRef(1);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!user) return;
    api.health().then((h) => setAiStatus(h.llm_status)).catch(() => setAiStatus(null));
  }, [user]);

  useEffect(() => {
    if (!dataset) return;
    setSuggestions([]);
    api.suggestions(dataset.id).then(setSuggestions).catch(() => setSuggestions([]));
  }, [dataset]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const run = useCallback(async (question: string, call: () => Promise<PipelineResponse>, isReply = false) => {
    const id = nextId.current++;
    setTurns((t) => [...t, { id, question, isReply, at: new Date(), response: null, error: null }]);
    setBusy(true);
    try {
      const response = await call();
      setSessionId(response.session_id);
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (e) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: e instanceof Error ? e.message : "Request failed" } : x)));
    } finally {
      setBusy(false);
    }
  }, []);

  const ask = (question: string) => {
    const q = question.trim();
    if (!q || busy || !dataset) return;
    setInput("");
    run(q, () => api.query(q, dataset.id, sessionId));
  };

  const clarify = (optionId: string | null, text: string | null, label: string) => {
    if (!sessionId || busy) return;
    run(label, () => api.clarify(sessionId, optionId, text), true);
  };

  const openEvidence = useCallback(async (response: PipelineResponse, key: string | number | null, title: string) => {
    setEvidence({ title, loading: true, error: null, data: null });
    try {
      const data = await api.evidence(response.session_id, response.response_id, key);
      setEvidence({ title, loading: false, error: null, data });
    } catch (e) {
      setEvidence({ title, loading: false, error: e instanceof Error ? e.message : "Could not load rows", data: null });
    }
  }, []);

  const newConversation = () => {
    setTurns([]);
    setSessionId(null);
    inputRef.current?.focus();
  };

  const switchDataset = (id: string) => {
    if (id === dataset?.id) return;
    selectDataset(id);
    setTurns([]);
    setSessionId(null);
  };

  if (!user) return null;

  const recent = Array.from(new Set(turns.filter((t) => t.response?.status === "completed").map((t) => t.question))).slice(-6).reverse();

  const sidebar = (
    <>
      <div className="sidebar-section">Dataset</div>
      <div className="dataset-picker">
        {datasets.map((d) => (
          <button key={d.id} className={d.id === dataset?.id ? "active" : ""} onClick={() => switchDataset(d.id)} disabled={busy}>
            <strong>{d.name}</strong>
            <small>{d.rows.toLocaleString()} {d.entity.plural} · {d.columns} columns</small>
          </button>
        ))}
      </div>
      {recent.length > 0 && (
        <>
          <div className="sidebar-section">Recent</div>
          <div className="sidebar-body">
            {recent.map((q) => (
              <button key={q} className="suggestion" onClick={() => ask(q)} disabled={busy}>{q}</button>
            ))}
          </div>
        </>
      )}
      <div className="sidebar-section">Explore</div>
      <div className="sidebar-body">
        {suggestions.map((g) => (
          <details key={g.group} className="topic" open={g === suggestions[0]}>
            <summary>{g.group}</summary>
            {g.questions.map((q) => (
              <button key={q} className="suggestion" onClick={() => ask(q)} disabled={busy}>{q}</button>
            ))}
          </details>
        ))}
      </div>
    </>
  );

  const lastResponseIdx = (() => {
    for (let i = turns.length - 1; i >= 0; i--) if (turns[i].response) return i;
    return -1;
  })();

  return (
    <Shell
      user={user}
      title={dataset ? dataset.name : "Ask your data"}
      sidebar={sidebar}
      actions={
        <>
          {aiStatus && (
            <span className="badge hide-mobile badge-success" title={aiStatus.reason ?? "Local ML pipeline is active"}>
              <span className="dot" />local models
            </span>
          )}
          <button className="btn btn-sm" onClick={newConversation} disabled={busy || turns.length === 0}>New session</button>
        </>
      }
    >
      <div className="chat">
        <div className="messages">
          {turns.length === 0 && (
            <div className="welcome">
              {datasetError && <div className="notice notice-danger">{datasetError}</div>}
              {dataset && (
                <div className="dataset-card">
                  <div className="eyebrow">Dataset · {dataset.source}</div>
                  <h2>{dataset.name}</h2>
                  <p className="muted" style={{ margin: 0 }}>{dataset.description}</p>
                  <div className="dataset-stats">
                    <div><span>{dataset.rows.toLocaleString()}</span>{dataset.entity.plural}</div>
                    <div><span>{dataset.columns}</span>columns</div>
                    <div><span>{dataset.tables.length}</span>{dataset.tables.length === 1 ? "table" : "tables"}</div>
                  </div>
                </div>
              )}
              <p className="muted">
                Ask in plain English. Each question is turned into SQL, checked against the real schema before it runs,
                and cross-checked after. Every answer is stamped with how well it held up, and links back to the rows
                that produced it.
              </p>
              <div className="topic-grid">
                {suggestions.filter((g) => !g.group.startsWith("Needs")).map((g) => (
                  <div key={g.group} className="topic-card">
                    <div className="topic-head">{g.group}</div>
                    {g.questions.slice(0, 3).map((q) => (
                      <button key={q} onClick={() => ask(q)}>{q}</button>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <article key={t.id} className="entry">
              <header className="entry-head">
                <span className="entry-no">{entryNo(i + 1)}</span>
                <h2 className={`entry-q${t.isReply ? " reply" : ""}`}>{t.question}</h2>
                <time className="entry-time" dateTime={t.at.toISOString()}>{clock(t.at)}</time>
              </header>
              {!t.response && !t.error && <LiveAgents />}
              {t.error && <div className="notice notice-danger">{t.error}</div>}
              {t.response && (
                <AnswerCard
                  response={t.response}
                  onClarify={clarify}
                  clarifyDisabled={busy || i !== lastResponseIdx}
                  onAsk={ask}
                  onEvidence={openEvidence}
                  busy={busy}
                />
              )}
            </article>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="composer-wrap">
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
          >
            <span className="composer-no" aria-hidden>{entryNo(turns.length + 1)}</span>
            <textarea
              ref={inputRef}
              rows={1}
              placeholder={dataset ? `Ask about ${dataset.entity.plural}, e.g. “${suggestions[1]?.questions[0] ?? "What factors matter most?"}”` : "Loading datasets…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              maxLength={500}
              aria-label="Your question"
            />
            <button className="btn btn-primary" type="submit" disabled={busy || !input.trim()}>
              {busy ? <span className="spinner spinner-light" /> : "Ask"}
            </button>
          </form>
          <div className="composer-hint">
            <span>↵ run · ⇧↵ new line</span>
            <span>follow-ups refine the previous entry: “only for …”, “what about …”</span>
          </div>
        </div>
      </div>
      <EvidenceDrawer state={evidence} onClose={() => setEvidence(null)} />
    </Shell>
  );
}
