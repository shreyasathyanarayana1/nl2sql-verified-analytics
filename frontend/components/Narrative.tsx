import type { NarrativeSection } from "@/lib/types";

function Section({ s }: { s: NarrativeSection }) {
  const before = s.paragraphs.slice(0, s.bullets_after);
  const after = s.paragraphs.slice(s.bullets_after);
  return (
    <section className={`narrative-section ns-${s.key}`}>
      <h3>{s.title}</h3>
      {before.map((p, i) => <p key={`b${i}`}>{p}</p>)}
      {s.bullets.length > 0 && (
        <ul className={s.key === "method" ? "numbered" : undefined}>
          {s.bullets.map((b, i) => <li key={i}>{b}</li>)}
        </ul>
      )}
      {after.map((p, i) => <p key={`a${i}`}>{p}</p>)}
    </section>
  );
}

/** The long-form explanation written by the Explanation agent. `hideFindings` keeps the numbers back when the answer is withheld. */
export default function Narrative({ sections, hideFindings }: { sections: NarrativeSection[]; hideFindings?: boolean }) {
  const shown = hideFindings ? sections.filter((s) => s.key !== "findings") : sections;
  if (shown.length === 0) return null;
  return (
    <div className="narrative">
      {shown.map((s) => <Section key={s.key} s={s} />)}
    </div>
  );
}
