import katex from "katex";
import { ReactNode, useMemo, useState } from "react";

/** Inline math. */
export function M({ children }: { children: string }) {
  const html = useMemo(
    () => katex.renderToString(children, { throwOnError: false }),
    [children],
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Display (block) math. */
export function MB({ children }: { children: string }) {
  const html = useMemo(
    () => katex.renderToString(children, { throwOnError: false, displayMode: true }),
    [children],
  );
  return <div className="mb" dangerouslySetInnerHTML={{ __html: html }} />;
}

/** A formula plus a legend that says what every symbol means. */
export function Formula({
  tex,
  symbols,
  title,
  sheet,
}: {
  tex: string;
  symbols: [string, ReactNode][];
  title?: string;
  /** "yes": printed on the midterm formula sheet; "no": must be memorized. */
  sheet?: "yes" | "no";
}) {
  return (
    <div className="formula">
      {(title || sheet) && (
        <div className="formula-title">
          {title}
          {sheet && (
            <span className={`sheet-badge ${sheet}`}>
              {sheet === "yes" ? "On the formula sheet" : "Not on the sheet: memorize"}
            </span>
          )}
        </div>
      )}
      <MB>{tex}</MB>
      <table className="symbols">
        <tbody>
          {symbols.map(([s, meaning], i) => (
            <tr key={i}>
              <td className="sym">
                <M>{s}</M>
              </td>
              <td>{meaning}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export type Step = { title: ReactNode; body: ReactNode };

/** A worked solution revealed one step at a time. */
export function Steps({ steps, title }: { steps: Step[]; title?: string }) {
  const [shown, setShown] = useState(1);
  const done = shown >= steps.length;
  return (
    <div className="steps">
      {title && <div className="steps-title">{title}</div>}
      <ol>
        {steps.slice(0, shown).map((s, i) => (
          <li key={i} className={i === shown - 1 ? "step current" : "step"}>
            <div className="step-title">{s.title}</div>
            <div className="step-body">{s.body}</div>
          </li>
        ))}
      </ol>
      <div className="steps-controls">
        <button disabled={done} onClick={() => setShown(shown + 1)}>
          {done ? "All steps shown" : `Next step (${shown}/${steps.length})`}
        </button>
        {!done && <button className="ghost" onClick={() => setShown(steps.length)}>Show all</button>}
        {shown > 1 && <button className="ghost" onClick={() => setShown(1)}>Restart</button>}
      </div>
    </div>
  );
}

type CalloutKind = "intuition" | "mistake" | "exam" | "note";
const calloutLabel: Record<CalloutKind, string> = {
  intuition: "Intuition",
  mistake: "Common mistake",
  exam: "Exam tip",
  note: "Note",
};

export function Callout({ kind, children }: { kind: CalloutKind; children: ReactNode }) {
  return (
    <div className={`callout ${kind}`}>
      <div className="callout-label">{calloutLabel[kind]}</div>
      <div>{children}</div>
    </div>
  );
}

/** Connects a lecture idea to where the labs practise it, with a link to the Labs section. */
export function FromLab({ to, label, children }: { to: string; label: string; children: ReactNode }) {
  return (
    <div className="callout lab">
      <div className="callout-label">From the lab</div>
      <div>{children}</div>
      <a className="lab-link" href={`#/${to}`}>
        Labs: {label} →
      </a>
    </div>
  );
}

/** Side by side: each math line next to the NumPy line that computes it. */
export function MathCode({ rows }: { rows: [string, string, ReactNode?][] }) {
  return (
    <table className="mathcode">
      <thead>
        <tr>
          <th>Math</th>
          <th>NumPy</th>
          <th>Shape / note</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([tex, code, note], i) => (
          <tr key={i}>
            <td>
              <M>{tex}</M>
            </td>
            <td>
              <code>{code}</code>
            </td>
            <td className="note">{note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Section heading inside a lesson. */
export function H({ children }: { children: ReactNode }) {
  return <h3 className="lesson-h">{children}</h3>;
}

export function Slides({ lecture, pages }: { lecture: number; pages: string }) {
  return (
    <p className="slides">
      Slides: Lecture {lecture}, pages {pages}
    </p>
  );
}

/** Small table helper for worked examples. */
export function Table({ head, rows }: { head: ReactNode[]; rows: ReactNode[][] }) {
  return (
    <table className="data">
      <thead>
        <tr>
          {head.map((h, i) => (
            <th key={i}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            {r.map((c, j) => (
              <td key={j}>{c}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A "predict before you look" prompt with a hidden answer. */
export function Think({ q, children }: { q: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="think">
      <div className="think-q">
        <strong>Think first:</strong> {q}
      </div>
      {open ? (
        <div className="think-a">{children}</div>
      ) : (
        <button className="ghost" onClick={() => setOpen(true)}>
          Show answer
        </button>
      )}
    </div>
  );
}
