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
  plain,
}: {
  tex: string;
  symbols: [string, ReactNode][];
  title?: string;
  /** "yes": printed on the midterm formula sheet; "no": must be memorized. */
  sheet?: "yes" | "no";
  /** The formula in plain words, shown under it. */
  plain?: ReactNode;
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
      {plain && (
        <p className="formula-plain">
          <strong>In words:</strong> {plain}
        </p>
      )}
      <div className="table-wrap">
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
    <div className="table-wrap">
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
    </div>
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
    <div className="table-wrap">
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
    </div>
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

/** The functions a piece of lab code uses: what each one does, its shapes, and a tiny example with its result. */
export function FnRef({ rows }: { rows: [string, ReactNode, ReactNode, string?, string?][] }) {
  return (
    <>
      <Table
        head={["Call", "What it does", "Shapes / types", "Example → result"]}
        rows={rows.map(([call, does, shapes, ex, res]) => [
          <code className="nowrap">{call}</code>,
          does,
          shapes,
          ex ? (
            <>
              <code>{ex}</code>
              {res !== undefined && (
                <>
                  {" "}→ <code>{res}</code>
                </>
              )}
            </>
          ) : (
            ""
          ),
        ])}
      />
    </>
  );
}

export type WalkStep = {
  /** 1-based line number, or an inclusive [from, to] range. */
  line: number | [number, number];
  say: ReactNode;
  /** Variables after these lines run: [name, value or shape]. */
  vars?: [string, ReactNode][];
};

/** Steps through a piece of code line by line: highlights the lines, says what they do, and shows the values they produce. */
export function CodeWalk({ title, code, steps }: { title?: string; code: string; steps: WalkStep[] }) {
  const [i, setI] = useState(0);
  const [all, setAll] = useState(false);
  const lines = code.replace(/^\n/, "").replace(/\s+$/, "").split("\n");
  const range = (s: WalkStep) => (typeof s.line === "number" ? [s.line, s.line] : s.line);
  const [from, to] = range(steps[i]);
  const label = (s: WalkStep) => {
    const [a, b] = range(s);
    return a === b ? `Line ${a}` : `Lines ${a}–${b}`;
  };
  const detail = (s: WalkStep) => (
    <>
      <div>{s.say}</div>
      {s.vars && s.vars.length > 0 && (
        <div className="table-wrap">
          <table className="walk-vars">
            <tbody>
              {s.vars.map(([name, val], k) => (
                <tr key={k}>
                  <td>
                    <code>{name}</code>
                  </td>
                  <td>{typeof val === "string" ? <code>{val}</code> : val}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
  return (
    <div className="steps walk">
      <div className="steps-title">{title ?? "Line by line"}</div>
      <pre className="walk-code">
        {lines.map((l, k) => {
          const on = !all && k + 1 >= from && k + 1 <= to;
          return (
            <div key={k} className={on ? "walk-line on" : "walk-line"} aria-current={on ? "step" : undefined}>
              <span className="walk-no">{k + 1}</span>
              {l || " "}
            </div>
          );
        })}
      </pre>
      {all ? (
        <ol className="walk-all">
          {steps.map((s, k) => (
            <li key={k}>
              <div className="step-title">{label(s)}</div>
              {detail(s)}
            </li>
          ))}
        </ol>
      ) : (
        <div className="walk-say" aria-live="polite">
          <div className="step-title">
            {label(steps[i])} <span className="muted">({i + 1}/{steps.length})</span>
          </div>
          {detail(steps[i])}
        </div>
      )}
      <div className="steps-controls">
        {!all && (
          <>
            <button disabled={i === 0} className="ghost" onClick={() => setI(i - 1)}>
              ← Back
            </button>
            <button disabled={i === steps.length - 1} onClick={() => setI(i + 1)}>
              Next line →
            </button>
          </>
        )}
        <button className="ghost" onClick={() => setAll(!all)}>
          {all ? "Step through instead" : "Show all as a list"}
        </button>
      </div>
    </div>
  );
}
