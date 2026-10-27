import { useMemo, useState } from "react";
import { lectures } from "../lessons";
import { lectureLabel, lectureShort } from "../lessons/types";
import { M } from "../components/ui";
import { PyCell } from "../components/PyCell";
import { allQuestions, isClose, parseNumber, questionsFor, recordResult, resetResults, useResults } from "./store";
import { QuizItem, QuizKind, kindLabel } from "./types";

/** Renders text with $math$ and {{code}} segments. */
export function Rich({ text }: { text: string }) {
  const parts = text.split(/(\$[^$]+\$|\{\{[\s\S]+?\}\})/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("$") && p.endsWith("$") && p.length > 1 ? (
          <M key={i}>{p.slice(1, -1)}</M>
        ) : p.startsWith("{{") ? (
          <code key={i}>{p.slice(2, -2)}</code>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
    </>
  );
}

function fmtAnswer(v: number) {
  return String(+v.toFixed(4));
}

export function QuestionCard({ q, n, showSource }: { q: QuizItem; n?: number; showSource?: boolean }) {
  const [picked, setPicked] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [result, setResult] = useState<boolean | null>(null);
  const [error, setError] = useState("");
  const submitted = result !== null;

  function submit() {
    let ok: boolean;
    if (q.type === "mc") {
      if (picked === null) return setError("Pick an option first.");
      ok = picked === q.answer;
    } else {
      const v = parseNumber(input);
      if (v === null) return setError("Enter a number (fractions like 1/3 are fine).");
      ok = isClose(v, q.answer, q.tol);
    }
    setError("");
    setResult(ok);
    recordResult(q.id, ok);
  }

  function retry() {
    setPicked(null);
    setInput("");
    setResult(null);
  }

  const lec = lectures.find((l) => l.id === q.lecture);
  const sec = lec?.sections.find((s) => s.id === q.section);

  return (
    <div className={"qcard" + (submitted ? (result ? " right" : " wrong") : "")}>
      <div className="qmeta">
        {n !== undefined && <span className="qnum">Q{n}</span>}
        <span className={`qkind ${q.kind}`}>{kindLabel[q.kind]}</span>
        {showSource && sec && (
          <a className="qsrc" href={`#/${q.lecture}/${q.section}`}>
            {lectureShort(lec!)} · {sec.title}
          </a>
        )}
      </div>
      <div className="qprompt">
        <Rich text={q.prompt} />
      </div>
      {q.code && <pre className="qcode">{q.code}</pre>}
      {q.type === "mc" ? (
        <div className="qoptions">
          {q.options.map((o, i) => {
            let cls = "qopt";
            if (picked === i) cls += " picked";
            if (submitted && i === q.answer) cls += " correct";
            if (submitted && picked === i && i !== q.answer) cls += " incorrect";
            return (
              <button key={i} className={cls} disabled={submitted} onClick={() => setPicked(i)}>
                <span className="qletter">{String.fromCharCode(65 + i)}</span>
                <span>
                  <Rich text={o} />
                </span>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="qnumrow">
          <input
            type="text"
            inputMode="decimal"
            value={input}
            disabled={submitted}
            placeholder="your answer"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !submitted && submit()}
          />
          {submitted && <span className="muted">correct answer: {fmtAnswer(q.answer)}</span>}
        </div>
      )}
      {error && <div className="qerror">{error}</div>}
      <div className="qactions">
        {submitted ? (
          <>
            <span className="qverdict">{result ? "✓ Correct" : "✗ Not quite"}</span>
            <button className="ghost" onClick={retry}>Try again</button>
          </>
        ) : (
          <button onClick={submit}>Check answer</button>
        )}
      </div>
      {submitted && (
        <div className="qexplain">
          <Rich text={q.explain} />
          {q.code && <PyCell code={q.code} title="Run it to check" />}
        </div>
      )}
    </div>
  );
}

/** "Check yourself" block at the end of a lesson. */
export function SectionQuiz({ lecture, section }: { lecture: string; section: string }) {
  const qs = questionsFor(lecture, section);
  const results = useResults();
  if (!qs.length) return null;
  const right = qs.filter((q) => results[q.id]).length;
  const tried = qs.filter((q) => q.id in results).length;
  return (
    <section className="section-quiz">
      <h3 className="lesson-h">
        Check yourself
        <span className="qscore">
          {tried ? `${right}/${qs.length} correct` : `${qs.length} questions`}
        </span>
      </h3>
      <p className="muted">Answer before scrolling back up. Hand calculations accept fractions (e.g. 1/3) and are checked to about 2%.</p>
      {qs.map((q, i) => (
        <QuestionCard key={q.id} q={q} n={i + 1} />
      ))}
    </section>
  );
}

type Mode = "all" | "todo";

function shuffled<T>(a: T[], seed: number): T[] {
  const out = [...a];
  let s = seed;
  for (let i = out.length - 1; i > 0; i--) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    const j = s % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** The full quiz page with filters and scores. */
export function QuizPage() {
  const results = useResults();
  const [lec, setLec] = useState<string>("all");
  const [kind, setKind] = useState<QuizKind | "all">("all");
  const [mode, setMode] = useState<Mode>("all");
  const [seed, setSeed] = useState(0);
  const [limit, setLimit] = useState(10);

  // Freeze the "todo" set when filters change, so answering doesn't make cards vanish.
  const list = useMemo(() => {
    let qs = allQuestions.filter((q) => (lec === "all" || q.lecture === lec) && (kind === "all" || q.kind === kind));
    if (mode === "todo") qs = qs.filter((q) => !results[q.id]);
    if (seed) qs = shuffled(qs, seed);
    return limit ? qs.slice(0, limit) : qs;
  }, [lec, kind, mode, seed, limit]); // results deliberately left out

  const total = allQuestions.length;
  const right = allQuestions.filter((q) => results[q.id]).length;

  return (
    <article>
      <div className="eyebrow">Practice</div>
      <h1>Quiz</h1>
      <p className="summary">
        {total} questions on formulas, hand calculations and predicting code output. Each one links back to the section it tests.
      </p>
      <div className="quiz-stats">
        <div className="stat"><b>{right}</b>/{total} correct overall</div>
        {lectures.map((l) => {
          const qs = allQuestions.filter((q) => q.lecture === l.id);
          const ok = qs.filter((q) => results[q.id]).length;
          return (
            <div className="stat" key={l.id}>
              {lectureShort(l)}: <b>{ok}</b>/{qs.length}
              <div className="bar"><i style={{ width: `${(100 * ok) / qs.length}%` }} /></div>
            </div>
          );
        })}
      </div>
      <div className="quiz-filters">
        <div className="toggle-row">
          {["all", ...lectures.map((l) => l.id)].map((id) => (
            <button key={id} className={lec === id ? "" : "off"} onClick={() => setLec(id)}>
              {id === "all" ? "All" : lectureLabel(lectures.find((l) => l.id === id)!)}
            </button>
          ))}
        </div>
        <div className="toggle-row">
          {(["all", "formula", "calc", "code"] as const).map((k) => (
            <button key={k} className={kind === k ? "" : "off"} onClick={() => setKind(k)}>
              {k === "all" ? "All types" : kindLabel[k]}
            </button>
          ))}
        </div>
        <div className="toggle-row">
          <button className={mode === "all" ? "" : "off"} onClick={() => setMode("all")}>All questions</button>
          <button className={mode === "todo" ? "" : "off"} onClick={() => setMode("todo")}>Not yet correct</button>
          <button className={limit === 10 ? "" : "off"} onClick={() => setLimit(10)}>10 at a time</button>
          <button className={limit === 0 ? "" : "off"} onClick={() => setLimit(0)}>Show all</button>
          <button className="ghost" onClick={() => setSeed(1 + Math.floor(Math.random() * 1e6))}>🔀 Shuffle</button>
          <button
            className="ghost"
            onClick={() => {
              if (confirm("Clear all your quiz results?")) resetResults();
            }}
          >
            Reset scores
          </button>
        </div>
      </div>
      {list.length === 0 ? (
        <p className="muted">Nothing left here — everything in this filter is answered correctly. 🎉</p>
      ) : (
        list.map((q, i) => <QuestionCard key={`${seed}-${q.id}`} q={q} n={i + 1} showSource />)
      )}
      {limit > 0 && list.length === limit && (
        <button className="ghost" onClick={() => setSeed(1 + Math.floor(Math.random() * 1e6))}>Next 10 (new shuffle)</button>
      )}
    </article>
  );
}
