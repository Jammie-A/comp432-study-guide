import { ReactNode, useEffect, useMemo, useState } from "react";
import { lectures } from "../lessons";
import { lectureShort } from "../lessons/types";
import { Rich } from "../quiz/Quiz";
import { ExamQuestion, FigureId, examPool } from "./questions";
import { privateExam } from "../private";

// The public pool plus any local-only questions (see src/private.tsx).
const pool: ExamQuestion[] = [...examPool, ...privateExam];
import { ArrowsFigure, EarlyStoppingFigure, LearningRateFigure, ZigzagFigure } from "./figures";

const figures: Record<FigureId, () => ReactNode> = {
  "early-stopping": EarlyStoppingFigure,
  "learning-rate": LearningRateFigure,
  arrows: ArrowsFigure,
  zigzag: ZigzagFigure,
};

type Mode = "full" | "quick" | "samples";
const modeName: Record<Mode, string> = { full: "Full mock midterm", quick: "Quick practice", samples: "The 3 handout examples" };

// Questions per lecture, mirroring "31 questions on lectures 1–5" plus 2 bonus.
const FULL_PLAN: Record<number, number> = { 1: 4, 2: 7, 3: 7, 4: 6, 5: 7 };
const QUICK_PLAN: Record<number, number> = { 1: 2, 2: 2, 3: 2, 4: 2, 5: 2 };
// Real exam: 90 points of work in a 120-minute slot.
const MINUTES_PER_POINT = 120 / 90;

type Running = { mode: Mode; ids: string[]; answers: Record<string, number>; endsAt: number };
type Attempt = { date: string; mode: Mode; score: number; max: number; bonus: number };

const CUR_KEY = "comp432-exam-current";
const HIST_KEY = "comp432-exam-history";

function read<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
function write(key: string, v: unknown) {
  try {
    if (v === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* not persisted */
  }
}

function pick<T>(a: T[], n: number): T[] {
  const out = [...a];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out.slice(0, n);
}

function buildExam(mode: Mode): ExamQuestion[] {
  if (mode === "samples") return pool.filter((q) => q.sample);
  const plan = mode === "full" ? FULL_PLAN : QUICK_PLAN;
  const main = Object.entries(plan).flatMap(([lec, n]) =>
    pick(pool.filter((q) => !q.bonus && q.lecture === +lec), n),
  );
  main.sort((a, b) => a.lecture - b.lecture);
  return mode === "full" ? [...main, ...pick(pool.filter((q) => q.bonus), 2)] : main;
}

const byId = new Map(pool.map((q) => [q.id, q]));

function scoreOf(qs: ExamQuestion[], answers: Record<string, number>) {
  let score = 0, bonus = 0, max = 0, right = 0, wrong = 0, blank = 0;
  for (const q of qs) {
    const a = answers[q.id];
    const pts = a === undefined ? 0 : a === q.answer ? q.points : -q.points / 4;
    if (a === undefined) blank++;
    else if (a === q.answer) right++;
    else wrong++;
    if (q.bonus) bonus += pts;
    else {
      score += pts;
      max += q.points;
    }
  }
  return { score, bonus, max, right, wrong, blank };
}

const letters = ["A", "B", "C", "D", "E"];

export function MockExam() {
  const [running, setRunning] = useState<Running | null>(() => read<Running | null>(CUR_KEY, null));
  const [finished, setFinished] = useState<{ ids: string[]; answers: Record<string, number>; mode: Mode } | null>(null);
  const [history, setHistory] = useState<Attempt[]>(() => read<Attempt[]>(HIST_KEY, []));
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  const qs = useMemo(() => (running ? running.ids.map((id) => byId.get(id)!).filter(Boolean) : []), [running]);
  const left = running ? Math.max(0, running.endsAt - now) : 0;

  function start(mode: Mode) {
    const exam = buildExam(mode);
    const pts = exam.filter((q) => !q.bonus).reduce((s, q) => s + q.points, 0);
    const r: Running = { mode, ids: exam.map((q) => q.id), answers: {}, endsAt: Date.now() + pts * MINUTES_PER_POINT * 60000 };
    setRunning(r);
    setFinished(null);
    write(CUR_KEY, r);
    window.scrollTo(0, 0);
  }

  function answer(id: string, a: number | undefined) {
    // Functional update: quick successive clicks must build on each other, not on a stale render.
    setRunning((prev) => {
      if (!prev) return prev;
      const answers = { ...prev.answers };
      if (a === undefined) delete answers[id];
      else answers[id] = a;
      const r = { ...prev, answers };
      write(CUR_KEY, r);
      return r;
    });
  }

  function submit() {
    if (!running) return;
    const s = scoreOf(qs, running.answers);
    const attempt: Attempt = { date: new Date().toISOString(), mode: running.mode, score: s.score, max: s.max, bonus: s.bonus };
    const h = [attempt, ...history].slice(0, 20);
    setHistory(h);
    write(HIST_KEY, h);
    setFinished({ ids: running.ids, answers: running.answers, mode: running.mode });
    setRunning(null);
    write(CUR_KEY, null);
    window.scrollTo(0, 0);
  }

  // Time's up: hand in automatically.
  useEffect(() => {
    if (running && left === 0) submit();
  }, [left === 0 && !!running]); // fires once when the timer reaches zero

  if (running) {
    const done = Object.keys(running.answers).length;
    const mm = Math.floor(left / 60000);
    const ss = Math.floor((left % 60000) / 1000);
    return (
      <article>
        <div className="exam-bar">
          <span className={left < 5 * 60000 ? "exam-time low" : "exam-time"}>
            ⏱ {mm}:{String(ss).padStart(2, "0")}
          </span>
          <span>{done}/{qs.length} answered</span>
          <button
            onClick={() => {
              if (confirm(`Hand in now? ${qs.length - done} question(s) left blank (blank = 0 points).`)) submit();
            }}
          >
            Hand in
          </button>
        </div>
        <div className="eyebrow">{modeName[running.mode]}</div>
        <h1>Midterm practice booklet</h1>
        <p className="muted">
          One answer per question. Correct: +points. Wrong: −¼ of the points. Blank: 0. Your answers are saved if you leave
          this page; the timer keeps running.
        </p>
        {qs.map((q, i) => (
          <ExamCard key={q.id} q={q} n={i + 1} picked={running.answers[q.id]} onPick={(a) => answer(q.id, a)} />
        ))}
        <button onClick={submit}>Hand in and see results</button>
      </article>
    );
  }

  if (finished) {
    const fqs = finished.ids.map((id) => byId.get(id)!).filter(Boolean);
    return <Results qs={fqs} answers={finished.answers} mode={finished.mode} onBack={() => setFinished(null)} />;
  }

  return (
    <article>
      <div className="eyebrow">Practice</div>
      <h1>Mock midterm</h1>
      <p className="summary">Practise under the real exam's rules, then review every answer with its explanation.</p>
      <div className="callout exam">
        <div className="callout-label">The real midterm (from the Lecture 5 handout)</div>
        <ul className="tight">
          <li>Monday 2 November, H-435, 17:45–19:45. Bring HB pencils, an eraser and photo ID.</li>
          <li>31 questions (90 points) + 2 bonus questions (4 points) on lectures 1–5 and labs 1–5; bonus on the extra lectures.</li>
          <li>Each question: text, a figure or code, <strong>five options, exactly one correct</strong>. No proofs, no code writing.</li>
          <li>Closed book, <strong>no calculator</strong>. A point is about a minute of work: 90 minutes of work in a 120-minute slot.</li>
          <li>Wrong answer: −¼ of the question's points. Blank: 0 (bonus included).</li>
          <li>Formula sheet in the booklet: Lecture 3's summary table, gradient descent, SGD, momentum.</li>
          <li>Only the scantron sheet is graded: bubble as you go, and skip the row when you skip a question.</li>
        </ul>
      </div>
      <div className="callout intuition">
        <div className="callout-label">Should you guess?</div>
        With 5 options and a ¼ penalty, a <strong>blind guess is worth 0 on average</strong>: ⅕·(+1) + ⅘·(−¼) = 0.
        But if you can rule out <strong>even one</strong> option, guessing among the other 4 gains on average:
        ¼·(+1) + ¾·(−¼) = +1/16 of the points. Rule out two → +1/6. So: never leave a question blank if you can eliminate
        anything.
      </div>
      <div className="exam-modes">
        <button onClick={() => start("full")}>
          <b>Full mock midterm</b>
          <span>31 questions + 2 bonus, timed like the real exam (≈ 1.5–2 h)</span>
        </button>
        <button onClick={() => start("quick")}>
          <b>Quick practice</b>
          <span>10 questions, 2 per lecture, ≈ 30 min</span>
        </button>
        <button onClick={() => start("samples")}>
          <b>The 3 handout examples</b>
          <span>The sample questions from the slides, 8 min</span>
        </button>
      </div>
      <p className="muted">
        The pool has {pool.filter((q) => !q.bonus).length} questions + {pool.filter((q) => q.bonus).length} bonus;
        each mock draws a new random set. Point values are my estimates of each question's effort (the real exam doesn't
        publish them in advance), and the answers to the handout examples are my own worked solutions — compare with your lab TA.
      </p>
      {history.length > 0 && (
        <>
          <h3 className="lesson-h">Past attempts</h3>
          <table className="data">
            <thead><tr><th>When</th><th>Mode</th><th>Score</th><th>%</th><th>Bonus</th></tr></thead>
            <tbody>
              {history.map((h, i) => (
                <tr key={i}>
                  <td>{new Date(h.date).toLocaleString()}</td>
                  <td>{modeName[h.mode]}</td>
                  <td>{fmt(h.score)} / {h.max}</td>
                  <td>{Math.round((100 * h.score) / h.max)}%</td>
                  <td>{fmt(h.bonus)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <button className="ghost" onClick={() => { if (confirm("Clear attempt history?")) { setHistory([]); write(HIST_KEY, null); } }}>
            Clear history
          </button>
        </>
      )}
    </article>
  );
}

function fmt(v: number) {
  return String(+v.toFixed(2));
}

function ExamCard({ q, n, picked, onPick, review }: { q: ExamQuestion; n: number; picked?: number; onPick?: (a: number | undefined) => void; review?: boolean }) {
  const Fig = q.figure ? figures[q.figure] : null;
  const state = review ? (picked === undefined ? "blank" : picked === q.answer ? "right" : "wrong") : "";
  return (
    <div className={`qcard exam-card ${state}`}>
      <div className="qmeta">
        <span className="qnum">Question {n}</span>
        <span className="qkind">{q.points} {q.points === 1 ? "point" : "points"}{q.bonus ? " · bonus" : ""}</span>
        {q.sample && <span className="qkind code">handout example</span>}
        {review && <span className="qverdict">{state === "right" ? `✓ +${q.points}` : state === "wrong" ? `✗ −${fmt(q.points / 4)}` : "blank · 0"}</span>}
      </div>
      <div className="qprompt"><Rich text={q.prompt} /></div>
      {Fig && <div className="exam-figure"><Fig /></div>}
      {q.code && <pre className="qcode">{q.code}</pre>}
      <div className="qoptions">
        {q.options.map((o, i) => {
          let cls = "qopt";
          if (picked === i) cls += " picked";
          if (review && i === q.answer) cls += " correct";
          if (review && picked === i && i !== q.answer) cls += " incorrect";
          return (
            <button key={i} className={cls} disabled={review} onClick={() => onPick?.(picked === i ? undefined : i)}>
              <span className={picked === i ? "bubble filled" : "bubble"}>{letters[i]}</span>
              <span><Rich text={o} /></span>
            </button>
          );
        })}
      </div>
      {!review && picked !== undefined && (
        <button className="ghost small" onClick={() => onPick?.(undefined)}>Clear answer (leave blank)</button>
      )}
      {review && (
        <div className="qexplain">
          <strong>Answer: {letters[q.answer]}.</strong> <Rich text={q.explain} />{" "}
          <a href={`#/${q.section}`}>Review this section →</a>
        </div>
      )}
    </div>
  );
}

function Results({ qs, answers, mode, onBack }: { qs: ExamQuestion[]; answers: Record<string, number>; mode: Mode; onBack: () => void }) {
  const s = scoreOf(qs, answers);
  const pct = s.max ? (100 * s.score) / s.max : 0;
  const [filter, setFilter] = useState<"all" | "missed">("missed");
  const shown = qs.filter((q) => filter === "all" || answers[q.id] !== q.answer);
  return (
    <article>
      <div className="eyebrow">{modeName[mode]} · results</div>
      <h1>{fmt(s.score)} / {s.max} points ({Math.round(pct)}%)</h1>
      <p className="summary">
        On the real 90-point scale that's about <b>{Math.round((pct / 100) * 90)}</b> / 90
        {s.bonus !== 0 && <>, plus {fmt(s.bonus)} bonus</>}.
      </p>
      <div className="quiz-stats">
        <div className="stat"><b>{s.right}</b> correct</div>
        <div className="stat"><b>{s.wrong}</b> wrong (−¼ each)</div>
        <div className="stat"><b>{s.blank}</b> blank</div>
        {lectures.map((l) => {
          const lq = qs.filter((q) => q.lecture === l.number && !q.bonus);
          if (!lq.length) return null;
          const ls = scoreOf(lq, answers);
          return (
            <div className="stat" key={l.id}>
              {lectureShort(l)}: <b>{fmt(ls.score)}</b>/{ls.max}
              <div className="bar"><i style={{ width: `${Math.max(0, (100 * ls.score) / ls.max)}%` }} /></div>
            </div>
          );
        })}
      </div>
      <div className="toggle-row">
        <button className={filter === "missed" ? "" : "off"} onClick={() => setFilter("missed")}>Review wrong & blank</button>
        <button className={filter === "all" ? "" : "off"} onClick={() => setFilter("all")}>Review all</button>
        <button className="ghost" onClick={onBack}>Back to mock midterm</button>
      </div>
      {shown.length === 0 && <p className="muted">Nothing to review — every answer was correct. 🎉</p>}
      {shown.map((q) => (
        <ExamCard key={q.id} q={q} n={qs.indexOf(q) + 1} picked={answers[q.id]} review />
      ))}
    </article>
  );
}
