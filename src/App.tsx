import { useEffect, useState } from "react";
import { lectures } from "./lessons";
import { lectureLabel } from "./lessons/types";
import { M } from "./components/ui";
import { QuizPage, SectionQuiz } from "./quiz/Quiz";
import { allQuestions, useResults } from "./quiz/store";
import { MockExam } from "./exam/MockExam";

function readHash() {
  const [, lec, sec] = window.location.hash.replace(/^#/, "").split("/");
  return { lec, sec };
}

function loadDone(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem("comp432-done") ?? "[]"));
  } catch {
    return new Set();
  }
}

function saveDone(s: Set<string>) {
  try {
    localStorage.setItem("comp432-done", JSON.stringify([...s]));
  } catch {
    /* progress just won't persist */
  }
}

const flat = lectures.flatMap((l) => l.sections.map((s) => ({ lec: l, sec: s })));

export default function App() {
  const [route, setRoute] = useState(readHash);
  const [done, setDone] = useState(loadDone);
  const [menuOpen, setMenuOpen] = useState(false);
  const results = useResults();
  const quizRight = allQuestions.filter((q) => results[q.id]).length;

  useEffect(() => {
    const f = () => {
      setRoute(readHash());
      setMenuOpen(false);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);

  const idx = flat.findIndex((f) => f.lec.id === route.lec && f.sec.id === route.sec);
  const cur = idx >= 0 ? flat[idx] : null;
  const key = cur ? `${cur.lec.id}/${cur.sec.id}` : "";

  function toggleDone() {
    const next = new Set(done);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setDone(next);
    saveDone(next);
  }

  return (
    <div className="app">
      <header className="topbar">
        <button className="menu-btn ghost" onClick={() => setMenuOpen(!menuOpen)} aria-label="Menu">
          ☰
        </button>
        <a href="#" className="brand">
          COMP 432 Study Guide
        </a>
        <span className="progress">
          {done.size}/{flat.length} sections understood · {quizRight}/{allQuestions.length} quiz correct
        </span>
      </header>
      <nav className={menuOpen ? "sidebar open" : "sidebar"}>
        <a href="#/quiz" className={"nav-quiz" + (route.lec === "quiz" ? " active" : "")}>
          <span className="check">✎</span>
          Quiz: practice all {allQuestions.length} questions
        </a>
        <a href="#/exam" className={"nav-quiz" + (route.lec === "exam" ? " active" : "")}>
          <span className="check">⏱</span>
          Mock midterm (exam format)
        </a>
        {lectures.map((l) => (
          <div key={l.id} className="nav-lecture">
            <div className="nav-lecture-title">
              {lectureLabel(l)}: {l.title}
            </div>
            {l.sections.map((s) => {
              const k = `${l.id}/${s.id}`;
              return (
                <a
                  key={s.id}
                  href={`#/${k}`}
                  className={(k === key ? "active " : "") + (done.has(k) ? "done" : "")}
                >
                  <span className="check">{done.has(k) ? "✓" : ""}</span>
                  {s.title}
                </a>
              );
            })}
          </div>
        ))}
      </nav>
      <main className="content">
        {cur ? (
          <article key={key}>
            <div className="eyebrow">
              {lectureLabel(cur.lec)} · {cur.lec.title}
            </div>
            <h1>{cur.sec.title}</h1>
            <p className="summary">{cur.sec.summary}</p>
            {cur.sec.body()}
            <SectionQuiz lecture={cur.lec.id} section={cur.sec.id} />
            <div className="lesson-footer">
              <label className="understood">
                <input type="checkbox" checked={done.has(key)} onChange={toggleDone} /> I understand this section
              </label>
              <div className="pager">
                {idx > 0 && (
                  <a href={`#/${flat[idx - 1].lec.id}/${flat[idx - 1].sec.id}`}>← {flat[idx - 1].sec.title}</a>
                )}
                {idx < flat.length - 1 && (
                  <a className="next" href={`#/${flat[idx + 1].lec.id}/${flat[idx + 1].sec.id}`}>
                    {flat[idx + 1].sec.title} →
                  </a>
                )}
              </div>
            </div>
          </article>
        ) : route.lec === "quiz" ? (
          <QuizPage />
        ) : route.lec === "exam" ? (
          <MockExam />
        ) : (
          <Home done={done} />
        )}
      </main>
    </div>
  );
}

function Home({ done }: { done: Set<string> }) {
  return (
    <article>
      <h1>COMP 432 — learn it step by step</h1>
      <p className="summary">
        Every section re-explains one idea from the slides in plain language, then works through it with real
        numbers, shows it visually, and connects each formula to the NumPy code that computes it.
      </p>
      <div className="callout exam">
        <div className="callout-label">Midterm: Monday 2 November, H-435, 17:45–19:45</div>
        <ul className="tight">
          <li>31 multiple-choice questions (90 points) + 2 bonus, on lectures 1–5 <strong>and labs 1–5</strong>.</li>
          <li>Every question has <strong>five options, one correct</strong>: text, a figure, or code to read. No proofs, no code writing.</li>
          <li>Closed book, <strong>no calculator</strong>. Wrong answers cost ¼ of the points; blanks cost nothing.</li>
          <li>The booklet's formula sheet has Lecture 3's summary table, gradient descent, SGD and momentum. Formula boxes in this app are tagged <span className="sheet-badge yes" style={{ float: "none" }}>On the formula sheet</span> or <span className="sheet-badge no" style={{ float: "none" }}>Not on the sheet: memorize</span>.</li>
        </ul>
        <div style={{ marginTop: 6 }}>
          Practise under exam conditions in the <a href="#/exam">mock midterm</a>, and drill single topics in the <a href="#/quiz">quiz</a>.
        </div>
      </div>
      <h3 className="lesson-h">How to use each section</h3>
      <ol className="howto">
        <li><strong>Why it matters</strong>: the idea in plain words, no symbols.</li>
        <li><strong>The formula, decoded</strong>: every symbol named, with its shape (e.g. <M>{"\\mathbf{X}\\in\\mathbb{R}^{N\\times D}"}</M>).</li>
        <li><strong>Worked example</strong>: click "Next step" and try each step yourself before you reveal it.</li>
        <li><strong>Visual</strong>: drag the sliders and watch what changes.</li>
        <li><strong>Math → code</strong>: run and edit the Python (Ctrl/Cmd+Enter runs a cell). Change a number and predict the output first.</li>
        <li><strong>Common mistakes / exam tips</strong>, and the slide pages to go back to.</li>
        <li><strong>Check yourself</strong>: a few quiz questions at the end. The <a href="#/quiz">Quiz page</a> mixes all of them, and can show only the ones you haven't got right yet.</li>
      </ol>
      {lectures.map((l) => (
        <div key={l.id} className="home-lecture">
          <h3 className="lesson-h">
            {lectureLabel(l)}: {l.title}
          </h3>
          <ul>
            {l.sections.map((s) => (
              <li key={s.id}>
                <a href={`#/${l.id}/${s.id}`}>{s.title}</a> {done.has(`${l.id}/${s.id}`) && "✓"}
                <div className="muted">{s.summary}</div>
              </li>
            ))}
          </ul>
        </div>
      ))}
      <p className="muted">
        The "Labs 1–5" sections cover the code from the lab tutorials and assignments, since the midterm includes the labs. Each
        one lists every function its code uses, with an example, and walks through the code line by line ("Next line"), showing
        the values each line produces. If Python itself is the obstacle, start with "Python you need to read the lab code".
      </p>
    </article>
  );
}
