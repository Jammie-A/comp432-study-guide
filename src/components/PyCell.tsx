import { KeyboardEvent, useEffect, useRef, useState } from "react";
import { isPythonReady, onPythonReady, runPython, stopPython, warmUpPython } from "./python";

type Out = { kind: "out" | "err" | "status"; text: string };

/**
 * An editable, runnable Python cell. `readOnly` cells (e.g. PyTorch code, which
 * can't run in the browser) are shown with the same styling but no Run button.
 */
export function PyCell({
  code,
  title,
  readOnly,
  expected,
}: {
  code: string;
  title?: string;
  readOnly?: boolean;
  expected?: string;
}) {
  const initial = code.trim() + "\n";
  const [src, setSrc] = useState(initial);
  const [out, setOut] = useState<Out[]>([]);
  const [figs, setFigs] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [pyReady, setPyReady] = useState(isPythonReady());
  const ta = useRef<HTMLTextAreaElement>(null);

  useEffect(() => onPythonReady(setPyReady), []);

  function run() {
    warmUpPython();
    setOut(pyReady ? [] : [{ kind: "status", text: "Starting Python (first run takes a few seconds)…" }]);
    setFigs([]);
    setRunning(true);
    runPython(src, (ev) => {
      if (ev.type === "done") {
        setRunning(false);
        setOut((o) => {
          const kept = o.filter((x) => x.kind !== "status");
          return ev.error ? [...kept, { kind: "err", text: ev.error }] : kept;
        });
        if (ev.figures) setFigs(ev.figures);
      } else if (ev.type === "status") {
        setOut((o) => [...o, { kind: "status", text: ev.text }]);
      } else {
        setOut((o) => [
          ...o.filter((x) => x.kind !== "status"),
          { kind: ev.type === "stdout" ? "out" : "err", text: ev.text },
        ]);
      }
    });
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!running) run();
    } else if (e.key === "Tab") {
      e.preventDefault();
      const el = e.currentTarget;
      const { selectionStart: s, selectionEnd: end } = el;
      const next = src.slice(0, s) + "    " + src.slice(end);
      setSrc(next);
      requestAnimationFrame(() => el.setSelectionRange(s + 4, s + 4));
    }
  }

  const lines = src.split("\n").length;

  return (
    <div className="pycell">
      <div className="pycell-head">
        <span className="pycell-title">{title ?? (readOnly ? "Python (read-only)" : "Python")}</span>
        {!readOnly && (
          <span className="pycell-actions">
            {src !== initial && (
              <button className="ghost" onClick={() => setSrc(initial)}>
                Reset code
              </button>
            )}
            {running ? (
              <button onClick={stopPython}>Stop</button>
            ) : (
              <button onClick={run} title="Ctrl/Cmd + Enter">
                ▶ Run
              </button>
            )}
          </span>
        )}
      </div>
      <textarea
        ref={ta}
        className="code"
        value={src}
        readOnly={readOnly}
        spellCheck={false}
        rows={Math.min(Math.max(lines, 3), 30)}
        onChange={(e) => setSrc(e.target.value)}
        onKeyDown={onKey}
      />
      {(out.length > 0 || figs.length > 0) && (
        <div className="pycell-out">
          {out.map((o, i) => (
            <pre key={i} className={o.kind}>
              {o.text}
            </pre>
          ))}
          {figs.map((f, i) => (
            <img key={i} src={`data:image/png;base64,${f}`} alt="matplotlib figure" />
          ))}
        </div>
      )}
      {expected && <div className="expected">Expected output: <code>{expected}</code></div>}
    </div>
  );
}
