// Shared Pyodide worker. One worker for the whole app; restarted if a run is stopped.
// The worker source is bundled as text and started from a Blob URL, so no
// separate script request is needed (some embedded browsers stall on those).
import workerSource from "./py-worker.js?raw";

export type RunEvent =
  | { type: "stdout" | "stderr" | "status"; text: string }
  | { type: "done"; error?: string; figures?: string[] };

type Listener = (ev: RunEvent) => void;

let worker: Worker | null = null;
let ready = false;
let nextId = 1;
const listeners = new Map<number, Listener>();
const readyListeners = new Set<(r: boolean) => void>();

function getWorker(): Worker {
  if (worker) return worker;
  const url = URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" }));
  worker = new Worker(url);
  worker.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg.type === "ready") {
      ready = true;
      readyListeners.forEach((f) => f(true));
      return;
    }
    const l = listeners.get(msg.id);
    if (!l) return;
    l(msg);
    if (msg.type === "done") listeners.delete(msg.id);
  };
  return worker;
}

export function isPythonReady() {
  return ready;
}

export function onPythonReady(f: (r: boolean) => void) {
  readyListeners.add(f);
  return () => {
    readyListeners.delete(f);
  };
}

export function warmUpPython() {
  getWorker();
}

export function runPython(code: string, onEvent: Listener): number {
  const id = nextId++;
  listeners.set(id, onEvent);
  getWorker().postMessage({ id, code });
  return id;
}

/** Kills a runaway program by terminating the worker; the next run starts a fresh one. */
export function stopPython() {
  worker?.terminate();
  worker = null;
  ready = false;
  readyListeners.forEach((f) => f(false));
  listeners.forEach((l) => l({ type: "done", error: "Stopped. (Python restarts on the next run.)" }));
  listeners.clear();
}
