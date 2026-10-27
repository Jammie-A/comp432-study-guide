// Runs Python (Pyodide) off the main thread so long loops don't freeze the page.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const ready = (async () => {
  const py = await loadPyodide();
  await py.loadPackage("numpy");
  return py;
})();

// Saves every open matplotlib figure as a base64 PNG and closes it.
const COLLECT_FIGURES = `
def __collect_figures():
    import sys
    if "matplotlib.pyplot" not in sys.modules:
        return []
    import io, base64
    import matplotlib.pyplot as plt
    out = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=100, bbox_inches="tight")
        out.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return out
__collect_figures()
`;

self.onmessage = async (e) => {
  const { id, code } = e.data;
  let py;
  try {
    py = await ready;
  } catch (err) {
    self.postMessage({ id, type: "done", error: "Could not load Python: " + err });
    return;
  }
  const send = (type, text) => self.postMessage({ id, type, text });
  py.setStdout({ batched: (t) => send("stdout", t) });
  py.setStderr({ batched: (t) => send("stderr", t) });
  try {
    if (code.includes("matplotlib")) {
      send("status", "Loading matplotlib (first time only)…");
      await py.loadPackage("matplotlib");
      py.runPython("import matplotlib; matplotlib.use('Agg')");
    }
    await py.loadPackagesFromImports(code);
    const globals = py.globals.get("dict")();
    await py.runPythonAsync(code, { globals });
    const figs = py.runPython(COLLECT_FIGURES, { globals }).toJs();
    self.postMessage({ id, type: "done", figures: figs });
    globals.destroy();
  } catch (err) {
    // Keep only the useful tail of Python tracebacks.
    const msg = String(err.message || err);
    const lines = msg.split("\n");
    const start = lines.findIndex((l) => l.includes('File "<exec>"'));
    self.postMessage({
      id,
      type: "done",
      error: start >= 0 ? lines.slice(start).join("\n") : msg,
    });
  }
};

ready.then(() => self.postMessage({ type: "ready" })).catch(() => {});
