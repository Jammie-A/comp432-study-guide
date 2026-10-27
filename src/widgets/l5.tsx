import { useState } from "react";
import { Chart, Slider, Widget, polyline } from "../components/chart";
import { M } from "../components/ui";
import { rng } from "./math";

// ---------- Vanishing / exploding gradients in a scalar chain ----------

export function VanishingWidget() {
  const [L, setL] = useState(10);
  const [w, setW] = useState(1);
  const [skip, setSkip] = useState(false);
  // Every z_k = 0 (bias −w/2, h = 1/2), so σ'(z_k) = 1/4 and each layer multiplies by w/4.
  const factor = (skip ? 1 : 0) + w * 0.25;
  const grads = Array.from({ length: L }, (_, i) => Math.pow(factor, i)); // |dJ/dh_k| relative to the last layer
  const ratio = Math.pow(factor, L - 1);
  const label = factor < 0.999 ? "vanishing" : factor > 1.001 ? "exploding" : "stable";
  return (
    <Widget
      title="How the gradient changes as it flows back through L layers"
      caption={<>Each sigmoid layer multiplies the gradient by <M>{"w\\,\\sigma'(z_k)"}</M>; here <M>{"\\sigma'(z_k)=\\tfrac14"}</M> in every layer (the slide's setup). The bars show <M>{"|dJ/dh_k|"}</M> relative to the last layer, on a log scale.</>}
    >
      <div className="widget-row">
        <Chart xDomain={[0.5, L + 0.5]} yDomain={[-12, 6]} xLabel="layer k (1 = first, L = last, next to the loss)" yLabel="log₁₀ |gradient|">
          {(s) => (
            <>
              <line x1={s.x(0.5)} x2={s.x(L + 0.5)} y1={s.y(0)} y2={s.y(0)} stroke="var(--muted)" strokeDasharray="4 3" />
              {grads.map((g, i) => {
                const k = L - i; // layer index: i = 0 is the last layer
                const v = Math.max(-12, Math.min(6, Math.log10(g)));
                return <rect key={k} x={s.x(k - 0.35)} width={s.x(0.7) - s.x(0)} y={Math.min(s.y(v), s.y(0))} height={Math.abs(s.y(v) - s.y(0))} fill={factor < 1 ? "var(--c1)" : factor > 1 ? "var(--c2)" : "var(--c3)"} />;
              })}
            </>
          )}
        </Chart>
        <div>
          <Slider label="depth L" value={L} min={2} max={20} step={1} onChange={setL} />
          <Slider label={<>weight <M>w</M></>} value={w} min={0.5} max={8} step={0.5} onChange={setW} />
          <label className="slider" style={{ gridTemplateColumns: "auto 1fr" }}>
            <input type="checkbox" checked={skip} onChange={(e) => setSkip(e.target.checked)} style={{ width: "auto" }} />
            <span>add a shortcut around every layer (factor becomes <M>{"1 + w\\sigma'"}</M>)</span>
          </label>
          <div className="readout">
            factor per layer = {skip ? "1 + " : ""}{w} · ¼ = <b>{factor.toFixed(3)}</b><br />
            ∂J/∂w₁ ÷ ∂J/∂w_L = factor^(L−1) = <b>{ratio < 1e-3 || ratio > 1e4 ? ratio.toExponential(1) : ratio.toPrecision(3)}</b><br />
            → <b>{label}</b>
          </div>
          <p className="muted">Slide values: factor ¼ → 0.25, 0.004, 4·10⁻⁶, 4·10⁻¹² for L = 2, 5, 10, 20. Factor 2 (w = 8) → 2, 16, 512, 5·10⁵.</p>
        </div>
      </div>
    </Widget>
  );
}

// ---------- 1-D convolution, step by step ----------

const CONV_X = [3, 5, 2, -1, 0, 4, -2, -3];
const KERNELS: Record<string, number[]> = {
  "difference (−1, 0, 1)": [-1, 0, 1],
  "blur (1, 1, 1)": [1, 1, 1],
  "identity (0, 1, 0)": [0, 1, 0],
};

export function ConvWidget() {
  const [kname, setKname] = useState(Object.keys(KERNELS)[0]);
  const [P, setP] = useState(1);
  const [stride, setStride] = useState(1);
  const [pos, setPos] = useState(0);
  const k = KERNELS[kname];
  const F = k.length;
  const xp = [...Array(P).fill(0), ...CONV_X, ...Array(P).fill(0)];
  const Dout = Math.floor((CONV_X.length + 2 * P - F) / stride) + 1;
  const out = Array.from({ length: Dout }, (_, i) => k.reduce((s, kv, j) => s + kv * xp[i * stride + j], 0));
  const p = Math.min(pos, Dout - 1);
  const start = p * stride;
  const cell = 44;
  const W = Math.max(xp.length, 8) * cell + 20;
  return (
    <Widget
      title="1-D convolution: slide the kernel, take dot products"
      caption={<>Output length <M>{"D_{\\text{out}} = \\lfloor (D + 2P - F)/s \\rfloor + 1"}</M> with D = 8 inputs. Gray cells are zero padding.</>}
    >
      <div className="toggle-row">
        {Object.keys(KERNELS).map((n) => (
          <button key={n} className={n === kname ? "" : "off"} onClick={() => setKname(n)}>{n}</button>
        ))}
      </div>
      <Slider label="padding P" value={P} min={0} max={2} step={1} onChange={(v) => { setP(v); setPos(0); }} />
      <Slider label="stride s" value={stride} min={1} max={3} step={1} onChange={(v) => { setStride(v); setPos(0); }} />
      <svg viewBox={`0 0 ${W} 200`} className="chart" style={{ maxWidth: W }}>
        <text x={4} y={14} style={{ fontSize: 12, fill: "var(--muted)" }}>input (padded)</text>
        {xp.map((v, i) => {
          const pad = i < P || i >= P + CONV_X.length;
          const inWin = i >= start && i < start + F;
          return (
            <g key={i}>
              <rect x={10 + i * cell} y={20} width={cell - 4} height={34} rx={5} fill={inWin ? "var(--accent-soft)" : pad ? "var(--code-bg)" : "var(--panel)"} stroke={inWin ? "var(--accent)" : "var(--border)"} strokeWidth={inWin ? 2 : 1} />
              <text x={10 + i * cell + (cell - 4) / 2} y={42} textAnchor="middle" style={{ fontSize: 14, fill: pad ? "var(--muted)" : "var(--text)" }}>{v}</text>
            </g>
          );
        })}
        {k.map((kv, j) => (
          <g key={j}>
            <rect x={10 + (start + j) * cell} y={64} width={cell - 4} height={28} rx={5} fill="none" stroke="var(--c2)" strokeWidth={1.5} />
            <text x={10 + (start + j) * cell + (cell - 4) / 2} y={83} textAnchor="middle" style={{ fontSize: 13, fill: "var(--c2)" }}>×{kv}</text>
          </g>
        ))}
        <text x={4} y={118} style={{ fontSize: 12, fill: "var(--muted)" }}>output ({Dout} values)</text>
        {out.map((v, i) => (
          <g key={i}>
            <rect x={10 + i * cell} y={124} width={cell - 4} height={34} rx={5} fill={i === p ? "var(--accent-soft)" : "var(--panel)"} stroke={i === p ? "var(--accent)" : "var(--border)"} strokeWidth={i === p ? 2 : 1} />
            <text x={10 + i * cell + (cell - 4) / 2} y={146} textAnchor="middle" style={{ fontSize: 14, fontWeight: i === p ? 700 : 400 }}>{v}</text>
          </g>
        ))}
        <text x={4} y={188} style={{ fontSize: 13 }}>
          z{p + 1} = {k.map((kv, j) => `${kv}·${xp[start + j] < 0 ? `(${xp[start + j]})` : xp[start + j]}`).join(" + ")} = {out[p]}
        </text>
      </svg>
      <div className="toggle-row">
        <button onClick={() => setPos((p + 1) % Dout)}>Next position →</button>
        <span className="readout">D_out = ⌊(8 + 2·{P} − {F}) / {stride}⌋ + 1 = <b>{Dout}</b></span>
      </div>
    </Widget>
  );
}

// ---------- Batch norm vs layer norm on the slide's matrix ----------

const NORM_Z = [
  [0.2, 12, 3],
  [-0.4, 9, 1],
  [0.6, 14, 2],
  [-0.4, 5, 6],
];

export function NormWidget() {
  const [mode, setMode] = useState<"batch" | "layer">("batch");
  const cols = NORM_Z[0].length;
  const stats = (vals: number[]) => {
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length);
    return { m, sd };
  };
  const colStats = Array.from({ length: cols }, (_, d) => stats(NORM_Z.map((r) => r[d])));
  const rowStats = NORM_Z.map((r) => stats(r));
  const norm = NORM_Z.map((r, n) => r.map((v, d) => {
    const st = mode === "batch" ? colStats[d] : rowStats[n];
    return (v - st.m) / st.sd;
  }));
  const color = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
  const tint = (n: number, d: number) => (mode === "batch" ? color[d % 4] : color[n % 4]);
  return (
    <Widget
      title="Batch norm vs layer norm: same matrix, different axis"
      caption="Rows are samples (B = 4), columns are hidden units (3). Cells with the same color share one mean and standard deviation."
    >
      <div className="toggle-row">
        <button className={mode === "batch" ? "" : "off"} onClick={() => setMode("batch")}>Batch norm (per column / unit)</button>
        <button className={mode === "layer" ? "" : "off"} onClick={() => setMode("layer")}>Layer norm (per row / sample)</button>
      </div>
      <div className="widget-row">
        <div>
          <div className="muted">Z (preactivations)</div>
          <table className="data">
            <tbody>
              {NORM_Z.map((r, n) => (
                <tr key={n}>
                  {r.map((v, d) => <td key={d} style={{ boxShadow: `inset 4px 0 0 ${tint(n, d)}`, textAlign: "right" }}>{v}</td>)}
                  {mode === "layer" && <td className="muted">μ={rowStats[n].m.toFixed(2)}, σ={rowStats[n].sd.toFixed(2)}</td>}
                </tr>
              ))}
              {mode === "batch" && (
                <>
                  <tr>{colStats.map((s, d) => <td key={d} className="muted">μ={s.m.toFixed(2)}</td>)}</tr>
                  <tr>{colStats.map((s, d) => <td key={d} className="muted">σ={s.sd.toFixed(2)}</td>)}</tr>
                </>
              )}
            </tbody>
          </table>
        </div>
        <div>
          <div className="muted">Ẑ = (Z − μ) / σ</div>
          <table className="data">
            <tbody>
              {norm.map((r, n) => (
                <tr key={n}>{r.map((v, d) => <td key={d} style={{ boxShadow: `inset 4px 0 0 ${tint(n, d)}`, textAlign: "right" }}>{v.toFixed(2)}</td>)}</tr>
              ))}
            </tbody>
          </table>
          <p className="readout">{mode === "batch" ? <>NumPy: <b>Z.mean(axis=0)</b>, <b>Z.std(axis=0)</b> → shape (3,)</> : <>NumPy: <b>Z.mean(axis=1)</b>, <b>Z.std(axis=1)</b> → shape (4,)</>}</p>
        </div>
      </div>
    </Widget>
  );
}

// ---------- Dropout ----------

const DROP_H = [0.5, 0.2, 0.0, 1.4, 0.9, 0.3];

export function DropoutWidget() {
  const [p, setP] = useState(0.5);
  const [seed, setSeed] = useState(1);
  const [train, setTrain] = useState(true);
  const r = rng(seed * 7919);
  const mask = DROP_H.map(() => (r() < p ? 0 : 1));
  const out = DROP_H.map((h, i) => (train ? (mask[i] * h) / (1 - p) : h));
  const meanIn = DROP_H.reduce((a, b) => a + b, 0) / DROP_H.length;
  // Expected value of each output equals its input: (1 − p)·h/(1 − p) = h.
  return (
    <Widget
      title="Dropout: random masks in training, nothing at inference"
      caption="Each training step draws a new mask. Dividing by (1 − p) keeps each unit's expected value equal to its input, so nothing needs rescaling at inference."
    >
      <div className="toggle-row">
        <button className={train ? "" : "off"} onClick={() => setTrain(true)}>model.train()</button>
        <button className={train ? "off" : ""} onClick={() => setTrain(false)}>model.eval()</button>
        {train && <button className="ghost" onClick={() => setSeed(seed + 1)}>New training step (new mask)</button>}
      </div>
      <Slider label={<>drop probability <M>p</M></>} value={p} min={0.1} max={0.9} step={0.1} onChange={setP} format={(v) => v.toFixed(1)} />
      <table className="data">
        <thead><tr><th></th>{DROP_H.map((_, i) => <th key={i}>unit {i + 1}</th>)}</tr></thead>
        <tbody>
          <tr><td>h</td>{DROP_H.map((h, i) => <td key={i}>{h}</td>)}</tr>
          {train && <tr><td>mask m</td>{mask.map((m, i) => <td key={i} style={{ color: m ? "var(--c3)" : "var(--mistake)" }}>{m}</td>)}</tr>}
          <tr><td>{train ? "m ⊙ h / (1 − p)" : "output (all units)"}</td>{out.map((v, i) => <td key={i}><b>{+v.toFixed(2)}</b></td>)}</tr>
        </tbody>
      </table>
      <div className="readout">mean of h = {meanIn.toFixed(2)} · mean of this output = <b>{(out.reduce((a, b) => a + b, 0) / out.length).toFixed(2)}</b> (equal on average over many masks)</div>
    </Widget>
  );
}

// ---------- Gradient clipping ----------

export function ClipWidget() {
  const [gx, setGx] = useState(3);
  const [gy, setGy] = useState(4);
  const [C, setC] = useState(2.5);
  const n = Math.hypot(gx, gy);
  const f = Math.min(1, C / n);
  const circle = Array.from({ length: 101 }, (_, i): [number, number] => [C * Math.cos((i / 50) * Math.PI), C * Math.sin((i / 50) * Math.PI)]);
  return (
    <Widget title="Gradient clipping keeps the direction, caps the length" caption="Dashed circle: radius C. Orange: original g. Green: clipped g.">
      <div className="widget-row">
        <Chart xDomain={[-6, 6]} yDomain={[-6, 6]} width={300} height={300} xLabel="g₁" yLabel="g₂">
          {(s) => (
            <>
              <path d={polyline(circle, s)} stroke="var(--muted)" strokeDasharray="5 4" fill="none" />
              <line x1={s.x(0)} y1={s.y(0)} x2={s.x(gx)} y2={s.y(gy)} stroke="var(--c2)" strokeWidth={3} />
              <line x1={s.x(0)} y1={s.y(0)} x2={s.x(gx * f)} y2={s.y(gy * f)} stroke="var(--c3)" strokeWidth={5} opacity={0.8} />
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>{"g_1"}</M>} value={gx} min={-5} max={5} step={0.5} onChange={setGx} />
          <Slider label={<M>{"g_2"}</M>} value={gy} min={-5} max={5} step={0.5} onChange={setGy} />
          <Slider label="threshold C" value={C} min={0.5} max={5} step={0.5} onChange={setC} />
          <div className="readout">
            ‖g‖ = {n.toFixed(2)}<br />
            factor min(1, C/‖g‖) = <b>{f.toFixed(3)}</b><br />
            clipped g = (<b>{(gx * f).toFixed(2)}</b>, <b>{(gy * f).toFixed(2)}</b>)
          </div>
        </div>
      </div>
    </Widget>
  );
}
