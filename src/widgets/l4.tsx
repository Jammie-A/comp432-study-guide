import { useState } from "react";
import { Chart, Slider, Widget, fnPath } from "../components/chart";
import { M } from "../components/ui";
import { sigmoid } from "./math";

// ---------- Activation functions ----------

const ACTS = {
  sigmoid: { f: sigmoid, df: (z: number) => sigmoid(z) * (1 - sigmoid(z)), y: [-0.2, 1.2] as [number, number] },
  tanh: { f: Math.tanh, df: (z: number) => 1 - Math.tanh(z) ** 2, y: [-1.2, 1.2] as [number, number] },
  ReLU: { f: (z: number) => Math.max(0, z), df: (z: number) => (z > 0 ? 1 : 0), y: [-0.5, 4] as [number, number] },
};
type Act = keyof typeof ACTS;

export function ActivationWidget() {
  const [act, setAct] = useState<Act>("ReLU");
  const [z, setZ] = useState(1);
  const a = ACTS[act];
  return (
    <Widget title="Activation functions and their derivatives" caption="Solid: f(z). Dashed: f′(z). Where f′ ≈ 0 (saturation), gradients vanish and learning stalls.">
      <div className="toggle-row">
        {(Object.keys(ACTS) as Act[]).map((k) => (
          <button key={k} className={k === act ? "" : "off"} onClick={() => setAct(k)}>{k}</button>
        ))}
      </div>
      <div className="widget-row">
        <Chart xDomain={[-4, 4]} yDomain={a.y} xLabel="z" yLabel="">
          {(s) => (
            <>
              <path d={fnPath(a.f, [-4, 4], s, 400)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              <path d={fnPath(a.df, [-4, 4], s, 400)} stroke="var(--c3)" strokeDasharray="5 4" strokeWidth={2} fill="none" />
              <circle cx={s.x(z)} cy={s.y(a.f(z))} r={6} fill="var(--c2)" />
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>z</M>} value={z} min={-4} max={4} step={0.1} onChange={setZ} format={(v) => v.toFixed(1)} />
          <div className="readout">
            f({z.toFixed(1)}) = <b>{a.f(z).toFixed(3)}</b><br />
            f′({z.toFixed(1)}) = <b>{a.df(z).toFixed(3)}</b>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ---------- Universal approximation with sigmoid pulses ----------

export function PulseWidget() {
  const [n, setN] = useState(5);
  const [k, setK] = useState(500);
  const target = (x: number) => Math.sin(2 * Math.PI * x);
  // average of sin(2πx) over [a, b]
  const avg = (a: number, b: number) => (Math.cos(2 * Math.PI * a) - Math.cos(2 * Math.PI * b)) / (2 * Math.PI * (b - a));
  const heights = Array.from({ length: n }, (_, i) => avg(i / n, (i + 1) / n));
  const f = (x: number) =>
    heights.reduce((s, hgt, i) => s + hgt * (sigmoid(k * (x - i / n)) - sigmoid(k * (x - (i + 1) / n))), 0);
  return (
    <Widget
      title="Universal approximation: sum of sigmoid pulses"
      caption="Each pulse = 2 hidden neurons: h₁ = σ(k(x − a)) switches on at a, h₂ = σ(k(x − b)) at b, output weight ±(average of f on [a, b])."
    >
      <div className="widget-row">
        <Chart xDomain={[0, 1]} yDomain={[-1.3, 1.3]} xLabel="x" yLabel="y">
          {(s) => (
            <>
              <path d={fnPath(target, [0, 1], s)} stroke="var(--muted)" strokeDasharray="5 4" fill="none" />
              <path d={fnPath(f, [0, 1], s, 600)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
            </>
          )}
        </Chart>
        <div>
          <Slider label="intervals" value={n} min={1} max={45} step={1} onChange={setN} />
          <Slider label={<>steepness <M>k</M></>} value={k} min={1} max={500} step={1} onChange={setK} />
          <div className="readout">hidden neurons H = <b>{2 * n}</b></div>
          {n === 5 && <p className="muted">Pulse heights (averages): {heights.map((h) => h.toFixed(2)).join(", ")} — the slide's 0.55, 0.89, 0, −0.89, −0.55.</p>}
        </div>
      </div>
    </Widget>
  );
}

// ---------- The non-convexity counterexample ----------

export function NonConvexWidget() {
  const [t, setT] = useState(0.5);
  const relu = (v: number) => Math.max(0, v);
  // θ = (ν1, ν2, w1, w2), x = 1, y = 1. θ = (1,0,1,0), θ' = (0,1,0,1)
  const J = (tt: number) => {
    const th = [tt, 1 - tt, tt, 1 - tt]; // tθ + (1 − t)θ'
    const yhat = th[2] * relu(th[0]) + th[3] * relu(th[1]);
    return (yhat - 1) ** 2;
  };
  return (
    <Widget
      title="A tiny ReLU network is not convex"
      caption="Loss along the straight line tθ + (1 − t)θ′ between two perfect solutions. The chord (dashed) lies at 0, but the loss in between is above it: that breaks convexity."
    >
      <div className="widget-row">
        <Chart xDomain={[0, 1]} yDomain={[-0.05, 0.4]} xLabel="t  (t = 0: θ′, t = 1: θ)" yLabel="J">
          {(s) => (
            <>
              <path d={fnPath(J, [0, 1], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              <line x1={s.x(0)} x2={s.x(1)} y1={s.y(0)} y2={s.y(0)} stroke="var(--c2)" strokeDasharray="5 4" strokeWidth={2} />
              <circle cx={s.x(t)} cy={s.y(J(t))} r={6} fill="var(--c2)" />
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>t</M>} value={t} min={0} max={1} step={0.01} onChange={setT} />
          <div className="readout">
            θ_t = ({t.toFixed(2)}, {(1 - t).toFixed(2)}, {t.toFixed(2)}, {(1 - t).toFixed(2)})<br />
            ŷ = {t.toFixed(2)}·ReLU({t.toFixed(2)}) + {(1 - t).toFixed(2)}·ReLU({(1 - t).toFixed(2)}) = <b>{(t * t + (1 - t) ** 2).toFixed(3)}</b><br />
            J = (ŷ − 1)² = <b>{J(t).toFixed(3)}</b>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ---------- Scalar backprop graph ----------

/** Draws x → f(1) → f(2) → J with values revealed up to `stage`. */
export function BackpropGraph({ stage }: { stage: number }) {
  // stage: 0 nothing, 1 h1, 2 h2, 3 J, 4 dJ/dh2, 5 dJ/dw2, 6 dJ/dh1, 7 dJ/dw1
  const node = (x: number, label: string, val: string | null, show: boolean) => (
    <g>
      <circle cx={x} cy={60} r={24} fill="var(--panel)" stroke="var(--accent)" strokeWidth={2} />
      <text x={x} y={65} textAnchor="middle" style={{ fontSize: 14 }}>{label}</text>
      {show && val && <text x={x} y={104} textAnchor="middle" style={{ fontSize: 13, fill: "var(--c1)", fontWeight: 600 }}>{val}</text>}
    </g>
  );
  const grad = (x: number, y: number, text: string, show: boolean) =>
    show && <text x={x} y={y} textAnchor="middle" style={{ fontSize: 13, fill: "var(--c2)", fontWeight: 600 }}>{text}</text>;
  return (
    <svg viewBox="0 0 560 190" className="chart" style={{ maxWidth: 560 }}>
      {[[54, 106], [154, 206], [254, 306], [354, 406]].map(([a, b], i) => (
        <line key={i} x1={a + 6} x2={b - 6} y1={60} y2={60} stroke="var(--muted)" strokeWidth={1.5} markerEnd="url(#arr)" />
      ))}
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
        </marker>
      </defs>
      {node(30, "x", "x = 2", true)}
      {node(130, "×w₁", "h₁ = 2", stage >= 1)}
      {node(230, "×w₂", "h₂ = 6", stage >= 2)}
      {node(330, "(·−y)²", "J = 4", stage >= 3)}
      <text x={130} y={20} textAnchor="middle" style={{ fontSize: 12 }}>w₁ = 1</text>
      <text x={230} y={20} textAnchor="middle" style={{ fontSize: 12 }}>w₂ = 3</text>
      <text x={330} y={20} textAnchor="middle" style={{ fontSize: 12 }}>y = 4</text>
      {grad(280, 140, "dJ/dh₂ = 4", stage >= 4)}
      {grad(230, 165, "∂J/∂w₂ = 8", stage >= 5)}
      {grad(180, 140, "dJ/dh₁ = 12", stage >= 6)}
      {grad(130, 165, "∂J/∂w₁ = 24", stage >= 7)}
      <text x={470} y={55} style={{ fontSize: 12, fill: "var(--c1)" }}>forward →</text>
      <text x={470} y={145} style={{ fontSize: 12, fill: "var(--c2)" }}>← backward</text>
    </svg>
  );
}

export function BackpropWidget() {
  const [stage, setStage] = useState(0);
  const notes = [
    "Start: inputs x = 2, y = 4, weights w₁ = 1, w₂ = 3.",
    "Forward: h₁ = w₁·x = 1·2 = 2.",
    "Forward: h₂ = w₂·h₁ = 3·2 = 6.",
    "Forward: J = (h₂ − y)² = (6 − 4)² = 4.",
    "Backward: dJ/dh₂ = 2(h₂ − y) = 2·2 = 4.",
    "Backward: ∂J/∂w₂ = dJ/dh₂ · ∂h₂/∂w₂ = 4 · h₁ = 4·2 = 8.",
    "Backward: dJ/dh₁ = dJ/dh₂ · dh₂/dh₁ = 4 · w₂ = 4·3 = 12. (We reuse dJ/dh₂, no recomputation.)",
    "Backward: ∂J/∂w₁ = dJ/dh₁ · ∂h₁/∂w₁ = 12 · x = 12·2 = 24. Done!",
  ];
  return (
    <Widget title="Backpropagation, one operation at a time" caption={notes[stage]}>
      <BackpropGraph stage={stage} />
      <div className="toggle-row">
        <button disabled={stage >= 7} onClick={() => setStage(stage + 1)}>Next ({stage}/7)</button>
        <button className="ghost" onClick={() => setStage(0)}>Restart</button>
      </div>
    </Widget>
  );
}
