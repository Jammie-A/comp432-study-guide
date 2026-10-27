import { useMemo, useState } from "react";
import { Chart, Slider, Widget, fnPath, polyline } from "../components/chart";
import { M } from "../components/ui";
import { sigmoid } from "./math";

// ---------- Dot product ----------

export function DotProductWidget() {
  const [angA, setAngA] = useState(63);
  const [lenA, setLenA] = useState(2.2);
  const b = [3, 1];
  const a = [lenA * Math.cos((angA * Math.PI) / 180), lenA * Math.sin((angA * Math.PI) / 180)];
  const d = a[0] * b[0] + a[1] * b[1];
  const nb = Math.hypot(b[0], b[1]);
  const cos = d / (lenA * nb);
  const sign = Math.abs(d) < 0.05 ? "≈ 0: orthogonal" : d > 0 ? "> 0: point the same way" : "< 0: point opposite ways";
  return (
    <Widget title="The dot product measures alignment" caption="b = (3, 1) is fixed. Rotate and stretch a.">
      <div className="widget-row">
        <Chart xDomain={[-3.5, 3.5]} yDomain={[-3, 3]} xLabel="" yLabel="">
          {(s) => (
            <>
              <line x1={s.x(0)} y1={s.y(0)} x2={s.x(b[0])} y2={s.y(b[1])} stroke="var(--c2)" strokeWidth={3} />
              <line x1={s.x(0)} y1={s.y(0)} x2={s.x(a[0])} y2={s.y(a[1])} stroke="var(--c1)" strokeWidth={3} />
              <text x={s.x(b[0]) + 4} y={s.y(b[1])}>b</text>
              <text x={s.x(a[0]) + 4} y={s.y(a[1])}>a</text>
            </>
          )}
        </Chart>
        <div>
          <Slider label="angle of a (°)" value={angA} min={-180} max={180} step={1} onChange={setAngA} />
          <Slider label="length ‖a‖" value={lenA} min={0.2} max={3} step={0.1} onChange={setLenA} format={(v) => v.toFixed(1)} />
          <div className="readout">
            a = ({a[0].toFixed(2)}, {a[1].toFixed(2)})<br />
            aᵀb = {a[0].toFixed(2)}·3 + {a[1].toFixed(2)}·1 = <b>{d.toFixed(2)}</b><br />
            ‖a‖‖b‖cos∠ = {lenA.toFixed(2)}·{nb.toFixed(2)}·{cos.toFixed(2)} = <b>{d.toFixed(2)}</b><br />
            → {sign}
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ---------- Linear regression loss bowl with GD path ----------

const XS = [1, 3, 5];
const YS = [0.2, 1.1, 1.7];

export function LossBowlWidget() {
  const [eta, setEta] = useState(0.05);
  const [steps, setSteps] = useState(30);
  const [center, setCenter] = useState(false);
  const xs = center ? XS.map((x) => x - 3) : XS;
  const N = 3;
  // A = X̃ᵀX̃ / N, c = X̃ᵀy / N
  const A = [
    [xs.reduce((s, x) => s + x * x, 0) / N, xs.reduce((s, x) => s + x, 0) / N],
    [xs.reduce((s, x) => s + x, 0) / N, 1],
  ];
  const c = [xs.reduce((s, x, i) => s + x * YS[i], 0) / N, YS.reduce((s, y) => s + y, 0) / N];
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const star = [(A[1][1] * c[0] - A[0][1] * c[1]) / det, (A[0][0] * c[1] - A[1][0] * c[0]) / det];
  const J = (w: number, b: number) => xs.reduce((s, x, i) => s + (w * x + b - YS[i]) ** 2, 0) / N;
  const Jstar = J(star[0], star[1]);
  // eigen-decomposition of the symmetric 2x2 A (Hessian = 2A)
  const tr = A[0][0] + A[1][1];
  const disc = Math.sqrt((A[0][0] - A[1][1]) ** 2 / 4 + A[0][1] ** 2);
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  const v1 = Math.abs(A[0][1]) < 1e-12 ? [1, 0] : norm([l1 - A[1][1], A[0][1]]);
  const v2 = [-v1[1], v1[0]];
  const kappa = l1 / l2;
  const start = [0, 1];
  const path: [number, number][] = [[start[0], start[1]]];
  let t = start;
  for (let i = 0; i < steps; i++) {
    const g = [
      (2 / N) * xs.reduce((s, x, k) => s + (t[0] * x + t[1] - YS[k]) * x, 0),
      (2 / N) * xs.reduce((s, x, k) => s + (t[0] * x + t[1] - YS[k]), 0),
    ];
    t = [t[0] - eta * g[0], t[1] - eta * g[1]];
    path.push([t[0], t[1]]);
  }
  const levels = [0.01, 0.05, 0.2, 0.5, 1, 2];
  const ell = (lev: number): [number, number][] => {
    const d = lev - Jstar;
    if (d <= 0) return [];
    return Array.from({ length: 121 }, (_, i) => {
      const ph = (i / 120) * 2 * Math.PI;
      const r1 = Math.sqrt(d / l1) * Math.cos(ph), r2 = Math.sqrt(d / l2) * Math.sin(ph);
      return [star[0] + r1 * v1[0] + r2 * v2[0], star[1] + r1 * v1[1] + r2 * v2[1]];
    });
  };
  const end = path[path.length - 1];
  const diverged = !isFinite(end[0]) || Math.hypot(end[0] - star[0], end[1] - star[1]) > 5;
  return (
    <Widget
      title="The MSE loss over (w, b) is a bowl"
      caption="Contours of J(w, b) for the 3-flower mini dataset. Star: closed-form θ*. Orange: gradient descent from (w, b) = (0, 1)."
    >
      <div className="widget-row">
        <Chart xDomain={[-0.6, 1.2]} yDomain={[-1.6, 1.6]} xLabel="w" yLabel="b">
          {(s) => (
            <>
              {levels.map((l) => <path key={l} d={polyline(ell(l), s)} stroke="var(--muted)" strokeOpacity={0.45} strokeWidth={1.2} fill="none" />)}
              {!diverged && <path d={polyline(path, s)} stroke="var(--c2)" strokeWidth={1.8} fill="none" />}
              {!diverged && path.map((p, i) => <circle key={i} cx={s.x(p[0])} cy={s.y(p[1])} r={2.2} fill="var(--c2)" />)}
              <text x={s.x(star[0]) - 7} y={s.y(star[1]) + 6} style={{ fontSize: 18, fill: "var(--c3)" }}>★</text>
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>\eta</M>} value={eta} min={0.005} max={0.09} step={0.005} onChange={setEta} format={(v) => v.toFixed(3)} />
          <Slider label="GD steps" value={steps} min={1} max={300} step={1} onChange={setSteps} />
          <label className="slider" style={{ gridTemplateColumns: "auto 1fr" }}>
            <input type="checkbox" checked={center} onChange={(e) => setCenter(e.target.checked)} style={{ width: "auto" }} />
            <span>pre-process: center x (subtract its mean, 3)</span>
          </label>
          <div className="readout">
            θ* = ({star[0].toFixed(3)}, {star[1].toFixed(3)}), J(θ*) = {Jstar.toFixed(3)}<br />
            after {steps} steps: {diverged ? <b>diverged!</b> : <>({end[0].toFixed(3)}, {end[1].toFixed(3)}), J = <b>{J(end[0], end[1]).toFixed(4)}</b></>}<br />
            Hessian eigenvalues: {(2 * l1).toFixed(2)}, {(2 * l2).toFixed(2)}<br />
            condition number κ = <b>{kappa.toFixed(1)}</b>
          </div>
          <p className="muted">GD is stable only if η &lt; 2/λ_max = {(2 / (2 * l1)).toFixed(3)}. Centering x makes the bowl rounder (smaller κ), so GD converges in far fewer steps.</p>
        </div>
      </div>
    </Widget>
  );
}

function norm(v: number[]) {
  const n = Math.hypot(v[0], v[1]);
  return [v[0] / n, v[1] / n];
}

// ---------- Sigmoid ----------

export function SigmoidWidget() {
  const [z, setZ] = useState(0.3);
  const s_ = sigmoid(z);
  return (
    <Widget title="The sigmoid squashes any z into (0, 1)" caption="Dashed: its derivative σ(z)(1 − σ(z)), largest (0.25) at z = 0.">
      <div className="widget-row">
        <Chart xDomain={[-6, 6]} yDomain={[0, 1]} xLabel="z" yLabel="σ(z)">
          {(s) => (
            <>
              <path d={fnPath(sigmoid, [-6, 6], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              <path d={fnPath((v) => sigmoid(v) * (1 - sigmoid(v)), [-6, 6], s)} stroke="var(--c3)" strokeDasharray="5 4" strokeWidth={2} fill="none" />
              <line x1={s.x(-6)} x2={s.x(6)} y1={s.y(0.5)} y2={s.y(0.5)} stroke="var(--muted)" strokeDasharray="2 3" />
              <circle cx={s.x(z)} cy={s.y(s_)} r={6} fill="var(--c2)" />
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>z</M>} value={z} min={-6} max={6} step={0.1} onChange={setZ} format={(v) => v.toFixed(1)} />
          <div className="readout">
            σ({z.toFixed(1)}) = 1 / (1 + e^{(-z).toFixed(1)}) = <b>{s_.toFixed(3)}</b><br />
            σ′(z) = {s_.toFixed(3)} · {(1 - s_).toFixed(3)} = <b>{(s_ * (1 - s_)).toFixed(3)}</b><br />
            prediction: <b>{s_ > 0.5 ? "class 1 (virginica)" : "class 0 (versicolor)"}</b>
          </div>
        </div>
      </div>
    </Widget>
  );
}

// ---------- Logistic regression decision boundary ----------

// Illustrative iris-like points (petal length, petal width); 0 = versicolor, 1 = virginica.
const LOG_PTS: [number, number, number][] = [
  [4.0, 1.3, 0], [4.5, 1.5, 0], [4.7, 1.4, 0], [3.9, 1.1, 0], [4.4, 1.4, 0], [4.1, 1.0, 0], [4.6, 1.3, 0],
  [3.5, 1.0, 0], [4.2, 1.2, 0], [4.8, 1.8, 0], [5.0, 1.7, 0], [3.3, 1.0, 0],
  [6.0, 2.5, 1], [5.1, 1.9, 1], [5.9, 2.1, 1], [5.6, 1.8, 1], [5.8, 2.2, 1], [6.6, 2.1, 1], [4.9, 1.8, 1],
  [5.5, 2.1, 1], [6.1, 2.3, 1], [5.3, 1.9, 1], [5.0, 1.5, 1], [5.7, 2.3, 1],
];

export function LogisticWidget() {
  const [w1, setW1] = useState(1);
  const [w2, setW2] = useState(1);
  const [b, setB] = useState(-5);
  const stats = useMemo(() => {
    let J = 0, correct = 0;
    for (const [x1, x2, y] of LOG_PTS) {
      const p = sigmoid(w1 * x1 + w2 * x2 + b);
      const pc = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
      J += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
      if ((p > 0.5 ? 1 : 0) === y) correct++;
    }
    return { J: J / LOG_PTS.length, acc: correct / LOG_PTS.length };
  }, [w1, w2, b]);
  function train() {
    let t = [w1, w2, b];
    const N = LOG_PTS.length;
    for (let e = 0; e < 1000; e++) {
      const g = [0, 0, 0];
      for (const [x1, x2, y] of LOG_PTS) {
        const err = sigmoid(t[0] * x1 + t[1] * x2 + t[2]) - y;
        g[0] += (err * x1) / N; g[1] += (err * x2) / N; g[2] += err / N;
      }
      t = t.map((v, k) => v - 0.2 * g[k]);
    }
    setW1(+t[0].toFixed(2)); setW2(+t[1].toFixed(2)); setB(+t[2].toFixed(2));
  }
  // boundary: w1 x1 + w2 x2 + b = 0 → x2 = -(w1 x1 + b)/w2
  return (
    <Widget
      title="Logistic regression: the decision boundary θᵀx̃ = 0"
      caption="Illustrative iris-like points (not the real dataset). Blue: versicolor (y = 0), orange: virginica (y = 1). Shading shows ŷ = σ(θᵀx̃)."
    >
      <div className="widget-row">
        <Chart xDomain={[3, 7]} yDomain={[0.8, 2.7]} xLabel="petal length x₁ [cm]" yLabel="petal width x₂ [cm]">
          {(s) => (
            <>
              {Array.from({ length: 20 * 12 }, (_, i) => {
                const gx = 3 + ((i % 20) + 0.5) * 0.2, gy = 0.8 + (Math.floor(i / 20) + 0.5) * (1.9 / 12);
                const p = sigmoid(w1 * gx + w2 * gy + b);
                return <rect key={i} x={s.x(gx - 0.1)} y={s.y(gy + 1.9 / 24)} width={s.x(0.2) - s.x(0)} height={s.y(0) - s.y(1.9 / 12)} fill={p > 0.5 ? "var(--c2)" : "var(--c1)"} opacity={0.08 + 0.25 * Math.abs(p - 0.5)} />;
              })}
              {Math.abs(w2) > 1e-6 && <path d={fnPath((x) => -(w1 * x + b) / w2, [3, 7], s)} stroke="var(--text)" strokeWidth={2} fill="none" />}
              {LOG_PTS.map(([x1, x2, y], i) => <circle key={i} cx={s.x(x1)} cy={s.y(x2)} r={4.5} fill={y ? "var(--c2)" : "var(--c1)"} stroke="var(--panel)" />)}
              <text x={s.x(4.0) + 6} y={s.y(1.3) + 14} style={{ fontSize: 11 }}>x_n</text>
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>{"w_1"}</M>} value={w1} min={-3} max={6} step={0.05} onChange={setW1} format={(v) => v.toFixed(2)} />
          <Slider label={<M>{"w_2"}</M>} value={w2} min={-3} max={12} step={0.05} onChange={setW2} format={(v) => v.toFixed(2)} />
          <Slider label={<M>b</M>} value={b} min={-45} max={5} step={0.1} onChange={setB} format={(v) => v.toFixed(1)} />
          <div className="readout">
            boundary: {w1.toFixed(2)}·x₁ + {w2.toFixed(2)}·x₂ + ({b.toFixed(1)}) = 0<br />
            BCE = <b>{stats.J.toFixed(3)}</b>, accuracy = <b>{(stats.acc * 100).toFixed(0)}%</b>
          </div>
          <div className="toggle-row">
            <button onClick={train}>Run 1000 GD steps (η = 0.2)</button>
            <button className="ghost" onClick={() => { setW1(1); setW2(1); setB(-5); }}>Reset to slide θ</button>
          </div>
          <p className="muted">The weights (w₁, w₂) are the normal vector of the boundary line; b shifts it.</p>
        </div>
      </div>
    </Widget>
  );
}

// ---------- Softmax ----------

export function SoftmaxWidget() {
  const [z, setZ] = useState([1, 0.3, -2]);
  const names = ["setosa", "versicolor", "virginica"];
  const e = z.map(Math.exp);
  const S = e.reduce((a, b) => a + b, 0);
  const p = e.map((v) => v / S);
  const set = (k: number) => (v: number) => setZ(z.map((x, i) => (i === k ? v : x)));
  const colors = ["var(--c3)", "var(--c1)", "var(--c2)"];
  return (
    <Widget title="Softmax turns K logits into K probabilities" caption="True class here: versicolor (y = (0, 1, 0)), so CCE = −ln ŷ₂.">
      <div className="widget-row">
        <div>
          {z.map((v, k) => (
            <Slider key={k} label={<>z<sub>{k + 1}</sub> ({names[k]})</>} value={v} min={-4} max={4} step={0.1} onChange={set(k)} format={(x) => x.toFixed(1)} />
          ))}
          <table className="data">
            <thead><tr><th>class</th><th>z</th><th>e^z</th><th>ŷ = e^z / Σ</th></tr></thead>
            <tbody>
              {z.map((v, k) => <tr key={k}><td>{names[k]}</td><td>{v.toFixed(1)}</td><td>{e[k].toFixed(2)}</td><td><b>{p[k].toFixed(2)}</b></td></tr>)}
              <tr><td colSpan={2}>sum</td><td>{S.toFixed(2)}</td><td>{p.reduce((a, b) => a + b, 0).toFixed(2)}</td></tr>
            </tbody>
          </table>
          <div className="readout">CCE = −ln({p[1].toFixed(2)}) = <b>{(-Math.log(p[1])).toFixed(2)}</b> · prediction: <b>{names[p.indexOf(Math.max(...p))]}</b></div>
        </div>
        <svg viewBox="0 0 300 200" className="chart">
          {p.map((v, k) => (
            <g key={k}>
              <rect x={30 + k * 90} y={170 - v * 150} width={60} height={v * 150} fill={colors[k]} rx={4} />
              <text x={60 + k * 90} y={165 - v * 150} textAnchor="middle" style={{ fontSize: 13 }}>{v.toFixed(2)}</text>
              <text x={60 + k * 90} y={190} textAnchor="middle" style={{ fontSize: 12 }}>{names[k]}</text>
            </g>
          ))}
        </svg>
      </div>
    </Widget>
  );
}
