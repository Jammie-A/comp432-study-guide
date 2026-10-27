import { useMemo, useState } from "react";
import { Chart, Slider, Widget, fnPath, polyline } from "../components/chart";
import { M } from "../components/ui";
import { dot, gauss, polyFeatures, ridge, rng } from "./math";

// ---------- Capacity / overfitting (also used in Lecture 4 with λ) ----------

function makeSine(n: number, seed: number, noise = 0.25) {
  const r = rng(seed);
  const xs = Array.from({ length: n }, (_, i) => (n === 1 ? 0.5 : (i + 0.5 * r()) / n));
  return xs.map((x) => ({ x, y: Math.sin(2 * Math.PI * x) + noise * gauss(r) }));
}

const LAMBDAS = [0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1];

export function CapacityWidget({ withLambda = false, showWeights = false }: { withLambda?: boolean; showWeights?: boolean }) {
  const [M_, setM] = useState(withLambda ? 9 : 3);
  const [N, setN] = useState(10);
  const [li, setLi] = useState(0);
  const lambda = LAMBDAS[li];
  const train = useMemo(() => makeSine(N, 7 + N), [N]);
  const test = useMemo(() => makeSine(100, 99), []);
  const theta = useMemo(
    () => ridge(train.map((p) => polyFeatures(p.x, M_)), train.map((p) => p.y), Math.max(lambda, 1e-13)),
    [train, M_, lambda],
  );
  const f = (x: number) => dot(polyFeatures(x, M_), theta);
  const mse = (d: { x: number; y: number }[]) => d.reduce((s, p) => s + (f(p.x) - p.y) ** 2, 0) / d.length;
  const tr = mse(train);
  const te = mse(test);
  const verdict = te > 0.3 && tr < 0.05 ? "overfitting" : tr > 0.2 ? "underfitting" : "good fit";
  return (
    <Widget
      title={withLambda ? "Polynomial fit with L2 regularization" : "Model capacity: polynomial degree"}
      caption="Black dots: training data (sin(2πx) + noise). Dashed: the true function. Blue: the fitted model."
    >
      <div className="widget-row">
        <Chart xDomain={[0, 1]} yDomain={[-1.8, 1.8]} xLabel="x" yLabel="y">
          {(s) => (
            <>
              <path d={fnPath((x) => Math.sin(2 * Math.PI * x), [0, 1], s)} stroke="var(--muted)" strokeDasharray="5 4" fill="none" />
              <path d={fnPath(f, [0, 1], s, 400)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              {train.map((p, i) => (
                <circle key={i} cx={s.x(p.x)} cy={s.y(p.y)} r={4} fill="var(--text)" />
              ))}
            </>
          )}
        </Chart>
        <div>
          <Slider label={<>degree <M>M</M></>} value={M_} min={1} max={9} step={1} onChange={setM} />
          <Slider label={<>samples <M>N</M></>} value={N} min={10} max={100} step={10} onChange={setN} />
          {withLambda && (
            <Slider label={<M>\lambda</M>} value={li} min={0} max={LAMBDAS.length - 1} step={1} onChange={setLi} format={(i) => (LAMBDAS[i] === 0 ? "0" : LAMBDAS[i].toExponential(0))} />
          )}
          <div className="readout">
            parameters P = {M_ + 1}<br />
            training MSE = <b>{tr.toFixed(3)}</b><br />
            test MSE = <b>{te.toFixed(3)}</b><br />
            → <b>{verdict}</b>
          </div>
          {showWeights && (
            <table className="data">
              <tbody>
                {theta.map((t, i) => (
                  <tr key={i}>
                    <td>{i < M_ ? `w${i + 1}` : "b"}</td>
                    <td style={{ textAlign: "right" }}>{Math.abs(t) > 1000 ? t.toExponential(1) : t.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </Widget>
  );
}

// ---------- 1-D gradient descent and the learning rate ----------

export function GD1DWidget() {
  const J = (t: number) => 1.5 * t * t;
  const dJ = (t: number) => 3 * t;
  const [eta, setEta] = useState(0.1);
  const [t0, setT0] = useState(-1.8);
  const steps = 12;
  const path: number[] = [t0];
  for (let i = 0; i < steps; i++) path.push(path[i] - eta * dJ(path[i]));
  const diverged = Math.abs(path[steps]) > Math.abs(t0);
  const label = diverged ? "too large: diverges" : eta < 0.06 ? "too small: slow" : "about right";
  return (
    <Widget
      title="Gradient descent on J(θ) = 1.5θ²"
      caption={<>Each orange dot is one update <M>{"\\theta \\leftarrow \\theta - \\eta\\, J'(\\theta)"}</M> with <M>{"J'(\\theta)=3\\theta"}</M>. For this bowl, any η above 2/3 makes the steps grow and GD diverges.</>}
    >
      <div className="widget-row">
        <Chart xDomain={[-2.5, 2.5]} yDomain={[0, 10]} xLabel="θ" yLabel="J(θ)">
          {(s) => (
            <>
              <path d={fnPath(J, [-2.5, 2.5], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              <path d={polyline(path.map((t) => [t, J(t)]), s)} stroke="var(--c2)" strokeWidth={1.5} fill="none" />
              {path.map((t, i) => (
                <circle key={i} cx={s.x(t)} cy={s.y(J(t))} r={i ? 4 : 6} fill="var(--c2)" opacity={1 - i / (steps + 4)} />
              ))}
            </>
          )}
        </Chart>
        <div>
          <Slider label={<>learning rate <M>\eta</M></>} value={eta} min={0.01} max={0.75} step={0.01} onChange={setEta} />
          <Slider label={<>start <M>{"\\theta_0"}</M></>} value={t0} min={-2.2} max={2.2} step={0.1} onChange={setT0} format={(v) => v.toFixed(1)} />
          <div className="readout">→ <b>{label}</b></div>
          <table className="data">
            <thead><tr><th>t</th><th>θ_t</th><th>J′(θ_t)</th><th>θ_t+1</th></tr></thead>
            <tbody>
              {path.slice(0, 4).map((t, i) => (
                <tr key={i}><td>{i}</td><td>{t.toFixed(3)}</td><td>{dJ(t).toFixed(3)}</td><td>{path[i + 1].toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Widget>
  );
}

// ---------- 2-D narrow valley: GD vs momentum vs Adam ----------

type Method = "gd" | "momentum" | "adam";

function runOptimizer(method: Method, h: [number, number], eta: number, rho: number, steps: number) {
  let t = [4, 1];
  let v = [0, 0];
  let s = [0, 0];
  const pts: [number, number][] = [[t[0], t[1]]];
  const losses: number[] = [];
  const loss = (t: number[]) => 0.5 * (h[0] * t[0] * t[0] + h[1] * t[1] * t[1]);
  for (let i = 0; i < steps; i++) {
    losses.push(loss(t));
    const g = [h[0] * t[0], h[1] * t[1]];
    if (method === "gd") t = t.map((ti, k) => ti - eta * g[k]);
    else if (method === "momentum") {
      v = v.map((vk, k) => rho * vk - eta * g[k]);
      t = t.map((ti, k) => ti + v[k]);
    } else {
      v = v.map((vk, k) => 0.9 * vk + 0.1 * g[k]);
      s = s.map((sk, k) => 0.999 * sk + 0.001 * g[k] * g[k]);
      t = t.map((ti, k) => ti - (eta / (1e-8 + Math.sqrt(s[k]))) * v[k]);
    }
    pts.push([t[0], t[1]]);
  }
  losses.push(loss(t));
  return { pts, losses };
}

const methodColor: Record<Method, string> = { gd: "var(--c2)", momentum: "var(--c3)", adam: "var(--c4)" };
const methodName: Record<Method, string> = { gd: "gradient descent", momentum: "momentum", adam: "Adam" };

export function ValleyWidget() {
  const [h2, setH2] = useState(10);
  const [eta, setEta] = useState(0.15);
  const [rho, setRho] = useState(0.5);
  const [on, setOn] = useState<Record<Method, boolean>>({ gd: true, momentum: true, adam: false });
  const h: [number, number] = [1, h2];
  const steps = 30;
  const runs = (Object.keys(on) as Method[])
    .filter((m) => on[m])
    .map((m) => ({ m, ...runOptimizer(m, h, eta, rho, steps) }));
  const levels = [0.05, 0.3, 1, 2.5, 5, 8];
  return (
    <Widget
      title="A narrow valley: J(θ) = ½(h₁θ₁² + h₂θ₂²) with h₁ = 1"
      caption="Ellipses are contour lines (equal loss). Start at θ₀ = (4, 1); the minimum is at (0, 0). Increase h₂ to make the valley narrower and watch GD zigzag."
    >
      <div className="toggle-row">
        {(Object.keys(on) as Method[]).map((m) => (
          <button key={m} className={on[m] ? "" : "off"} style={{ background: on[m] ? methodColor[m] : undefined, borderColor: methodColor[m] }} onClick={() => setOn({ ...on, [m]: !on[m] })}>
            {methodName[m]}
          </button>
        ))}
      </div>
      <div className="widget-row">
        <Chart xDomain={[-1.5, 4.5]} yDomain={[-1.6, 1.6]} xLabel="θ₁" yLabel="θ₂">
          {(s) => (
            <>
              {levels.map((c) => (
                <ellipse key={c} cx={s.x(0)} cy={s.y(0)} rx={s.x(Math.sqrt((2 * c) / h[0])) - s.x(0)} ry={s.y(0) - s.y(Math.sqrt((2 * c) / h[1]))} fill="none" stroke="var(--muted)" strokeOpacity={0.45} strokeWidth={1.2} />
              ))}
              {runs.map((r) => (
                <g key={r.m}>
                  <path d={polyline(r.pts, s)} stroke={methodColor[r.m]} strokeWidth={2} fill="none" />
                  {r.pts.map((p, i) => <circle key={i} cx={s.x(p[0])} cy={s.y(p[1])} r={2.5} fill={methodColor[r.m]} />)}
                </g>
              ))}
              <circle cx={s.x(0)} cy={s.y(0)} r={4} fill="var(--text)" />
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>{"h_2"}</M>} value={h2} min={1} max={18} step={1} onChange={setH2} />
          <Slider label={<M>\eta</M>} value={eta} min={0.01} max={0.25} step={0.01} onChange={setEta} />
          <Slider label={<>momentum <M>\rho</M></>} value={rho} min={0} max={0.95} step={0.05} onChange={setRho} />
          <Chart xDomain={[0, steps]} yDomain={[-6, 1.5]} width={360} height={200} xLabel="step" yLabel="log₁₀ loss">
            {(s) => runs.map((r) => (
              <path key={r.m} d={polyline(r.losses.map((l, i) => [i, Math.log10(Math.max(l, 1e-7))]), s)} stroke={methodColor[r.m]} strokeWidth={2} fill="none" />
            ))}
          </Chart>
          <p className="muted">GD diverges once η &gt; 2/h₂ (try h₂ = 15, η = 0.15). Adam uses β₁ = 0.9, β₂ = 0.999 and no bias correction, like the slides.</p>
        </div>
      </div>
    </Widget>
  );
}

// ---------- SGD: minibatches, noise floor, schedule ----------

export function SGDWidget() {
  const [B, setB] = useState(4);
  const [eta, setEta] = useState(0.1);
  const [decay, setDecay] = useState(false);
  const data = useMemo(() => {
    const r = rng(3);
    return Array.from({ length: 64 }, () => {
      const x = 2 * r() - 1;
      return { x, y: 2 * x + 0.5 + 0.3 * gauss(r) };
    });
  }, []);
  const N = data.length;
  const fullLoss = (w: number, b: number) => data.reduce((s, p) => s + (w * p.x + b - p.y) ** 2, 0) / N;
  const best = useMemo(() => {
    const th = ridge(data.map((p) => [p.x, 1]), data.map((p) => p.y), 1e-12);
    return fullLoss(th[0], th[1]);
  }, [data]);
  const epochs = 30;
  function train(batch: number) {
    const r = rng(11);
    let w = -2, b = 2;
    const out: number[] = [fullLoss(w, b) - best];
    for (let e = 0; e < epochs; e++) {
      const idx = Array.from({ length: N }, (_, i) => i);
      for (let i = N - 1; i > 0; i--) {
        const j = Math.floor(r() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      const lr = decay ? eta * Math.pow(0.9, e) : eta;
      for (let k = 0; k < N; k += batch) {
        const mb = idx.slice(k, k + batch);
        let gw = 0, gb = 0;
        for (const n of mb) {
          const res = w * data[n].x + b - data[n].y;
          gw += (2 * res * data[n].x) / mb.length;
          gb += (2 * res) / mb.length;
        }
        w -= lr * gw;
        b -= lr * gb;
      }
      out.push(fullLoss(w, b) - best);
    }
    return out;
  }
  const sgd = train(B);
  const gd = train(N);
  return (
    <Widget
      title="SGD vs gradient descent on a 64-point linear regression"
      caption={<>Plotted: how far the loss on <em>all</em> data is above the best possible loss, after each epoch (log scale). One epoch = one pass through all 64 samples, i.e. N/B updates.</>}
    >
      <div className="widget-row">
        <Chart xDomain={[0, epochs]} yDomain={[-6, 1]} xLabel="epoch" yLabel="log₁₀ (J − J*)">
          {(s) => (
            <>
              <path d={polyline(gd.map((l, i) => [i, Math.log10(Math.max(l, 1e-7))]), s)} stroke="var(--c2)" strokeWidth={2} fill="none" />
              <path d={polyline(sgd.map((l, i) => [i, Math.log10(Math.max(l, 1e-7))]), s)} stroke="var(--c1)" strokeWidth={2} fill="none" />
            </>
          )}
        </Chart>
        <div>
          <div className="legend"><span><i style={{ background: "var(--c1)" }} />SGD (B = {B})</span><span><i style={{ background: "var(--c2)" }} />GD (B = N = 64)</span></div>
          <Slider label={<>batch size <M>B</M></>} value={B} min={1} max={32} step={1} onChange={setB} />
          <Slider label={<M>\eta</M>} value={eta} min={0.01} max={0.3} step={0.01} onChange={setEta} />
          <label className="slider" style={{ gridTemplateColumns: "auto 1fr" }}>
            <input type="checkbox" checked={decay} onChange={(e) => setDecay(e.target.checked)} style={{ width: "auto" }} />
            <span>learning-rate schedule <M>{"\\eta_t = \\eta_0 \\cdot 0.9^t"}</M></span>
          </label>
          <div className="readout">updates per epoch: SGD <b>{Math.ceil(N / B)}</b>, GD <b>1</b></div>
          <p className="muted">SGD drops fast (many cheap updates) but then flattens at a noise floor. Turn on the schedule to push the floor down. Smaller B: noisier floor.</p>
        </div>
      </div>
    </Widget>
  );
}
