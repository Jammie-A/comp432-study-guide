import { useState } from "react";
import { Chart, Slider, Widget, fnPath } from "../components/chart";
import { M } from "../components/ui";
import { f2 } from "./math";

const ICE_X = [10, 20, 30];
const ICE_Y = [0, 20, 40];

/** Drag w and b; see the line, the residuals and the MSE (Lecture 1, ice cream). */
export function LineFitWidget() {
  const [w, setW] = useState(1);
  const [b, setB] = useState(10);
  const preds = ICE_X.map((x) => w * x + b);
  const sq = preds.map((p, i) => (ICE_Y[i] - p) ** 2);
  const mse = sq.reduce((s, v) => s + v, 0) / 3;
  return (
    <Widget
      title="Fit a line to the ice cream data"
      caption="Each dashed red segment is an error y − ŷ. MSE squares them and averages. Try to reach MSE = 0."
    >
      <div className="widget-row">
        <Chart xDomain={[0, 40]} yDomain={[-20, 60]} xLabel="temperature x (°C)" yLabel="sales y">
          {(s) => (
            <>
              <path d={fnPath((x) => w * x + b, [0, 40], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              {ICE_X.map((x, i) => (
                <g key={i}>
                  <line x1={s.x(x)} x2={s.x(x)} y1={s.y(ICE_Y[i])} y2={s.y(preds[i])} stroke="var(--c2)" strokeDasharray="4 3" strokeWidth={2} />
                  <circle cx={s.x(x)} cy={s.y(ICE_Y[i])} r={5} fill="var(--text)" />
                </g>
              ))}
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>w</M>} value={w} min={-1} max={3} step={0.1} onChange={setW} format={(v) => v.toFixed(1)} />
          <Slider label={<M>b</M>} value={b} min={-40} max={40} step={1} onChange={setB} />
          <table className="data">
            <thead>
              <tr><th>x</th><th>y</th><th>ŷ = wx + b</th><th>(y − ŷ)²</th></tr>
            </thead>
            <tbody>
              {ICE_X.map((x, i) => (
                <tr key={i}><td>{x}</td><td>{ICE_Y[i]}</td><td>{f2(preds[i], 1)}</td><td>{f2(sq[i], 1)}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="readout">MSE = <b>{f2(mse, 1)}</b></div>
        </div>
      </div>
    </Widget>
  );
}

/** Slider over p: shows −ln p, the cross-entropy cost of putting probability p on the true class. */
export function NegLogWidget() {
  const [p, setP] = useState(0.5);
  return (
    <Widget
      title="The cost −ln(p) of the probability on the correct class"
      caption="p → 1: cost → 0 (confident and right). p → 0: cost blows up (confident and wrong)."
    >
      <div className="widget-row">
        <Chart xDomain={[0, 1]} yDomain={[0, 5]} xLabel="p (probability given to the correct class)" yLabel="−ln p">
          {(s) => (
            <>
              <path d={fnPath((x) => -Math.log(x), [0.005, 1], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
              <circle cx={s.x(p)} cy={s.y(-Math.log(p))} r={6} fill="var(--c2)" />
            </>
          )}
        </Chart>
        <div>
          <Slider label={<M>p</M>} value={p} min={0.01} max={1} step={0.01} onChange={setP} />
          <div className="readout">−ln({p.toFixed(2)}) = <b>{(-Math.log(p)).toFixed(3)}</b></div>
          <p className="muted">Slide values: −ln 0.3 = 1.20, −ln 0.5 = 0.69, −ln 0.2 = 1.61.</p>
        </div>
      </div>
    </Widget>
  );
}

/** Probe-and-step on the ice cream data: evaluate all 8 neighbours, move to the best. */
export function ProbeStepWidget() {
  const start = { w: 1, b: 0 };
  const [cur, setCur] = useState(start);
  const [history, setHistory] = useState<number[]>([]);
  const d = 0.05;
  const mse = (w: number, b: number) =>
    ICE_X.reduce((s, x, i) => s + (ICE_Y[i] - (w * x + b)) ** 2, 0) / ICE_X.length;
  const probes: { dw: number; db: number; J: number }[] = [];
  for (const dw of [d, 0, -d])
    for (const db of [d, 0, -d]) if (dw || db) probes.push({ dw, db, J: mse(cur.w + dw, cur.b + db) });
  const best = probes.reduce((a, p) => (p.J < a.J ? p : a));
  const curJ = mse(cur.w, cur.b);
  function step(n: number) {
    let c = cur;
    const h = [...history];
    for (let k = 0; k < n; k++) {
      let bst = { w: c.w, b: c.b, J: mse(c.w, c.b) };
      for (const dw of [d, 0, -d])
        for (const db of [d, 0, -d]) {
          const J = mse(c.w + dw, c.b + db);
          if (J < bst.J) bst = { w: c.w + dw, b: c.b + db, J };
        }
      h.push(mse(c.w, c.b));
      c = { w: bst.w, b: bst.b };
    }
    setCur(c);
    setHistory(h);
  }
  return (
    <Widget
      title="Probe and step (the naive training loop)"
      caption={`Each step costs 3^P − 1 = 8 objective evaluations for P = 2 parameters. With a million parameters that number is astronomically large, which is why Lecture 2 uses the gradient instead.`}
    >
      <div className="widget-row">
        <div>
          <div className="readout">
            epoch {history.length}: w = <b>{cur.w.toFixed(2)}</b>, b = <b>{cur.b.toFixed(2)}</b>, MSE = <b>{curJ.toFixed(2)}</b>
          </div>
          <table className="data">
            <thead><tr><th>Δw</th><th>Δb</th><th>MSE</th></tr></thead>
            <tbody>
              {probes.map((p, i) => (
                <tr key={i} style={p === best && best.J < curJ ? { fontWeight: 700, color: "var(--c3)" } : undefined}>
                  <td>{p.dw > 0 ? "+" : ""}{p.dw}</td><td>{p.db > 0 ? "+" : ""}{p.db}</td><td>{p.J.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="toggle-row">
            <button onClick={() => step(1)}>Take 1 step</button>
            <button className="ghost" onClick={() => step(50)}>+50 steps</button>
            <button className="ghost" onClick={() => { setCur(start); setHistory([]); }}>Reset</button>
          </div>
        </div>
        <Chart xDomain={[0, Math.max(10, history.length + 1)]} yDomain={[0, Math.max(10, mse(1, 0) * 1.05)]} xLabel="epoch" yLabel="MSE">
          {(s) => (
            <path
              d={[...history, curJ].map((J, i) => (i ? "L" : "M") + s.x(i) + "," + s.y(J)).join("")}
              stroke="var(--c1)" strokeWidth={2} fill="none"
            />
          )}
        </Chart>
      </div>
    </Widget>
  );
}
