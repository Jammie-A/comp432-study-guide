import { Chart, fnPath } from "../components/chart";

/** Training and validation loss over epochs; validation bottoms out around epoch 40. */
export function EarlyStoppingFigure() {
  const train = (e: number) => 0.15 + 1.6 * Math.exp(-e / 15);
  const val = (e: number) => 0.45 + 1.4 * Math.exp(-e / 15) + 0.00012 * (e - 40) ** 2 * (e > 40 ? 1 : 0.15);
  return (
    <Chart xDomain={[0, 100]} yDomain={[0, 2]} width={420} height={240} xLabel="epoch" yLabel="loss" xTicks={[0, 20, 40, 60, 80, 100]}>
      {(s) => (
        <>
          <path d={fnPath(train, [0, 100], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
          <path d={fnPath(val, [0, 100], s)} stroke="var(--c2)" strokeWidth={2.5} fill="none" />
          <text x={s.x(78)} y={s.y(train(78)) - 8} style={{ fontSize: 12, fill: "var(--c1)" }}>training</text>
          <text x={s.x(72)} y={s.y(val(72)) - 8} style={{ fontSize: 12, fill: "var(--c2)" }}>validation</text>
        </>
      )}
    </Chart>
  );
}

/** Three training-loss curves labelled P, Q, R for different learning rates. */
export function LearningRateFigure() {
  const P = (t: number) => 2.4 * Math.exp(-t / 60);
  const Q = (t: number) => 2.4 * Math.exp(-t / 6) + 0.05;
  const R = (t: number) => 2.4 * Math.exp(t / 9) * (1 + 0.25 * Math.sin(t * 1.3));
  return (
    <Chart xDomain={[0, 30]} yDomain={[0, 6]} width={420} height={240} xLabel="step" yLabel="training loss">
      {(s) => (
        <>
          <path d={fnPath(P, [0, 30], s)} stroke="var(--c1)" strokeWidth={2.5} fill="none" />
          <path d={fnPath(Q, [0, 30], s)} stroke="var(--c3)" strokeWidth={2.5} fill="none" />
          <path d={fnPath(R, [0, 30], s, 400)} stroke="var(--c2)" strokeWidth={2.5} fill="none" />
          <text x={s.x(27)} y={s.y(P(27)) - 8} style={{ fontSize: 13, fontWeight: 700, fill: "var(--c1)" }}>P</text>
          <text x={s.x(27)} y={s.y(Q(27)) - 8} style={{ fontSize: 13, fontWeight: 700, fill: "var(--c3)" }}>Q</text>
          <text x={s.x(7.5)} y={s.y(5.2)} style={{ fontSize: 13, fontWeight: 700, fill: "var(--c2)" }}>R</text>
        </>
      )}
    </Chart>
  );
}

/** The four arrow classes of the slides' sample question 2 (9×9 images). */
export function ArrowsFigure() {
  // Each arrow drawn on a 9×9 pixel grid.
  const right = [
    ".........",
    "....#....",
    ".....#...",
    "......#..",
    ".#######.",
    "......#..",
    ".....#...",
    "....#....",
    ".........",
  ];
  const rot = (g: string[]) => g.map((_, i) => g.map((row) => row[i]).reverse().join("")); // 90° clockwise
  const flipH = (g: string[]) => g.map((row) => row.split("").reverse().join(""));
  const down = rot(right);
  const left = flipH(right);
  const up = rot(left);
  const items: [string, string[]][] = [["left", left], ["right", right], ["up", up], ["down", down]];
  const px = 9;
  return (
    <svg viewBox="0 0 400 110" className="chart" style={{ maxWidth: 400 }}>
      {items.map(([name, g], k) => (
        <g key={name} transform={`translate(${12 + k * 98} 6)`}>
          {g.flatMap((row, i) =>
            row.split("").map((c, j) => (
              <rect key={`${i}-${j}`} x={j * px} y={i * px} width={px - 1} height={px - 1} fill={c === "#" ? "var(--text)" : "var(--code-bg)"} />
            )),
          )}
          <text x={(9 * px) / 2} y={9 * px + 18} textAnchor="middle" style={{ fontSize: 13 }}>{name}</text>
        </g>
      ))}
    </svg>
  );
}

/** Gradient descent zig-zagging in an elongated bowl. */
export function ZigzagFigure() {
  const pts: [number, number][] = [];
  let a = -4, b = 1;
  for (let i = 0; i < 14; i++) {
    pts.push([a, b]);
    a = a - 0.09 * a;
    b = b - 1.9 * b;
  }
  return (
    <Chart xDomain={[-5, 1]} yDomain={[-1.5, 1.5]} width={420} height={220} xLabel="w₁ (multiplies feature x₁ ∈ [0, 1])" yLabel="w₂">
      {(s) => (
        <>
          {[0.3, 1, 2, 3.5].map((c) => (
            <ellipse key={c} cx={s.x(0)} cy={s.y(0)} rx={s.x(c * 1.4) - s.x(0)} ry={s.y(0) - s.y(c * 0.33)} fill="none" stroke="var(--muted)" strokeOpacity={0.45} />
          ))}
          <path d={pts.map(([x, y], i) => (i ? "L" : "M") + s.x(x) + "," + s.y(y)).join("")} stroke="var(--c2)" strokeWidth={2} fill="none" />
          {pts.map(([x, y], i) => <circle key={i} cx={s.x(x)} cy={s.y(y)} r={2.5} fill="var(--c2)" />)}
        </>
      )}
    </Chart>
  );
}
