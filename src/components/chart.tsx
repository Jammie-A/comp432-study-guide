import { ReactNode } from "react";

export type Scale = { x: (v: number) => number; y: (v: number) => number };

/** Minimal SVG chart: axes, ticks and a render-prop for the marks. */
export function Chart({
  xDomain,
  yDomain,
  width = 420,
  height = 280,
  xLabel,
  yLabel,
  xTicks,
  yTicks,
  children,
}: {
  xDomain: [number, number];
  yDomain: [number, number];
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number[];
  yTicks?: number[];
  children: (s: Scale) => ReactNode;
}) {
  const m = { l: 44, r: 12, t: 12, b: 36 };
  const [x0, x1] = xDomain;
  const [y0, y1] = yDomain;
  const x = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (width - m.l - m.r);
  const y = (v: number) => height - m.b - ((v - y0) / (y1 - y0)) * (height - m.t - m.b);
  const xt = xTicks ?? ticks(x0, x1);
  const yt = yTicks ?? ticks(y0, y1);
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img">
      <defs>
        <clipPath id={`clip-${width}-${height}`}>
          <rect x={m.l} y={m.t} width={width - m.l - m.r} height={height - m.t - m.b} />
        </clipPath>
      </defs>
      {xt.map((t) => (
        <g key={"x" + t}>
          <line className="grid" x1={x(t)} x2={x(t)} y1={m.t} y2={height - m.b} />
          <text className="tick" x={x(t)} y={height - m.b + 14} textAnchor="middle">
            {fmt(t)}
          </text>
        </g>
      ))}
      {yt.map((t) => (
        <g key={"y" + t}>
          <line className="grid" x1={m.l} x2={width - m.r} y1={y(t)} y2={y(t)} />
          <text className="tick" x={m.l - 6} y={y(t) + 4} textAnchor="end">
            {fmt(t)}
          </text>
        </g>
      ))}
      <rect className="frame" x={m.l} y={m.t} width={width - m.l - m.r} height={height - m.t - m.b} />
      {xLabel && (
        <text className="axis-label" x={(m.l + width - m.r) / 2} y={height - 4} textAnchor="middle">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          className="axis-label"
          transform={`translate(12 ${(m.t + height - m.b) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
      )}
      <g clipPath={`url(#clip-${width}-${height})`}>{children({ x, y })}</g>
    </svg>
  );
}

function ticks(a: number, b: number): number[] {
  const span = b - a;
  const raw = span / 5;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * pow).find((s) => span / s <= 6) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function fmt(v: number) {
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(0);
  return String(+v.toFixed(2));
}

/** Path through a sampled function. */
export function fnPath(f: (x: number) => number, [a, b]: [number, number], s: Scale, n = 200) {
  let d = "";
  for (let i = 0; i <= n; i++) {
    const xv = a + ((b - a) * i) / n;
    const yv = f(xv);
    if (!isFinite(yv)) continue;
    d += (d ? "L" : "M") + s.x(xv).toFixed(2) + "," + s.y(yv).toFixed(2);
  }
  return d;
}

export function polyline(pts: [number, number][], s: Scale) {
  return pts.map(([a, b], i) => (i ? "L" : "M") + s.x(a).toFixed(2) + "," + s.y(b).toFixed(2)).join("");
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="slider">
      <span className="slider-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <span className="slider-val">{format ? format(value) : value}</span>
    </label>
  );
}

/** Frame around an interactive visual. */
export function Widget({ title, children, caption }: { title: string; children: ReactNode; caption?: ReactNode }) {
  return (
    <div className="widget">
      <div className="widget-title">{title}</div>
      {children}
      {caption && <div className="widget-caption">{caption}</div>}
    </div>
  );
}
