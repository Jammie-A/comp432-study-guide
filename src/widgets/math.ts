// Tiny numeric helpers for the interactive widgets.

/** Seeded RNG so the "random" datasets look the same every time. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample (Box–Muller). */
export function gauss(r: () => number) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}

/** Solve A x = b with Gaussian elimination and partial pivoting. */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/** Polynomial features (x, x^2, ..., x^M, 1), matching Lecture 4's ordering. */
export function polyFeatures(x: number, M: number) {
  const f: number[] = [];
  for (let m = 1; m <= M; m++) f.push(Math.pow(x, m));
  f.push(1);
  return f;
}

/** Closed-form (optionally L2-regularized) least squares: (ΦᵀΦ + Nλ/2 I)⁻¹ Φᵀy. */
export function ridge(Phi: number[][], y: number[], lambda: number) {
  const N = Phi.length;
  const P = Phi[0].length;
  const A = Array.from({ length: P }, (_, i) =>
    Array.from({ length: P }, (_, j) => {
      let s = 0;
      for (let n = 0; n < N; n++) s += Phi[n][i] * Phi[n][j];
      return s + (i === j ? (N * lambda) / 2 : 0);
    }),
  );
  const b = Array.from({ length: P }, (_, i) => {
    let s = 0;
    for (let n = 0; n < N; n++) s += Phi[n][i] * y[n];
    return s;
  });
  return solve(A, b);
}

export const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const f2 = (v: number, d = 2) => (Math.abs(v) < 1e-12 ? "0" : v.toFixed(d));
