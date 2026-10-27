import { useEffect, useState } from "react";
import { lectures } from "../lessons";
import { quiz } from "./questions";
import { privateQuiz } from "../private";
import { QuizItem } from "./types";

/** Every question in course order, with a stable id. */
export const allQuestions: QuizItem[] = lectures.flatMap((l) =>
  l.sections.flatMap((s) =>
    [...(quiz[`${l.id}/${s.id}`] ?? []), ...(privateQuiz[`${l.id}/${s.id}`] ?? [])].map((q, i) => ({ ...q, id: `${l.id}/${s.id}#${i}`, lecture: l.id, section: s.id })),
  ),
);

export function questionsFor(lecture: string, section: string) {
  return allQuestions.filter((q) => q.lecture === lecture && q.section === section);
}

// Latest result per question id: true = answered correctly.
type Results = Record<string, boolean>;
const KEY = "comp432-quiz";
const listeners = new Set<() => void>();

function load(): Results {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}

let results: Results = load();

export function recordResult(id: string, correct: boolean) {
  results = { ...results, [id]: correct };
  try {
    localStorage.setItem(KEY, JSON.stringify(results));
  } catch {
    /* results just won't persist */
  }
  listeners.forEach((f) => f());
}

export function resetResults(ids?: string[]) {
  if (ids) {
    results = { ...results };
    ids.forEach((id) => delete results[id]);
  } else results = {};
  try {
    localStorage.setItem(KEY, JSON.stringify(results));
  } catch {
    /* ignore */
  }
  listeners.forEach((f) => f());
}

export function useResults(): Results {
  const [r, setR] = useState(results);
  useEffect(() => {
    const f = () => setR(results);
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  }, []);
  return r;
}

/** Accepts "0.333", "1/3", "-2", "1e-3". */
export function parseNumber(s: string): number | null {
  const t = s.trim().replace(",", ".").replace("−", "-");
  if (!t) return null;
  const frac = t.match(/^(-?[\d.]+)\s*\/\s*(-?[\d.]+)$/);
  if (frac) return parseFloat(frac[1]) / parseFloat(frac[2]);
  const v = Number(t);
  return isFinite(v) ? v : null;
}

export function isClose(given: number, answer: number, tol?: number) {
  const t = tol ?? Math.max(0.01, 0.02 * Math.abs(answer));
  return Math.abs(given - answer) <= t;
}
