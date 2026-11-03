// Quiz question types.
// Text fields support inline math as $...$ and inline code as {{...}}.

export type QuizKind = "formula" | "calc" | "code" | "reason";

type Base = {
  kind: QuizKind;
  prompt: string;
  /** Code shown above the question (for "predict the output" questions). */
  code?: string;
  explain: string;
  /** Goes beyond what the slides cover. */
  extra?: boolean;
};

export type MCQuestion = Base & { type: "mc"; options: string[]; answer: number };
export type NumQuestion = Base & { type: "num"; answer: number; tol?: number };
export type Question = MCQuestion | NumQuestion;

/** A question together with where it lives. */
export type QuizItem = Question & { id: string; lecture: string; section: string };

export const kindLabel: Record<QuizKind, string> = {
  formula: "Formula / concept",
  calc: "Hand calculation",
  code: "Predict the output",
  reason: "Reasoning: what if / why",
};
