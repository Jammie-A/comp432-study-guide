// Optional content that is NOT published: material close to graded-assignment
// solutions lives in src/private/ (git-ignored). Local builds pick it up; the
// shared GitHub Pages build doesn't have the folder and shows a short note instead.
import { ReactNode } from "react";
import type { Question } from "./quiz/types";
import type { ExamQuestion } from "./exam/questions";

type PrivateModule = {
  blocks?: Record<string, () => ReactNode>;
  quiz?: Record<string, Question[]>;
  exam?: ExamQuestion[];
};

const modules = import.meta.glob<PrivateModule>("./private/*.tsx", { eager: true });
const loaded = Object.values(modules);

export const privateBlocks: Record<string, () => ReactNode> = Object.assign({}, ...loaded.map((m) => m.blocks ?? {}));
export const privateQuiz: Record<string, Question[]> = Object.assign({}, ...loaded.map((m) => m.quiz ?? {}));
export const privateExam: ExamQuestion[] = loaded.flatMap((m) => m.exam ?? []);

/** Renders a private block if this build has it, otherwise a short note (or nothing, if `quiet`). */
export function Private({ id, quiet }: { id: string; quiet?: boolean }) {
  const block = privateBlocks[id];
  if (block) return <>{block()}</>;
  if (quiet) return null;
  return (
    <p className="private-note">
      Solution code for a graded lab assignment is left out of the shared version. Work through the assignment notebook
      yourself.
    </p>
  );
}
