import { ReactNode } from "react";

export type Section = {
  id: string;
  title: string;
  /** One line shown under the title and on the overview page. */
  summary: string;
  body: () => ReactNode;
};

export type Lecture = {
  id: string;
  number: number;
  title: string;
  /** Overrides "Lecture N" in headings, e.g. for the labs track. */
  label?: string;
  /** Short tag for score tables, e.g. "L3" or "Labs". */
  short?: string;
  sections: Section[];
};

export const lectureLabel = (l: Lecture) => l.label ?? `Lecture ${l.number}`;
export const lectureShort = (l: Lecture) => l.short ?? `L${l.number}`;
