import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { glossary, GlossaryEntry } from "../glossary";
import { lectures } from "../lessons";

/** The term without its parenthetical, e.g. "gradient (∇J)" → "gradient". */
const shortName = (g: GlossaryEntry) => g.term.split(" (")[0];

const names = (g: GlossaryEntry) => [shortName(g).toLowerCase(), ...(g.aliases ?? [])];

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Matches a whole word or phrase, treating letters, digits and hyphens as word characters. */
const wordRe = (n: string) => new RegExp(`(^|[^a-z0-9-])${escape(n)}([^a-z0-9-]|$)`);

function sectionTitle(key: string) {
  const [lec, sec] = key.split("/");
  return lectures.find((l) => l.id === lec)?.sections.find((s) => s.id === sec)?.title;
}

function WhereLink({ g, here }: { g: GlossaryEntry; here?: string }) {
  if (!g.where || g.where === here) return null;
  return (
    <a href={`#/${g.where}`} className="term-where">
      Explained in: {sectionTitle(g.where)} →
    </a>
  );
}

/** "Terms on this page": every non-basic glossary word that appears in the surrounding article, each with its definition. */
export function PageTerms({ here }: { here: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [found, setFound] = useState<GlossaryEntry[]>([]);
  const [open, setOpen] = useState<string | null>(null);

  useLayoutEffect(() => {
    const article = ref.current?.closest("article");
    if (!article) return;
    const text = article.innerText.toLowerCase();
    setFound(glossary.filter((g) => !g.basic && names(g).some((n) => wordRe(n).test(text))));
    setOpen(null);
  }, [here]);

  if (found.length === 0) return <div ref={ref} />;
  const cur = found.find((g) => g.id === open);
  return (
    <div className="page-terms" ref={ref}>
      <span className="page-terms-label">Terms on this page:</span>
      {found.map((g) => (
        <button
          key={g.id}
          className={"term-chip" + (g.id === open ? " on" : "")}
          aria-expanded={g.id === open}
          onClick={() => setOpen(g.id === open ? null : g.id)}
        >
          {shortName(g)}
        </button>
      ))}
      {cur && (
        <div className="term-def" role="region" aria-label={`Definition of ${shortName(cur)}`}>
          <strong>{cur.term}</strong>: {cur.plain} <WhereLink g={cur} here={here} />{" "}
          <a href={`#/glossary/${cur.id}`} className="term-where">
            All terms
          </a>
        </div>
      )}
    </div>
  );
}

/** The full glossary with a search box. `focus` highlights one entry (from #/glossary/<id>). */
export function GlossaryPage({ focus }: { focus?: string }) {
  const [q, setQ] = useState("");
  const sorted = useMemo(() => [...glossary].sort((a, b) => a.term.localeCompare(b.term)), []);
  const shown = sorted.filter((g) => !q || (g.term + " " + g.plain + " " + (g.aliases ?? []).join(" ")).toLowerCase().includes(q.toLowerCase()));

  useLayoutEffect(() => {
    if (focus) document.getElementById(`term-${focus}`)?.scrollIntoView({ block: "center" });
  }, [focus]);

  return (
    <article>
      <h1>Glossary</h1>
      <p className="summary">Every term the course uses, in plain words, with a link to the section that teaches it.</p>
      <input
        className="term-search"
        type="search"
        placeholder="Search terms…"
        aria-label="Search terms"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <p className="muted">
        {shown.length} of {glossary.length} terms
      </p>
      <dl className="glossary">
        {shown.map((g) => (
          <div key={g.id} id={`term-${g.id}`} className={g.id === focus ? "glossary-item focus" : "glossary-item"}>
            <dt>{g.term}</dt>
            <dd>
              {g.plain}
              {g.example && <div className="muted">Example: {g.example}</div>}
              <WhereLink g={g} />
            </dd>
          </div>
        ))}
      </dl>
    </article>
  );
}
