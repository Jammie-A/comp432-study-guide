# COMP 432 Study Guide

An interactive study guide for COMP 432, lectures 1–5 and labs 1–5. Each section has a plain-language explanation, every formula's symbols explained, step-by-step worked examples, interactive visuals, and runnable Python (NumPy and scikit-learn, via Pyodide in the browser).

For the midterm:

- **Check yourself** questions at the end of every section (142 in total), mixed together on the Quiz page (`#/quiz`).
- **Mock midterm** (`#/exam`) in the real exam's format: five options per question, −¼ for a wrong answer, 0 for a blank, a timer, two bonus questions, and a review with explanations. It includes the three example questions from the Lecture 5 handout.
- Formula boxes are tagged **On the formula sheet** or **Not on the sheet: memorize**, following the handout's description of the booklet.

## Run

```bash
npm install
npm run dev
```

Then open the URL it prints (usually http://localhost:5173). The first time you run a Python cell, Python is downloaded from the jsDelivr CDN, so you need to be online. Cells that use scikit-learn take a little longer the first time. PyTorch can't run in the browser, so PyTorch code is shown read-only.

Your progress (the "I understand this section" checkboxes, quiz scores and mock-exam attempts) is stored in the browser for that exact address. Opening the app in a different browser or on a different port starts from zero.

## Shared version (GitHub Pages)

Every push to `main` builds the site and publishes it to https://jammie-a.github.io/comp432-study-guide/ (see `.github/workflows/deploy.yml`).

Material close to graded lab-assignment solutions lives in `src/private/`, which is git-ignored. Local builds include it automatically (via `src/private.tsx`); the shared site is built without it and shows a short note in its place. Never commit that folder.

## Layout

- `src/lessons/l1.tsx` … `l5.tsx`: lesson content, one file per lecture (`handout.pdf` is Lecture 4, `handout5.pdf` is Lecture 5)
- `src/lessons/labs.tsx`: the Labs 1–5 track (NumPy, finite differences, from-scratch models and scikit-learn, PyTorch, convolutions in code, debugging)
- `src/widgets/`: interactive visuals for each lecture
- `src/components/`: math rendering, step-by-step reveal, Python cells, charts
- `src/quiz/questions.ts`: section quiz questions (multiple choice or numeric; `$...$` is math, `{{...}}` is inline code)
- `src/exam/questions.ts`: the mock-midterm pool (five options each, with points); `src/exam/figures.tsx` draws its figures
- `content-outline.md`: what each slide and lab covers, with page numbers

To add a lecture:

1. Write `src/lessons/lN.tsx` following the same pattern and register it in `src/lessons/index.ts`.
2. Add its interactive visuals in `src/widgets/lN.tsx`, if any.
3. Add its questions to `src/quiz/questions.ts` under keys like `"lN/<section-id>"`, and exam-style ones to `src/exam/questions.ts`.
4. Add its slide summary to `content-outline.md`.
