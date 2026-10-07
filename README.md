# KnowCS — COMP2211 interactive learning lab

[中文](./README_ZH.md) | **English**

KnowCS is an interactive study site for HKUST **COMP2211 (Exploring Artificial Intelligence)**. Every page is built around the ideas the midterms and finals keep asking about, and every number on screen is computed live from tested code.

**Live site:** <https://knowcs.online> — English / 简体中文 / 繁體中文 (`?lang=en`, `?lang=zh`, `?lang=zh-HK`).

## What is on the site

| Section | URL | What it does |
|---|---|---|
| Course map | `/` | Lectures 1–11 with their practice pages, explainers and exam badges. |
| Practice pages | `/<topic>/` | 18 hands-on pages (NumPy, Bayes, KNN, evaluation, K-Means, perceptron, backprop, convolution, Otsu, CNN shapes, alpha-beta …). Most fill in the exact table an exam asks for, with a “Quiz me” mode. |
| Explainers | `/watch/` | 15 animated, 3Blue1Brown-style explainers that run in the browser: chapters, captions, “pause and ponder” stops, a transcript and an exam corner with a past-paper question. |
| Concept check | `/drill/` | 42 original true/false statements in the style of the finals’ Problem 1, with explanations. |
| Formula sheet | `/formulas/` | Every formula the exams use, with the trap to avoid; prints to three A4 pages. |
| Exam map | `/papers/` | Which topics each of nine past papers asked. |
| Notes / Extend | `/notes/`, `/extend/` | Package notes (NumPy with a live API panel, pandas, PyTorch …) and extension pages. |

## Development

```bash
npm install
npm run dev        # http://localhost:5174
npm run lint       # ESLint
npm test           # Vitest (400+ tests)
npm run typecheck  # tsc --noEmit
npm run build      # multi-page build into dist/
```

React 19, TypeScript (strict), Vite 7 (multi-page; pages come from `src/lib/sitemap.ts`), Tailwind CSS 4, Framer Motion, KaTeX and react-i18next. All calculations live in `src/lib/` with unit tests; the explainers are timeline-driven React/SVG scenes in `src/site/watch/`. Project conventions for contributors are in [CLAUDE.md](./CLAUDE.md); the change log and plan are in [docs/](./docs/).

## Acknowledgements

Early content drew on the study notes at [moyunxiang/COMP2211](https://github.com/moyunxiang/COMP2211/blob/main/COMP2211.md). Exam examples are paraphrased or recomputed from the course’s lectures and past papers for study use.
