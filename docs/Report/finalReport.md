# COMP 4971B

# KnowCS: An Interactive Visualization Lab for Learning Machine Learning Concepts in COMP 2211

COMP 4971B Final Report

By:

ZHANG Jiahao (Charles) — Student ID: 21121168

Advised By:

Dr. Desmond Yau-chat Tsoi

Submitted in partial fulfillment of the requirements of

COMP 4971B (Independent Work)

in the

Department of Computer Science and Engineering

The Hong Kong University of Science and Technology

2025–2026 (Spring 2026)

Date of Submission:

31 May 2026

Project Link: <https://knowcs.pinme.dev>

---

# Abstract

Introductory machine learning courses such as HKUST's COMP 2211 ("Exploring Artificial Intelligence") compress a large body of mathematically dense material — array broadcasting, Bayesian inference, the chain rule of backpropagation, convolution, and distance-based classification — into a single semester. Students frequently understand each formula in isolation yet struggle to connect a symbol on a slide to the computation it performs. This project, **KnowCS**, addresses that gap with an interactive, browser-based learning lab that turns static lecture formulas into manipulable experiments. The platform reframes six core COMP 2211 topics as visual modules in which the learner adjusts parameters — dragging a slider, clicking a canvas, editing a convolution kernel — and observes the algorithm's response in real time, with every intermediate quantity rendered as live LaTeX. KnowCS is built as a zero-backend, single-file web application (React 19, TypeScript, Vite, Tailwind CSS, Framer Motion, KaTeX) so that all computation runs client-side and the entire site ships as one self-contained HTML file. It is fully bilingual (English / Chinese) and is continuously deployed to a decentralized host (IPFS via PinMe) at <https://knowcs.pinme.dev>. This report documents the project's objectives, a survey of comparable educational tools, the system's architecture and the implementation of each visualization module, an evaluation against the stated objectives, and a discussion of the current limitations and the planned course-aligned roadmap.

---

# Table of Contents

1. Introduction
   - 1.1 Overview
   - 1.2 Objectives
     - Objective 1: Make abstract COMP 2211 formulas tangible through real-time interaction
     - Objective 2: Build a maintainable, zero-friction platform that is easy to deploy and extend
     - Objective 3: Serve both local and international students with a fully bilingual experience
   - 1.3 Literature Survey
     - 1.3.1 3Blue1Brown & Manim
     - 1.3.2 TensorFlow Playground
     - 1.3.3 Seeing Theory & Distill.pub
     - 1.3.4 The moyunxiang/COMP2211 study notes
2. Methodology
   - 2.1 Design
     - 2.1.1 System Design
     - 2.1.2 Design choices and rationale
   - 2.2 Implementation
     - 2.2.1 Technical foundation
     - 2.2.2 Shared components
     - 2.2.3 The six visualization modules
     - 2.2.4 Internationalization
     - 2.2.5 Deployment pipeline
   - 2.3 Evaluation
   - 2.4 Limitations and Further Study
3. Conclusion
4. Hardware & Software
5. References

---

# 1. Introduction

## 1.1 Overview

COMP 2211 ("Exploring Artificial Intelligence") is HKUST's large introductory course on machine learning and artificial intelligence. Across a single semester it covers NumPy and image fundamentals, Naïve Bayes, K-Nearest Neighbors, K-Means clustering, perceptrons and multilayer neural networks, backpropagation, image processing and convolution, convolutional neural networks, PyTorch, and adversarial search (Minimax with Alpha-Beta pruning). Each of these topics is built on its own piece of mathematics, and the course assessments — as reflected in past papers — repeatedly require students to *hand-trace* algorithms: to expand a Naïve Bayes posterior term by term, to step through a K-Means iteration, or to apply the chain rule across a small network.

The difficulty for many students is not the individual formula but the gap between the formula and what it *does*. A lecture slide can state that basic NumPy slicing returns a *view* while fancy indexing returns a *copy*, or that broadcasting only stretches dimensions of size 1, but a static slide cannot show the learner *which cells in memory are shared* when they mutate a slice. Similarly, a slide can write Bayes' theorem, but it cannot let the student drag the prior probability and watch the posterior fail to follow their intuition. The conceptual load is high precisely because the feedback loop is missing: the student changes nothing and observes nothing.

KnowCS is an interactive learning platform designed to close that feedback loop. Rather than presenting machine learning concepts as text and static images, it presents them as small, self-contained *experiments*. Every module exposes the parameters of an algorithm as interactive controls and renders the resulting computation immediately, with the relevant mathematics displayed as live, typeset formulas that update alongside the visualization. The guiding philosophy is "learning by doing": the learner is invited to manipulate the inputs and discover the behavior of the model directly, transforming a static lecture note into a dynamic lab manual that accompanies the COMP 2211 syllabus.

The platform is delivered as a single web page that runs entirely in the browser, requires no installation or login, and is available in both English and Chinese. A live deployment is hosted at <https://knowcs.pinme.dev>.

## 1.2 Objectives

The primary objective of this project is to help COMP 2211 students build correct intuition for the course's core algorithms by letting them interact with those algorithms directly. This breaks down into three concrete objectives.

### Objective 1: Make abstract COMP 2211 formulas tangible through real-time interaction

The first and central objective is to take the concepts that students find hardest to grasp from text alone and render them as manipulable, real-time visualizations. For each targeted topic, the learner should be able to change an input and immediately see both the visual consequence and the corresponding update to the underlying formula. The aim is not to replace the lecture, but to provide the missing experimental companion to it — a place where a student can answer "what happens if…?" by simply trying it.

### Objective 2: Build a maintainable, zero-friction platform that is easy to deploy and extend

The second objective concerns engineering. For a teaching tool to actually be used, it must be trivial to reach (no installation, no account) and cheap to host, and for it to survive beyond a single semester it must be easy for future contributors to extend with new modules. This objective therefore calls for a zero-backend, client-side architecture that packages into a single distributable artifact, together with a modular code structure and an automated build-and-deploy pipeline that gates every release on quality checks.

### Objective 3: Serve both local and international students with a fully bilingual experience

COMP 2211 serves a mixed cohort of local and international students. The third objective is therefore full bilingual support: every piece of explanatory text, every "exam tip," and every richly formatted sentence containing inline mathematics must be available in both English and Chinese, switchable with a single click, without duplicating or forking the application.

## 1.3 Literature Survey

A range of high-quality tools exist for explaining mathematical and machine learning concepts visually. Surveying them clarifies both what works and where a course-specific gap remains.

### 1.3.1 3Blue1Brown & Manim

3Blue1Brown is a widely admired educational channel whose animations, produced with the open-source **Manim** engine, make linear algebra and neural networks visually intuitive. Its great strength is the clarity of its motion design. Its limitation, for the purpose of this project, is that the videos are *non-interactive*: the viewer watches a fixed narrative and cannot change the parameters or pose their own "what if." KnowCS borrows the visual sensibility of such work but replaces passive viewing with active manipulation.

### 1.3.2 TensorFlow Playground

TensorFlow Playground is a celebrated in-browser neural network sandbox that lets users add layers and neurons and watch a decision boundary form in real time. It is one of the closest spiritual predecessors to this project and demonstrates the pedagogical power of immediate, client-side interaction. However, it is scoped to a single topic (small neural networks on toy datasets) and is not aligned to any particular course syllabus. KnowCS adopts the same "live sandbox" principle but applies it across the *breadth* of COMP 2211's topics rather than one of them, and ties each module to the exact formulas and examples the course assesses.

### 1.3.3 Seeing Theory & Distill.pub

*Seeing Theory* (Brown University) is an interactive visual introduction to probability and statistics, and *Distill.pub* published a body of interactive machine learning articles. Both show how interactive web pages can convey statistical and ML ideas far more effectively than static text. They are general-audience resources rather than course companions; their content does not map onto a specific lecture sequence or examination format. KnowCS is differentiated by being deliberately *course-aligned*: its modules, datasets, and "exam tips" are derived from COMP 2211's lectures, labs, and past papers.

### 1.3.4 The moyunxiang/COMP2211 study notes

The community-maintained `moyunxiang/COMP2211` notes are a comprehensive textual study guide for the exact course this project targets. They are an authoritative content reference, but they are static Markdown. KnowCS is conceived as the *interactive supplement* to these notes: where the notes state a formula, KnowCS lets the student operate it. The long-term plan is for the COMP 2211 portion of KnowCS to be contributed back and cross-linked with these notes under an open-source (MIT) license.

**Summary of the gap.** Excellent visual and interactive ML resources exist, but they are either non-interactive (videos), narrow in scope (single-topic sandboxes), or unaligned to a specific course. KnowCS occupies the intersection that none of them fully fill: *broad coverage of one specific course's syllabus, delivered as live, hands-on, bilingual interaction.*

---

# 2. Methodology

## 2.1 Design

### 2.1.1 System Design

KnowCS is architected as a **single-page application with no backend**. All state, computation, and rendering happen in the browser; there is no server, database, or API call at runtime. The application is structured as a thin application shell surrounding a set of independent, self-contained visualization modules.

The high-level structure is as follows:

```
index.html  ──▶  main.tsx  ──▶  <App/>
                                  │  (shell: Header, Sidebar nav, Content, Footer)
                                  │  holds activeTab + current language
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ Shared components       │ Visualization modules  │
         │  • Latex (KaTeX)        │  • NumpyModule         │
         │  • SectionTitle         │  • BackpropModule      │
         │  • SeniorAdvice         │  • KernelModule        │
         │                         │  • BayesBasicsModule   │
         │                         │  • NaiveBayesModule    │
         │                         │  • KnnModule           │
         └─────────────────────────────────────────────────┘
                          i18n (react-i18next): en.json / zh.json
```

The shell (`App.tsx`) owns exactly two pieces of global state: the currently selected tab (`activeTab`) and the current UI language (managed by i18next). Navigation is a sidebar of tabs; selecting a tab conditionally renders the corresponding module, with an `AnimatePresence` fade transition between modules. Each module is otherwise *fully independent*: it owns its own local state, performs its own computation, and shares nothing with its siblings. The roles of each component are summarized below.

| Component | Function |
| --- | --- |
| `App` (shell) | Renders the header, sidebar navigation, content area, and footer; holds the active tab and triggers language switching. |
| Shared components | `Latex` wraps KaTeX rendering; `SectionTitle` provides a consistent module header; `SeniorAdvice` renders an amber "exam tip" callout per module. |
| Visualization modules | Six independent modules, each implementing the interaction, computation, and explanation for one COMP 2211 topic. |
| i18n layer | Supplies all user-facing strings from `en.json` / `zh.json`, including rich-text sentences with embedded formulas. |

*Figure 1 — System architecture: a stateful shell routing between six self-contained, client-side visualization modules.*

### 2.1.2 Design choices and rationale

**Zero backend / client-side computation.** Every algorithm visualized in KnowCS (Euclidean and standardized distances, 3×3 convolution, Bayesian posteriors, the chain rule) is light enough to run instantly in JavaScript. Pushing all computation to the client removes the need for a server, eliminates network latency from the interaction loop, and makes the application free to host. This directly serves Objective 2.

**Single-file distribution.** Using `vite-plugin-singlefile`, the production build inlines all JavaScript and CSS into a single `index.html`. The resulting artifact has no external resource requests and can be opened directly from disk, served from any static host, or pinned to IPFS. The trade-off — a larger single payload that defeats incremental caching — is acceptable for an educational site whose value lies in being instantly and durably reachable.

**Per-module local state ("nearest-owner" management).** Rather than introducing a global state library, each module manages its own state with React's `useState`, and all derived quantities (distances, probabilities, convolution outputs) are computed with `useMemo`. This keeps modules decoupled and independently reasoned about, and makes adding or removing a module a local change. It is the structural decision that makes the platform extensible (Objective 2).

**Live mathematics via KaTeX.** Because the pedagogical goal is to connect *interaction* to *formula*, every module renders its mathematics with KaTeX as first-class, dynamically updating content rather than as static images. The same formulas the student sees on a lecture slide appear in the tool, and their terms update as the student manipulates the inputs (Objective 1).

**Externalized, course-derived content.** All explanatory copy lives in the i18n resource files, and all teaching datasets live in a single typed constants file (`src/data/constants.ts`). This separates *content* (which is derived from the course's lectures, labs, and past papers) from *code*, so that the pedagogy can be refined without touching the rendering logic.

## 2.2 Implementation

### 2.2.1 Technical foundation

KnowCS is implemented on a modern, strictly-typed front-end stack:

- **React 19** with function components and Hooks for the UI.
- **TypeScript (strict mode)** for the entire codebase; `tsc --noEmit` is part of both the type-check and build steps, so type errors block a release.
- **Vite 7** as the dev server and bundler, with `vite-plugin-singlefile` for single-file output.
- **Tailwind CSS 4** for utility-first styling.
- **Framer Motion 12** for transitions and process animations (module fades, animated value bars, the backpropagation "error particle," neighbor connectors).
- **KaTeX** for high-performance LaTeX formula rendering.
- **Recharts 3** for the KNN model-complexity chart.
- **react-i18next** (with browser language detection) for internationalization.
- **lucide-react** for iconography.

The codebase is organized for extensibility: `App.tsx` is a ~209-line shell handling routing and layout; the six modules live independently under `src/modules/`; shared UI lives in `src/components/`; all teaching data is centralized in `src/data/constants.ts`; and shared types are declared in `src/types.ts`. (An earlier iteration kept all modules inline in a single ~1,065-line file; the project was subsequently migrated to strict TypeScript and refactored into this modular structure.)

### 2.2.2 Shared components

Three shared components establish a consistent visual and pedagogical grammar across all modules:

- **`Latex`** — a thin wrapper over KaTeX. It obtains a DOM container via `useRef` and calls `katex.render` inside a `useEffect` keyed on `[formula, displayMode]`, supporting both inline and block math. It sets `throwOnError: false` so that a malformed formula degrades gracefully instead of crashing the page.
- **`SectionTitle`** — standardizes each module's header (icon + uppercase title + italic subtitle).
- **`SeniorAdvice`** ("senior's advice") — an amber callout box that carries each module's exam-oriented takeaway or common-mistake warning. Its content is passed through i18next's `<Trans>` component so that a single sentence can simultaneously embed `<Latex>` formulas and `<strong>` emphasis *and* be translated, achieving rich text + math + bilingual rendering in one place. It animates into view with Framer Motion.

### 2.2.3 The six visualization modules

Each module follows a consistent layout — an interactive control area, a live visualization, and a `SeniorAdvice` summary — and targets a specific COMP 2211 topic.

**(a) NumpyModule — NumPy memory model (Lecture 1 / Lab 1).** This module visualizes the two NumPy concepts students most often get wrong: the *view vs. copy* distinction and *broadcasting*. A 4×4 matrix is displayed, and three indexing modes can be selected — basic `slice` (which returns a *view*, i.e. a reference to shared memory), `fancy` indexing (which returns a *copy*, i.e. new memory), and boolean `mask` indexing (also a *copy*). A highlight function colors exactly the cells each operation touches, and the side panel explains the memory semantics. A separate static diagram illustrates broadcasting, e.g. how shapes $(3,1) + (1,4) \rightarrow (3,4)$, paired with the rule "align from the trailing dimension; dimensions must be equal or one of them must be 1." The accompanying exam tip warns that broadcasting can only stretch dimensions of size 1, and that the dot product $\mathbf{A}\cdot\mathbf{B}$ and the element-wise product $\mathbf{A}\odot\mathbf{B}$ are physically different operations.

**(b) BackpropModule — backpropagation and the chain rule (Lecture 6 / Lab 6).** The left panel shows a small neuron graph ($i/j \rightarrow k$). Clicking "trace gradient" launches a Framer Motion animation in which an error particle flows *backward* from the output layer along the network (driven by an `isAnimating` flag that resets after ~2 s). An `errorVal` slider represents the output error $(T_k - O_k)$. The right "notebook" panel walks through the chain rule in three steps that mirror the lecture derivation:

$$\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial O_k}\cdot\frac{\partial O_k}{\partial \text{net}_k}\cdot\frac{\partial \text{net}_k}{\partial w_{jk}}, \qquad \delta_k = (T_k - O_k)\, f'(\text{net}_k), \qquad \Delta w_{jk} = \eta\,\delta_k\,O_j.$$

This makes the otherwise abstract "error signal flowing backward" concrete and connects each animated step to its formula.

**(c) KernelModule — convolution and edge detection (Lecture 7 / Lab 7).** A three-panel view shows an input image $I$, a convolution kernel $K$, and the resulting output feature map $I * K$. The learner can choose from preset kernels — **Sobel-X** and **Sobel-Y** (edge detection), **Laplacian** (second derivative / abrupt change), and **Identity** — or type a custom kernel (which switches the preset to "Custom"). The source image is an 8×8 grid with a central bright square; the module computes each output pixel via the 3×3 convolution, takes the absolute value, and clamps to $[0, 255]$ for grayscale rendering. This lets a student see *directly* how Sobel kernels respond to horizontal versus vertical edges and how a kernel acts as a local feature filter.

**(d) BayesBasicsModule — Bayes' theorem fundamentals (Lecture 2 / Lab 2).** The formula area presents

$$P(B\mid E) = \frac{P(B)\,P(E\mid B)}{P(E)},$$

with four colored blocks labeling the posterior, prior, likelihood, and evidence. A "fire alarm" worked example provides three sliders — $P(\text{Fire})$, $P(\text{Smoke})$, and $P(\text{Smoke}\mid\text{Fire})$ — and computes $P(\text{Fire}\mid\text{Smoke})$ in real time, showing the full substitution. The pedagogical payoff is that students can drive the prior to a small value and watch a strong piece of evidence still yield a low posterior, internalizing the lesson that "evidence ≠ conclusion."

**(e) NaiveBayesModule — Naïve Bayes inference and smoothing (Lecture 2 / Lab 2).** The module uses a hard-coded teaching dataset (a "disease Z" diagnosis example) stored in `BAYES_DATA`, with class priors ($P(\text{yes}) = 9/14$, $P(\text{no}) = 5/14$), per-feature conditional counts, and the number of values $m$ for each feature. Four feature dropdowns (BP / Fever / Diabetes / Vomit) let the student pose a query, and two controls expose the parts students most need to understand:

- a **smoothing coefficient** $\alpha$ implementing Laplace / m-estimate smoothing, $\;P = \dfrac{\text{count} + \alpha}{\text{total} + m\,\alpha}$, so the learner can see how smoothing eliminates the zero-probability problem; and
- a **log-space toggle**, contrasting the naïve product of likelihoods against a sum of log-likelihoods, illustrating how log-space avoids numerical underflow.

The output shows the normalized $P(\text{Yes})$ / $P(\text{No})$ as probability bars together with the term-by-term likelihood chain.

**(f) KnnModule — K-Nearest Neighbors (Lecture 3 / Lab 3).** A 400×400 SVG scatter plot renders a real teaching dataset (`KNN_RAW_DATA`: 18 height/weight samples labeled M or L). The student **clicks the canvas** to place a test point. A `useMemo`-derived pipeline then computes the distance from the test point to every sample, sorts them, takes the nearest $K$, draws a decision circle whose radius is the $K$-th distance, and predicts the class by majority vote. A **standardization toggle** switches between raw Euclidean distance and Z-score-standardized distance, demonstrating why feature scaling matters when height and weight have different units. A Recharts line chart plots a "K vs. error" model-complexity curve with a `ReferenceLine` marking the current $K$, conveying the bias-variance intuition that small $K$ overfits and large $K$ underfits. A voting panel shows the M/L tallies and the predicted class.

### 2.2.4 Internationalization

Internationalization is implemented with `react-i18next` plus `i18next-browser-languagedetector`. The resources are two JSON files, `en.json` and `zh.json`, organized by module namespace (e.g. `numpy_module.*`, `bayes.naive.*`, `knn.*`, `app.*`), with `fallbackLng: 'en'`. The non-trivial cases are sentences that mix prose, bold text, and inline formulas; these are authored with i18next's `<Trans>` component, injecting `<Latex>` and `<strong>` as placeholder components so that the same rich sentence renders correctly and consistently in both languages. A single header button toggles the language by calling `i18n.changeLanguage`. The principal maintenance constraint is that the key sets of the two JSON files must stay strictly aligned, and placeholder indices in `<Trans>` sentences must match across languages.

### 2.2.5 Deployment pipeline

KnowCS is continuously deployed through GitHub Actions to **PinMe**, a decentralized (IPFS) host, at <https://knowcs.pinme.dev>. A push to `main` triggers a pipeline that runs, in order: `npm ci` (install), `npm run lint` (ESLint), `npm run typecheck` (`tsc --noEmit`), and `npm run build` (`tsc --noEmit && vite build`, producing the single-file `dist/index.html`), and then publishes the artifact to the `knowcs` subdomain. Any failing step aborts the pipeline and prevents deployment, so lint, type-checking, and a successful build act as a three-part release gate. A path filter skips deployment when a push touches only documentation or configuration (`**.md`, `.gitignore`, `LICENSE`, `docs/**`), avoiding needless rebuilds. The PinMe app key is stored only as a GitHub repository secret and never committed.

## 2.3 Evaluation

The project is evaluated against the three objectives stated in Section 1.2.

**Objective 1 — Make abstract COMP 2211 formulas tangible through real-time interaction.** This objective is met. Six of the course's core topics — NumPy memory semantics, backpropagation, convolution, Bayes' theorem, Naïve Bayes, and KNN — are each realized as a working module in which the learner manipulates inputs and observes both the visualization and the live formula update instantly. In each module the interaction is tied directly to the exact mathematics the course teaches (the chain-rule decomposition, the smoothed conditional-probability formula, the standardized-distance comparison), rather than to a generic illustration. The breadth across six distinct topics, rather than depth in a single sandbox, is what distinguishes the tool from the prior art surveyed in Section 1.3.

**Objective 2 — Build a maintainable, zero-friction platform that is easy to deploy and extend.** This objective is met. The application requires no installation or login, runs entirely client-side, and ships as a single self-contained HTML file that can be hosted anywhere, including IPFS. The codebase was refactored from a single large file into a modular structure (thin shell + six independent modules + shared components + centralized typed data), and adding a new module is a well-defined, localized procedure. The automated CI/CD pipeline enforces lint, strict type-checking, and a successful build on every release, providing a quality gate without manual effort.

**Objective 3 — Serve both local and international students with a fully bilingual experience.** This objective is met. The entire interface, including formula-bearing rich-text explanations and per-module exam tips, is available in both English and Chinese and switches with a single click, with no duplicated application logic. The `<Trans>`-based approach keeps mathematics and formatting consistent across both languages.

Taken together, the platform delivers a working, deployed, bilingual, six-module interactive companion to COMP 2211 that satisfies all three objectives. The evaluation at this stage is primarily *qualitative and feature-based* — measuring what the platform does and verifying that each module computes and explains its target concept correctly. Quantitative evaluation of learning impact is identified as future work below.

## 2.4 Limitations and Further Study

**Teaching datasets are hard-coded.** The datasets (`BAYES_DATA`, `KNN_RAW_DATA`) and the convolution source image are fixed teaching examples bound to specific lecture material rather than general datasets. This is intentional for pedagogical focus, but a future improvement is to let students load or perturb their own small datasets to test edge cases.

**The KNN model-complexity curve is illustrative.** The "K vs. error" curve is currently synthesized to convey the bias-variance shape; it is not the result of a real cross-validation over the dataset. Replacing it with an actual leave-one-out or k-fold cross-validation computed in the browser would make the curve faithful as well as illustrative.

**No automated test coverage of the computational core.** The pure computational functions (distance, convolution, Bayesian probability) are not yet covered by unit tests. Because these functions are now cleanly separable after the modular refactor, introducing a test runner (e.g. Vitest) and testing the pure functions is the highest-priority engineering follow-up, serving as a safety net for future refactoring and data changes.

**Bilingual content is maintained by hand.** The `en.json` and `zh.json` files are kept aligned manually; a key-consistency check integrated into linting/CI would prevent accidental missing translations.

**Syllabus coverage is incomplete.** The six current modules cover a substantial part of COMP 2211 but not all of it. Mapped against the course's lectures and labs, notable gaps and partial coverage include: **K-Means clustering** (Lecture 4 — a high-frequency, hand-traced exam topic and the single largest coverage gap), the **single-neuron / perceptron** error-driven update (Lecture 5, distinct from the multilayer case), **data augmentation** and **CNN pooling / feature maps** extending the convolution module (Lectures 7–8), **PyTorch** tensors and autograd (Lecture 9), and **Minimax with Alpha-Beta pruning** (Lecture 10) — for which an interactive prototype already exists as a standalone HTML sandbox and is awaiting integration into the modular application. The roadmap prioritizes K-Means and the Naïve Bayes Gaussian-likelihood extension, both because they are the largest gaps and because past papers emphasize them as hand-traced questions where step-by-step interaction is most valuable.

**Evaluation is not yet empirical.** The current evaluation establishes that the platform works and is correct, but it has not yet been measured against actual student learning outcomes. A natural next step is to release a user survey to COMP 2211 students and gather feedback on which modules most improve their understanding, using the results to prioritize the roadmap.

---

# 3. Conclusion

This project set out to narrow the gap between the dense formulas of an introductory machine learning course and the intuition those formulas are meant to convey. The result, KnowCS, is a deployed, bilingual, zero-backend web application that reframes six core COMP 2211 topics — NumPy memory semantics, backpropagation, convolution, Bayes' theorem, Naïve Bayes, and K-Nearest Neighbors — as interactive experiments in which students manipulate parameters and watch both the visualization and the underlying mathematics respond in real time. By running entirely in the browser and packaging into a single self-contained file that is continuously deployed to a decentralized host, the platform is free to run, trivial to reach, and structured so that future contributors can add modules with a localized, well-defined procedure.

The motivation for choosing this topic was that interactive, hands-on tools demonstrably help students learn machine learning, yet no existing tool combines broad coverage of *one specific course's* syllabus with live, bilingual interaction. KnowCS aims to be exactly that: the interactive companion to COMP 2211 and, ultimately, an open-source supplement to the community's existing study notes. While there remain clear limitations — uncovered syllabus topics, illustrative rather than empirical curves, and an as-yet-untested computational core — the platform already provides a usable, correct, and extensible foundation. The next phase of work is well defined: close the highest-value syllabus gaps (K-Means first), make the model-complexity curve faithful, add automated tests, and evaluate the tool's learning impact directly with students. I hope KnowCS can grow into a sustainable, course-aligned learning resource that makes the abstract concepts of artificial intelligence a little more tangible for the students who come after me.

---

# 4. Hardware & Software

**Hardware**

| Product | Specification | Usage |
| --- | --- | --- |
| Development laptop | Personal laptop (macOS) | Development, build, and testing |
| Test devices | Desktop and mobile browsers | Cross-device verification of the responsive UI |

**Software**

| Product | Version | Usage | Price |
| --- | --- | --- | --- |
| React | 19 | UI framework | Free (MIT) |
| TypeScript | 5.9 (strict) | Static typing | Free (Apache-2.0) |
| Vite | 7 | Dev server / bundler | Free (MIT) |
| vite-plugin-singlefile | 2.3 | Single-file build | Free (MIT) |
| Tailwind CSS | 4 | Styling | Free (MIT) |
| Framer Motion | 12 | Animation | Free (MIT) |
| KaTeX | 0.16 | LaTeX formula rendering | Free (MIT) |
| Recharts | 3 | Charts (KNN complexity curve) | Free (MIT) |
| react-i18next / i18next | 16 / 25 | Internationalization | Free (MIT) |
| lucide-react | — | Icons | Free (ISC) |
| ESLint | 9 | Linting (deploy gate) | Free (MIT) |
| GitHub Actions | — | CI/CD | Free for the project |
| PinMe (IPFS) | — | Decentralized hosting | Wallet-balance based |

---

# 5. References

[1] React — A JavaScript library for building user interfaces. [Online]. Available: https://react.dev/. [Accessed: 31-May-2026].

[2] Vite — Next Generation Frontend Tooling. [Online]. Available: https://vitejs.dev/. [Accessed: 31-May-2026].

[3] Tailwind CSS — Rapidly build modern websites without leaving your HTML. [Online]. Available: https://tailwindcss.com/. [Accessed: 31-May-2026].

[4] Framer Motion — Production-ready animation library for React. [Online]. Available: https://www.framer.com/motion/. [Accessed: 31-May-2026].

[5] KaTeX — The fastest math typesetting library for the web. [Online]. Available: https://katex.org/. [Accessed: 31-May-2026].

[6] Recharts — A composable charting library built on React components. [Online]. Available: https://recharts.org/. [Accessed: 31-May-2026].

[7] react-i18next — Internationalization for React done right. [Online]. Available: https://react.i18next.com/. [Accessed: 31-May-2026].

[8] vite-plugin-singlefile — Inline all JavaScript and CSS into a single HTML file. [Online]. Available: https://github.com/richardtallent/vite-plugin-singlefile. [Accessed: 31-May-2026].

[9] HKUST COMP 2211 — Exploring Artificial Intelligence (Course Website). [Online]. Available: https://course.cse.ust.hk/comp2211/. [Accessed: 31-May-2026].

[10] Mo Yunxiang, "COMP2211 Study Notes." [Online]. Available: https://github.com/moyunxiang/COMP2211/blob/main/COMP2211.md. [Accessed: 31-May-2026].

[11] D. Smilkov and S. Carter, "TensorFlow Playground — A Neural Network Playground." [Online]. Available: https://playground.tensorflow.org/. [Accessed: 31-May-2026].

[12] G. Sanderson (3Blue1Brown) and the Manim Community, "Manim — Mathematical Animation Engine." [Online]. Available: https://www.3blue1brown.com/ and https://www.manim.community/. [Accessed: 31-May-2026].

[13] D. Kunin et al., "Seeing Theory — A visual introduction to probability and statistics," Brown University. [Online]. Available: https://seeing-theory.brown.edu/. [Accessed: 31-May-2026].

[14] Distill — Machine Learning Research (interactive articles). [Online]. Available: https://distill.pub/. [Accessed: 31-May-2026].

[15] PinMe — Decentralized hosting (IPFS). [Online]. Available: https://pinme.dev/. [Accessed: 31-May-2026].
