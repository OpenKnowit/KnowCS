// --- 全站页面表：每个模块 / 笔记 / 拓展各是一个独立 HTML 页面；旧的 #/… 与 /lab/*.html 链接重定向到新地址 ---

export type ModuleId =
  | 'numpy' | 'bayes-basics' | 'naive-bayes' | 'bayes-virus' | 'gaussian-nb' | 'knn' | 'evaluation' | 'cross-validation'
  | 'kmeans' | 'kmeans-table' | 'perceptron' | 'backprop' | 'xor-mlp' | 'kernel' | 'convolution' | 'otsu' | 'affine' | 'cnn-shapes' | 'alphabeta'

export interface ModuleInfo {
  id: ModuleId
  /** lecture number (1–11) */
  lec: number
  /** papers that asked this pattern, e.g. '22F' (midterm) or 'F24' (final) */
  exams: string[]
  /** 'classic' = original site module, 'lab' = page added with the lab redesign */
  kind: 'classic' | 'lab'
  /** id used by the old single-page site (#/course/<legacy>) */
  legacy?: string
}

// Order = reading order (prev / next): by lecture, basics before exam drills.
export const MODULES: ModuleInfo[] = [
  { id: 'numpy', lec: 1, kind: 'classic', legacy: 'numpy', exams: ['22F', '22S', '23F', '23S', '24S', '25S'] },
  { id: 'bayes-basics', lec: 2, kind: 'classic', legacy: 'bayesBasics', exams: [] },
  { id: 'bayes-virus', lec: 2, kind: 'lab', exams: ['22F', '22S', '25S'] },
  { id: 'naive-bayes', lec: 2, kind: 'classic', legacy: 'naiveBayes', exams: ['22F', '22S', '23F', '24S', 'F22', 'F23'] },
  { id: 'gaussian-nb', lec: 2, kind: 'lab', exams: ['23F', '24S', 'F24'] },
  { id: 'knn', lec: 3, kind: 'classic', legacy: 'knn', exams: ['22F', '22S', '23F', '23S', '24S', '25S', 'F22'] },
  { id: 'evaluation', lec: 3, kind: 'lab', exams: ['23F', '24S', 'F22'] },
  { id: 'cross-validation', lec: 3, kind: 'lab', exams: ['22F', '23F', '24S', 'F24'] },
  { id: 'kmeans', lec: 4, kind: 'classic', legacy: 'kmeans', exams: [] },
  { id: 'kmeans-table', lec: 4, kind: 'lab', exams: ['22F', '22S', '23F', '23S', '25S', 'F23'] },
  { id: 'perceptron', lec: 5, kind: 'lab', exams: ['22F', '22S', '23F', '23S'] },
  { id: 'backprop', lec: 6, kind: 'classic', legacy: 'backprop', exams: [] },
  { id: 'xor-mlp', lec: 6, kind: 'lab', exams: ['22S', '23S', '25S', 'F23', 'F24'] },
  { id: 'kernel', lec: 7, kind: 'classic', legacy: 'kernel', exams: [] },
  { id: 'convolution', lec: 7, kind: 'lab', exams: ['F22', 'F23', 'F24'] },
  { id: 'otsu', lec: 7, kind: 'lab', exams: ['F22', 'F24'] },
  { id: 'affine', lec: 7, kind: 'lab', exams: ['F22', 'F24'] },
  { id: 'cnn-shapes', lec: 8, kind: 'lab', exams: ['F22', 'F23', 'F24'] },
  { id: 'alphabeta', lec: 10, kind: 'classic', legacy: 'alphabeta', exams: ['F22', 'F23', 'F24'] },
]

/** Explainer videos (/watch/<id>/): in-browser animations with captions, one per heavily examined idea. */
export interface EpisodeInfo {
  id: string
  lec: number
  /** practice pages that go with it */
  modules: ModuleId[]
  exams: string[]
  /** thumbnail frame (ms into the episode) */
  thumb: number
}

export const EPISODES: EpisodeInfo[] = [
  { id: 'broadcast', lec: 1, modules: ['numpy'], exams: ['22F', '22S', '23F', '23S', '24S', '25S'], thumb: 86_000 },
  { id: 'bayes', lec: 2, modules: ['bayes-virus', 'naive-bayes', 'bayes-basics'], exams: ['22F', '22S', '23F', '24S', '25S', 'F22', 'F23', 'F24'], thumb: 62_000 },
  { id: 'gaussian', lec: 2, modules: ['gaussian-nb', 'naive-bayes'], exams: ['23F', '24S', 'F24'], thumb: 40_000 },
  { id: 'knn', lec: 3, modules: ['knn', 'evaluation', 'cross-validation'], exams: ['22F', '22S', '23F', '23S', '24S', '25S', 'F22'], thumb: 71_000 },
  { id: 'evaluate', lec: 3, modules: ['evaluation', 'cross-validation'], exams: ['22F', '23F', '24S', 'F22', 'F24'], thumb: 42_000 },
  { id: 'kmeans', lec: 4, modules: ['kmeans-table', 'kmeans'], exams: ['22F', '22S', '23F', '23S', '25S', 'F23'], thumb: 40_000 },
  { id: 'perceptron', lec: 5, modules: ['perceptron'], exams: ['22F', '22S', '23F', '23S'], thumb: 66_000 },
  { id: 'xor', lec: 6, modules: ['xor-mlp', 'perceptron', 'backprop'], exams: ['22S', '23S', '25S', 'F23', 'F24'], thumb: 102_000 },
  { id: 'backprop', lec: 6, modules: ['xor-mlp', 'backprop'], exams: ['F22', 'F23', 'F24'], thumb: 62_000 },
  { id: 'imagenp', lec: 7, modules: ['convolution', 'numpy', 'kernel'], exams: ['F24'], thumb: 52_000 },
  { id: 'otsu', lec: 7, modules: ['otsu', 'convolution'], exams: ['F22', 'F24'], thumb: 58_000 },
  { id: 'affine', lec: 7, modules: ['affine'], exams: ['F22', 'F24'], thumb: 81_000 },
  { id: 'cnn', lec: 8, modules: ['convolution', 'cnn-shapes', 'kernel'], exams: ['F22', 'F23', 'F24'], thumb: 112_000 },
  { id: 'dilated', lec: 8, modules: ['convolution', 'cnn-shapes'], exams: ['F24'], thumb: 50_000 },
  { id: 'autograd', lec: 9, modules: [], exams: ['F22'], thumb: 70_000 },
  { id: 'alphabeta', lec: 10, modules: ['alphabeta'], exams: ['F22', 'F23', 'F24'], thumb: 52_000 },
  { id: 'ethics', lec: 11, modules: [], exams: ['F23', 'F24'], thumb: 40_000 },
]

export const NOTE_IDS = ['numpy', 'pandas', 'pytorch', 'tensorflow', 'keras', 'kevin'] as const
export const EXTEND_IDS = ['attention'] as const

export const LECTURES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] as const
/** midterms cover lectures 1–6; finals cover everything */
export const lectureScope = (n: number): 'mid' | 'final' => (n <= 6 ? 'mid' : 'final')

export interface Paper {
  id: string
  kind: 'M' | 'F'
  year: number
  term?: 'Fall' | 'Spring'
}

/** The nine papers with solutions that the site's exam badges refer to. */
export const PAPERS: Paper[] = [
  { id: '22F', kind: 'M', year: 2022, term: 'Fall' },
  { id: '22S', kind: 'M', year: 2022, term: 'Spring' },
  { id: '23F', kind: 'M', year: 2023, term: 'Fall' },
  { id: '23S', kind: 'M', year: 2023, term: 'Spring' },
  { id: '24S', kind: 'M', year: 2024, term: 'Spring' },
  { id: '25S', kind: 'M', year: 2025, term: 'Spring' },
  { id: 'F22', kind: 'F', year: 2022 },
  { id: 'F23', kind: 'F', year: 2023 },
  { id: 'F24', kind: 'F', year: 2024 },
]

export const moduleUrl = (id: string): string => `/${id}/`
export const noteUrl = (id?: string): string => (id ? `/notes/${id}/` : '/notes/')
export const drillUrl = (): string => '/drill/'
export const formulasUrl = (): string => '/formulas/'
export const papersUrl = (): string => '/papers/'
export const watchUrl = (id?: string): string => (id ? `/watch/${id}/` : '/watch/')
export const extendUrl = (id?: string): string => (id ? `/extend/${id}/` : '/extend/')

export const neighbours = (id: ModuleId): { prev: ModuleInfo | null; next: ModuleInfo | null } => {
  const i = MODULES.findIndex((m) => m.id === id)
  return { prev: i > 0 ? MODULES[i - 1] : null, next: i >= 0 && i < MODULES.length - 1 ? MODULES[i + 1] : null }
}

/** Where an old single-page-site hash (#/course/knn, #/package/numpy, …) now lives; null = stay on the home page. */
export const legacyTarget = (hash: string): string | null => {
  let parts: string[]
  try {
    parts = hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent)
  } catch {
    return null
  }
  const [mode, id] = parts
  if (mode === 'course') {
    const m = MODULES.find((x) => x.legacy === id)
    return m ? moduleUrl(m.id) : null
  }
  if (mode === 'package') return id && (NOTE_IDS as readonly string[]).includes(id) ? noteUrl(id) : noteUrl()
  if (mode === 'extend') return id && (EXTEND_IDS as readonly string[]).includes(id) ? extendUrl(id) : extendUrl()
  return null
}

/** Old /lab/<file>.html prototype URLs → new pages. */
export const LAB_REDIRECTS: Record<string, string> = {
  index: '/',
  'home-a': '/',
  'home-b': '/',
  'home-c': '/',
  // the ten /lab/ prototypes that existed before the merge; later lab pages never had a prototype URL
  ...Object.fromEntries(
    (['bayes-virus', 'gaussian-nb', 'evaluation', 'cross-validation', 'kmeans-table', 'perceptron', 'xor-mlp', 'convolution', 'otsu', 'cnn-shapes'] as const).map((id) => [id, moduleUrl(id)]),
  ),
}

export type PageSpec =
  | { kind: 'home'; path: string }
  | { kind: 'module'; path: string; id: ModuleId }
  | { kind: 'notes'; path: string }
  | { kind: 'note'; path: string; id: string }
  | { kind: 'extend'; path: string }
  | { kind: 'extend-item'; path: string; id: string }
  | { kind: 'watch'; path: string }
  | { kind: 'drill'; path: string }
  | { kind: 'formulas'; path: string }
  | { kind: 'papers'; path: string }
  | { kind: 'notfound'; path: string }
  | { kind: 'watch-item'; path: string; id: string }
  | { kind: 'redirect'; path: string; to: string }

/** Every HTML file the build emits. path is the output file, relative to the site root. */
export const PAGES: PageSpec[] = [
  { kind: 'home', path: 'index.html' },
  ...MODULES.map((m): PageSpec => ({ kind: 'module', path: `${m.id}/index.html`, id: m.id })),
  { kind: 'watch', path: 'watch/index.html' },
  { kind: 'drill', path: 'drill/index.html' },
  { kind: 'formulas', path: 'formulas/index.html' },
  { kind: 'papers', path: 'papers/index.html' },
  { kind: 'notfound', path: '404.html' },
  ...EPISODES.map((e): PageSpec => ({ kind: 'watch-item', path: `watch/${e.id}/index.html`, id: e.id })),
  { kind: 'notes', path: 'notes/index.html' },
  ...NOTE_IDS.map((id): PageSpec => ({ kind: 'note', path: `notes/${id}/index.html`, id })),
  { kind: 'extend', path: 'extend/index.html' },
  ...EXTEND_IDS.map((id): PageSpec => ({ kind: 'extend-item', path: `extend/${id}/index.html`, id })),
  ...Object.entries(LAB_REDIRECTS).map(([from, to]): PageSpec => ({ kind: 'redirect', path: from === 'index' ? 'lab/index.html' : `lab/${from}.html`, to })),
]
