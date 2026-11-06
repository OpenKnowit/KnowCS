// Any program can import any of the sandbox's libraries. They are large, so each is its own chunk, fetched the
// first time a program names it; a page re-runs the program once the chunk has arrived.
import type { PyLib } from './minipy'

/** the standard library modules few programs import: one chunk (pyStdlibMore.ts) */
export const RARE_STDLIB = ['statistics', 'datetime', 'csv', 'argparse', 'pathlib', 'enum', 'contextlib', 'textwrap', 'types', 'tempfile', 'glob', 'shutil', 'fnmatch', 'inspect', 'fractions', 'uuid', 'secrets', 'platform', 'getpass']

const LOADERS: Record<string, () => Promise<PyLib>> = {
  pandas: () => import('./pyPandas').then((m) => m.PANDAS),
  matplotlib: () => import('./pyPlot').then((m) => m.MATPLOTLIB),
  mpl_toolkits: () => import('./pyPlot').then((m) => m.MATPLOTLIB),
  seaborn: () => import('./pyPlot').then((m) => m.MATPLOTLIB),
  torch: () => import('./pyTorch').then((m) => m.TORCH),
  torchvision: () => import('./pyTorch').then((m) => m.TORCH),
  keras: () => import('./pyKeras').then((m) => m.KERAS),
  tensorflow: () => import('./pyKeras').then((m) => m.KERAS),
  sklearn: () => import('./pySklearn').then((m) => m.SKLEARN),
  cv2: () => import('./pyCv2').then((m) => m.CV2),
  ...Object.fromEntries(RARE_STDLIB.map((name) => [name, () => import('./pyStdlibMore').then((m) => m.STDLIB_MORE)])),
}

/** what a program can import that arrives only when its text names it */
export const ON_DEMAND = Object.keys(LOADERS)

const loaded = new Map<string, PyLib>()
const failed = new Set<string>()
const loading = new Map<string, Promise<void>>()

// A statement starts a line, or follows ; or : (`try: import x`), or opens the string given to exec.
const STATEMENT = String.raw`(?:^|[;:]|\bexec\(\s*["'])[ \t]*`
const IMPORT = new RegExp(`${STATEMENT}import[ \\t]+([^#\\n;"']+)`, 'gm')
const FROM = new RegExp(`${STATEMENT}from[ \\t]+(\\w+)`, 'gm')

/** the top-level modules a program imports, as far as its text shows */
export const importsOf = (code: string): string[] => {
  const text = code.replace(/#[^\n]*/g, '').replace(/\\\n/g, ' ')
  const names = new Set<string>()
  for (const m of text.matchAll(IMPORT)) {
    for (const part of m[1].split(',')) names.add(part.trim().split(/[.\s]/)[0])
  }
  for (const m of text.matchAll(FROM)) names.add(m[1])
  return [...names]
}

/** the libraries fetched so far, for runPython's libs */
export const loadedLibs = (): PyLib[] => [...new Set(loaded.values())]

/** the libraries this program still has to wait for */
export const waitingFor = (code: string): string[] =>
  importsOf(code).filter((name) => LOADERS[name] && !loaded.has(name) && !failed.has(name))

/** fetch what the program needs; null when nothing is missing. A chunk that fails to load leaves the import to raise as usual. */
export const loadLibsFor = (code: string): Promise<void> | null => {
  const names = waitingFor(code)
  if (!names.length) return null
  return Promise.all(names.map((name) => {
    let p = loading.get(name)
    if (!p) {
      p = LOADERS[name]().then((lib) => { loaded.set(name, lib) }, () => { failed.add(name) }).finally(() => { loading.delete(name) })
      loading.set(name, p)
    }
    return p
  })).then(() => undefined)
}
