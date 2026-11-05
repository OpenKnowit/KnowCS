// Any program can import any of the sandbox's libraries. They are large, so each is its own chunk, fetched the
// first time a program names it; a page re-runs the program once the chunk has arrived.
import type { PyLib } from './minipy'

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
}

const loaded = new Map<string, PyLib>()
const failed = new Set<string>()
const loading = new Map<string, Promise<void>>()

/** the top-level modules a program imports, as far as its text shows */
export const importsOf = (code: string): string[] => {
  const names = new Set<string>()
  for (const m of code.matchAll(/^[ \t]*import[ \t]+([^#\n]+)/gm)) {
    for (const part of m[1].split(',')) names.add(part.trim().split(/[.\s]/)[0])
  }
  for (const m of code.matchAll(/^[ \t]*from[ \t]+(\w+)/gm)) names.add(m[1])
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
