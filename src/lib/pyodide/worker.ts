/// <reference lib="webworker" />
// The Pyodide worker: downloads CPython (once) and the packages each run imports, then runs the code with harness.py.
// It lives off the main thread so a long run never freezes the page, and Stop can simply terminate it.
import HARNESS from './harness.py?raw'
import { MICROPIP_IMPORTS, packageOf } from './config'
import type { FromWorker, ToWorker } from './protocol'

interface PyProxy {
  (...args: unknown[]): unknown
  toJs(): unknown
  destroy(): void
}
interface Pyodide {
  runPython(code: string): unknown
  loadPackage(names: string | string[], options?: { messageCallback?: (m: string) => void }): Promise<unknown>
  loadPackagesFromImports(code: string, options?: { messageCallback?: (m: string) => void }): Promise<unknown>
  pyimport(name: string): { install(name: string): Promise<void> }
  globals: { get(name: string): PyProxy }
}

const post = (m: FromWorker) => self.postMessage(m)
const quiet = () => {}

let booting: Promise<Pyodide> | null = null

const boot = async (base: string): Promise<Pyodide> => {
  const mod = (await import(/* @vite-ignore */ `${base}pyodide.mjs`)) as { loadPyodide(o: object): Promise<Pyodide> }
  const py = await mod.loadPyodide({ indexURL: base, stdout: quiet, stderr: quiet })
  py.runPython(HARNESS)
  return py
}

/** the imports that still need downloading (not built in, not loaded yet) */
const missing = (py: Pyodide, code: string): string[] => {
  const find = py.globals.get('knowcs_missing')
  try {
    return JSON.parse(find(code) as string) as string[]
  } finally {
    find.destroy()
  }
}

let queue: Promise<unknown> = Promise.resolve()

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const { id, base, code } = e.data
  // one run at a time: Python has one global interpreter
  queue = queue.then(async () => {
    const started = performance.now()
    let py: Pyodide
    try {
      if (!booting) post({ id, type: 'status', status: { phase: 'runtime' } })
      booting ??= boot(base)
      py = await booting
    } catch (err) {
      booting = null
      post({ id, type: 'failed', message: String(err) })
      return
    }
    try {
      const need = missing(py, code)
      if (need.length) {
        post({ id, type: 'status', status: { phase: 'packages', names: need.map(packageOf) } })
        await py.loadPackagesFromImports(code, { messageCallback: quiet })
        const extra = need.filter((m) => m in MICROPIP_IMPORTS)
        if (extra.length) {
          await py.loadPackage('micropip', { messageCallback: quiet })
          const micropip = py.pyimport('micropip')
          for (const m of extra) await micropip.install(MICROPIP_IMPORTS[m])
        }
      }
    } catch (err) {
      post({ id, type: 'failed', message: String(err) })
      return
    }
    post({ id, type: 'status', status: { phase: 'running' } })
    const run = py.globals.get('knowcs_run')
    const versions = py.globals.get('knowcs_versions')
    try {
      const out = JSON.parse(run(code, /\bmatplotlib\b|\bseaborn\b/.test(code)) as string)
      post({ id, type: 'done', result: { ...out, versions: JSON.parse(versions() as string), ms: Math.round(performance.now() - started) } })
    } catch (err) {
      post({ id, type: 'failed', message: String(err) })
    } finally {
      run.destroy()
      versions.destroy()
    }
  })
}
