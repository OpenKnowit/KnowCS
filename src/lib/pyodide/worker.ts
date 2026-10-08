/// <reference lib="webworker" />
// The Pyodide worker: downloads CPython (once) and the packages each run imports, then runs the code with harness.py.
// It lives off the main thread so a long run never freezes the page, and Stop can simply terminate it.
// Everything fetched from the Pyodide base (the wasm runtime, the standard library, package wheels) is kept in
// Cache Storage, so the next visit starts Python without downloading it again.
import HARNESS from './harness.py?raw'
import { MICROPIP_IMPORTS, packageOf } from './config'
import type { FromWorker, ToWorker } from './protocol'

interface PyProxy {
  (...args: unknown[]): unknown
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

const CACHE_PREFIX = 'knowcs-pyodide:'

/** fetch from the Pyodide base through Cache Storage: cache first, store what was downloaded; old versions are dropped */
const cacheFetches = (base: string) => {
  if (typeof caches === 'undefined') return // not a secure context: the browser's HTTP cache still helps
  const name = CACHE_PREFIX + base
  void caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith(CACHE_PREFIX) && k !== name).map((k) => caches.delete(k)))).catch(quiet)
  const plain = self.fetch.bind(self)
  self.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    if (!url.startsWith(base) || (init?.method ?? 'GET') !== 'GET') return plain(input, init)
    let cache: Cache
    try {
      cache = await caches.open(name)
      const hit = await cache.match(url)
      if (hit) return hit
    } catch {
      return plain(input, init)
    }
    const res = await plain(input, init)
    if (res.ok) cache.put(url, res.clone()).catch(quiet) // a full disk only costs the cache
    return res
  }
}

let booting: Promise<Pyodide> | null = null

const boot = async (base: string): Promise<Pyodide> => {
  cacheFetches(base)
  const mod = (await import(/* @vite-ignore */ `${base}pyodide.mjs`)) as { loadPyodide(o: object): Promise<Pyodide> }
  const py = await mod.loadPyodide({ indexURL: base, stdout: quiet, stderr: quiet })
  py.runPython(HARNESS)
  return py
}

const start = (base: string): Promise<Pyodide> => {
  if (!booting) {
    booting = boot(base)
    booting.then(
      () => post({ type: 'engine', state: 'ready' }),
      (err) => {
        booting = null
        post({ type: 'engine', state: 'failed', message: String(err) })
      },
    )
  }
  return booting
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

const run = async (id: number, base: string, code: string) => {
  const started = performance.now()
  let py: Pyodide
  try {
    if (!booting) post({ id, type: 'status', status: { phase: 'runtime' } })
    py = await start(base)
  } catch (err) {
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
  const exec = py.globals.get('knowcs_run')
  const versions = py.globals.get('knowcs_versions')
  try {
    const out = JSON.parse(exec(code, /\bmatplotlib\b|\bseaborn\b/.test(code)) as string)
    post({ id, type: 'done', result: { ...out, versions: JSON.parse(versions() as string), ms: Math.round(performance.now() - started) } })
  } catch (err) {
    post({ id, type: 'failed', message: String(err) })
  } finally {
    exec.destroy()
    versions.destroy()
  }
}

// the page sends one run at a time (client.ts queues them), so runs never overlap here
self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data
  if (m.type === 'boot') void start(m.base).catch(quiet)
  else void run(m.id, m.base, m.code)
}
