// The page's side of real Python: one shared worker. It boots in the background once a playground opens (warmReal),
// runs one piece of code at a time (a newer request replaces one still waiting), and Stop or a run that takes too
// long terminates it; a fresh worker then starts again from the cache.
import { pyodideBase } from './config'
import type { FromWorker, RealPhase, RealResult } from './protocol'

export type { RealPhase, RealResult } from './protocol'

/** Python or a package could not be downloaded (offline, CDN blocked …) */
export class RealLoadError extends Error {}
/** the run was stopped (Stop) */
export class RealStopped extends Error {}
/** the run took longer than RUN_LIMIT_MS (an endless loop?) */
export class RealTimeout extends Error {}
/** a newer run replaced this one before it started */
export class RealSuperseded extends Error {}

/** the most a run may take once its packages are loaded */
export const RUN_LIMIT_MS = 15_000

export type EngineState = 'cold' | 'loading' | 'ready' | 'failed'

interface Job {
  code: string
  onStatus: (s: RealPhase) => void
  resolve: (r: RealResult) => void
  reject: (e: Error) => void
}

let worker: Worker | null = null
let engine: EngineState = 'cold'
let seenReady = false
const listeners = new Set<(s: EngineState) => void>()
let nextId = 0
let current: (Job & { id: number }) | null = null
let queued: Job | null = null
let timer: ReturnType<typeof setTimeout> | undefined

const setEngine = (s: EngineState) => {
  engine = s
  if (s === 'ready') seenReady = true
  listeners.forEach((f) => f(s))
}

export const engineState = () => engine
/** has Python started in this tab before (a restart after Stop then comes from the cache: no size warning) */
export const engineSeenReady = () => seenReady
export const onEngineState = (f: (s: EngineState) => void) => {
  listeners.add(f)
  return () => void listeners.delete(f)
}

const finish = () => {
  clearTimeout(timer)
  current = null
  const next = queued
  queued = null
  if (next) start(next)
}

const spawn = (): Worker => {
  const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  w.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data
    if (m.type === 'engine') {
      if (m.state === 'failed') console.warn('[real python]', m.message)
      return setEngine(m.state)
    }
    if (!current || m.id !== current.id) return
    const job = current
    if (m.type === 'status') {
      if (m.status.phase === 'running') timer = setTimeout(() => stopReal(new RealTimeout('timeout')), RUN_LIMIT_MS)
      return job.onStatus(m.status)
    }
    finish()
    if (m.type === 'done') job.resolve(m.result)
    else {
      console.warn('[real python]', m.message)
      job.reject(new RealLoadError(m.message))
    }
  }
  w.onerror = (e) => {
    e.preventDefault()
    stopReal(new RealLoadError(e.message || 'worker error'))
  }
  return w
}

const ensureWorker = (): Worker => {
  if (!worker) {
    worker = spawn()
    setEngine('loading')
  }
  return worker
}

function start(job: Job) {
  const w = ensureWorker()
  if (engine === 'failed') setEngine('loading') // the worker tries the download again
  current = { ...job, id: ++nextId }
  w.postMessage({ id: current.id, base: pyodideBase(), type: 'run', code: job.code })
}

/** download and start Python in the background (the packages still wait for an import) */
export function warmReal(): void {
  if (worker && engine !== 'failed') return
  if (engine === 'failed') stopWorker()
  ensureWorker().postMessage({ id: 0, base: pyodideBase(), type: 'boot' })
}

/** run code with real CPython and the real libraries; onStatus reports downloading and running */
export function runReal(code: string, onStatus: (s: RealPhase) => void): Promise<RealResult> {
  return new Promise<RealResult>((resolve, reject) => {
    const job = { code, onStatus, resolve, reject }
    if (!current) return start(job)
    queued?.reject(new RealSuperseded('superseded'))
    queued = job
  })
}

function stopWorker() {
  worker?.terminate()
  worker = null
  setEngine('cold')
}

/** stop whatever is running; a new worker boots again at once (from the cache), so the next run is quick */
export function stopReal(reason: Error = new RealStopped('stopped')): void {
  clearTimeout(timer)
  const jobs = [current, queued]
  current = null
  queued = null
  stopWorker()
  jobs.forEach((j) => j?.reject(reason))
  if (!(reason instanceof RealLoadError)) warmReal()
}
