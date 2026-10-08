// The page's side of real Python: one shared worker, started on the first run and terminated by Stop.
import { pyodideBase } from './config'
import type { FromWorker, RealPhase, RealResult } from './protocol'

export type { RealPhase, RealResult } from './protocol'

/** Python or a package could not be downloaded (offline, CDN blocked …) */
export class RealLoadError extends Error {}
/** the run was stopped (Stop, or the worker was replaced) */
export class RealStopped extends Error {}

let worker: Worker | null = null
let warm = false
let nextId = 0
const pending = new Map<number, { onStatus: (s: RealPhase) => void; resolve: (r: RealResult) => void; reject: (e: Error) => void }>()

const spawn = (): Worker => {
  const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
  w.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data
    const p = pending.get(m.id)
    if (!p) return
    if (m.type === 'status') return p.onStatus(m.status)
    pending.delete(m.id)
    if (m.type === 'done') {
      warm = true
      p.resolve(m.result)
    } else {
      console.warn('[real python]', m.message)
      p.reject(new RealLoadError(m.message))
    }
  }
  w.onerror = (e) => {
    e.preventDefault()
    stopReal(new RealLoadError(e.message || 'worker error'))
  }
  return w
}

/** has Python already been downloaded and started in this tab? */
export const isRealWarm = () => warm

/** run code with real CPython and the real libraries; onStatus reports downloading and running */
export function runReal(code: string, onStatus: (s: RealPhase) => void): Promise<RealResult> {
  worker ??= spawn()
  const id = ++nextId
  return new Promise<RealResult>((resolve, reject) => {
    pending.set(id, { onStatus, resolve, reject })
    worker!.postMessage({ id, base: pyodideBase(), code })
  })
}

/** stop whatever is running: the worker is thrown away, so the next run starts Python again (from the browser cache) */
export function stopReal(reason: Error = new RealStopped('stopped')): void {
  worker?.terminate()
  worker = null
  warm = false
  for (const p of pending.values()) p.reject(reason)
  pending.clear()
}
