import { useEffect, useState, useSyncExternalStore } from 'react'
import { RealLoadError, RealSuperseded, RealTimeout, engineState, onEngineState, runReal, stopReal, warmReal } from '../../lib/pyodide/client'
import type { EngineState, RealPhase, RealResult } from '../../lib/pyodide/client'

// Which engine runs the playground's code: minipy (every step drawn) or real CPython (Pyodide in a worker).
// The choice is shared by every panel on the page and remembered in this browser.

export type Engine = 'mini' | 'real'

const KEY = 'knowcs-py-engine'
const read = (): Engine => {
  try {
    return localStorage.getItem(KEY) === 'real' ? 'real' : 'mini'
  } catch {
    return 'mini'
  }
}
let chosen: Engine = typeof window === 'undefined' ? 'mini' : read()
const subs = new Set<() => void>()
const choose = (e: Engine) => {
  chosen = e
  try {
    localStorage.setItem(KEY, e)
  } catch {
    /* private mode: the choice lasts for this page */
  }
  subs.forEach((f) => f())
}
const subscribe = (f: () => void) => {
  subs.add(f)
  return () => void subs.delete(f)
}

export function useEngineChoice(): [Engine, (e: Engine) => void] {
  return [useSyncExternalStore(subscribe, () => chosen, () => 'mini' as Engine), choose]
}

export const useEngineState = (): EngineState => useSyncExternalStore(onEngineState, engineState, () => 'cold' as EngineState)

/** start downloading Python soon after a playground that can use it opens (not on a data-saving connection) */
export function useWarmEngine(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return
    const conn = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection
    if (conn?.saveData) return
    const id = setTimeout(warmReal, 1200)
    return () => clearTimeout(id)
  }, [enabled])
}

export type RealRun =
  | { s: 'idle' }
  /** prev: the last finished run, still shown (faded) while this one works */
  | { s: 'busy'; phase: RealPhase; prev: RealResult | null }
  | { s: 'done'; r: RealResult }
  | { s: 'failed'; why: 'load' | 'stopped' | 'timeout' }

const lastResult = (r: RealRun): RealResult | null => (r.s === 'done' ? r.r : r.s === 'busy' ? r.prev : null)

/** run code with real Python whenever it changes (while active); a newer run replaces an older one */
export function useRealRun(code: string, active: boolean) {
  const [run, setRun] = useState<RealRun>({ s: 'idle' })
  const [nonce, setNonce] = useState(0)
  useEffect(() => {
    if (!active) return
    let live = true
    const mine = () => live
    const busy = (phase: RealPhase) => setRun((p) => ({ s: 'busy', phase, prev: lastResult(p) }))
    // queueMicrotask: the state change belongs to the run starting, not to the render that asked for it
    queueMicrotask(() => mine() && busy({ phase: engineState() === 'ready' ? 'running' : 'runtime' }))
    runReal(code, (phase) => mine() && busy(phase))
      .then((r) => mine() && setRun({ s: 'done', r }))
      .catch((e: unknown) => {
        if (e instanceof RealSuperseded || !mine()) return
        setRun({ s: 'failed', why: e instanceof RealLoadError ? 'load' : e instanceof RealTimeout ? 'timeout' : 'stopped' })
      })
    return () => {
      live = false
    }
  }, [code, active, nonce])
  return { run, rerun: () => setNonce((n) => n + 1), stop: () => stopReal(), busy: active && run.s === 'busy' }
}

/** the editor line real Python's error points at */
export const realErrorLine = (run: RealRun): number | undefined => (run.s === 'done' ? (run.r.error?.line ?? undefined) : undefined)
