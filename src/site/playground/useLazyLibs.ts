import { useEffect, useState } from 'react'
import { loadLibsFor, waitingFor } from '../../lib/pyLazy'

/**
 * Fetches the libraries a program imports. `waiting` is true until they have arrived, so a page does not flash
 * "No module named"; `tick` changes when they do, so the page runs the program again.
 */
export function useLazyLibs(code: string, active: boolean): { waiting: boolean; tick: number } {
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (!active) return
    const p = loadLibsFor(code)
    if (!p) return
    let live = true
    void p.then(() => {
      if (live) setTick((t) => t + 1)
    })
    return () => {
      live = false
    }
  }, [code, active])
  return { waiting: active && waitingFor(code).length > 0, tick }
}
