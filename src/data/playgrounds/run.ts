import { runPython } from '../../lib/minipy'
import type { RunResult } from '../../lib/minipy'
import { loadedLibs } from '../../lib/pyLazy'
import type { PlayConfig } from './types'

/** run code with a playground's libraries and limits */
export const runIn = (config: PlayConfig, code: string, budget?: number): RunResult =>
  runPython(code, { libs: [...new Set([...config.libs, ...loadedLibs()])], maxSize: config.maxSize, timeBudget: budget ?? config.timeBudget })
