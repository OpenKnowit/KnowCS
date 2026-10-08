import { runPython } from '../../lib/minipy'
import type { RunResult } from '../../lib/minipy'
import type { PlayConfig } from './types'

/** run code with a playground's libraries and limits */
export const runIn = (config: PlayConfig, code: string, budget?: number): RunResult =>
  runPython(code, { libs: config.libs, maxSize: config.maxSize, timeBudget: budget ?? config.timeBudget })
