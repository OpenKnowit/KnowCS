import { vi } from 'vitest'
import fs from 'node:fs'
vi.mock('../../src/lib/minipy', async (orig) => {
  const m = await orig() as Record<string, unknown>
  const real = m.runPython as (c: string, o?: unknown) => { stdout: string; error: unknown }
  return { ...m, runPython: (c: string, o?: unknown) => {
    const r = real(c, o)
    if (c && !r.error && r.stdout && c.length < 1800) fs.appendFileSync(process.env.REC_OUT!, JSON.stringify({ src: c, out: r.stdout.length }) + '\n')
    return r
  } }
})
