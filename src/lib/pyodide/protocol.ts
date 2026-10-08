// Messages between the page (client.ts) and the Pyodide worker (worker.ts).

export interface RealError {
  type: string
  message: string
  /** the editor line the error points at */
  line: number | null
  trace: string
}

export interface RealResult {
  stdout: string
  stderr: string
  /** repr of a last bare expression (like a notebook cell), null when there is none */
  value: string | null
  /** PNG data URLs */
  figures: string[]
  error: RealError | null
  /** library versions after the run ({ python: '3.14.2', numpy: '2.4.6' … }) */
  versions: Record<string, string>
  ms: number
}

export type RealPhase = { phase: 'runtime' } | { phase: 'packages'; names: string[] } | { phase: 'running' }

export type ToWorker = { id: number; base: string; code: string }

export type FromWorker =
  | { id: number; type: 'status'; status: RealPhase }
  | { id: number; type: 'done'; result: RealResult }
  /** Python or its packages could not be downloaded */
  | { id: number; type: 'failed'; message: string }
