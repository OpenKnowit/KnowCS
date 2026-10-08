// --- 各库实验台（matplotlib / PyTorch / Keras / TensorFlow / pandas）共用的配置类型 ---
import type { PyLib } from '../../lib/minipy'

export type PlaygroundId = 'matplotlib' | 'pytorch' | 'keras' | 'tensorflow' | 'pandas'

export interface PlayEntry {
  id: string
  cat: string
  /** button text (code-like, not translated); the explanation is playground.<id>.e.<entry id> */
  label: string
  code: string
  /** step to open first: an api name ('torch.matmul') or an event type ('figure', 'flow' …); default = the last step */
  focus?: string[]
  /** the example ends in this error on purpose (it demonstrates a mistake) */
  expectError?: string
}

export interface PlayConfig {
  id: PlaygroundId
  libs: PyLib[]
  /** element limit for one array in this sandbox */
  maxSize: number
  /** ms before a run is stopped */
  timeBudget: number
  /** ms after the last keystroke before re-running */
  debounce: number
  cats: string[]
  entries: PlayEntry[]
  /** imports put in front of a note's code block that has none */
  prelude: string
  /** which traced calls are listed as steps, by api prefix ('torch.', 'op:' …); [] = only library events */
  callApis: string[]
}
