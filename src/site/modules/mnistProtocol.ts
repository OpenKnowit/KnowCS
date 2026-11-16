// Messages between the /mnist/ page and its training worker (mnistWorker.ts).
import type { TrainConfig } from '../../lib/mnistNet'

export type ToWorker =
  | { type: 'train-batch'; batch: number; images: Uint8Array; labels: Uint8Array }
  | { type: 'test'; images: Uint8Array; labels: Uint8Array; shown: number[] }
  | { type: 'run'; on: boolean }
  | { type: 'config'; cfg: TrainConfig }
  | { type: 'reset' }
  | { type: 'evaluate' }

export interface Point { seen: number; loss: number; train: number; val: number }

export interface Stats {
  type: 'stats'
  running: boolean
  seen: number
  /** training images loaded (the 1 in 10 kept for validation not counted) */
  trainSize: number
  loss: number | null
  trainAcc: number | null
  valAcc: number | null
  fwdMs: number | null
  bwdMs: number | null
  perSec: number | null
  points: Point[]
}

export interface Vis {
  type: 'vis'
  label: number
  /** each shown layer, channel by channel */
  layers: Float32Array[][]
  kernels: Float32Array[]
  probs: number[]
}

/** the shown test images: the top three classes and their probabilities */
export interface Preds { type: 'preds'; top: { k: number[]; p: number[] }[] }

export interface Evaluation {
  type: 'eval'
  seen: number
  accuracy: number
  /** confusion[true][predicted] */
  confusion: number[][]
  mistakes: { i: number; k: number; p: number }[]
}

export type FromWorker = Stats | Vis | Preds | Evaluation
