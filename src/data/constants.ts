import type { BayesData, KernelPresetName, KnnPoint } from '../types'

// --- Constants & Helper Data ---
export const INITIAL_MATRIX: number[][] = Array.from({ length: 4 }, (_, r) =>
  Array.from({ length: 4 }, (_, c) => r * 4 + c)
)

export const KERNEL_PRESETS: Record<KernelPresetName, number[][]> = {
  'Sobel-X': [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  'Sobel-Y': [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
  'Laplacian': [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
  'Identity': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}

export const BAYES_DATA: BayesData = {
  priors: { yes: 9 / 14, no: 5 / 14 },
  counts: {
    BP: { High: { yes: 2, no: 3 }, Normal: { yes: 3, no: 2 }, Low: { yes: 4, no: 0 } },
    Fever: { High: { yes: 2, no: 2 }, Mild: { yes: 4, no: 2 }, No: { yes: 3, no: 1 } },
    Diabetes: { Yes: { yes: 3, no: 4 }, No: { yes: 6, no: 1 } },
    Vomit: { Yes: { yes: 3, no: 3 }, No: { yes: 6, no: 2 } },
  },
  m_values: { BP: 3, Fever: 3, Diabetes: 2, Vomit: 2 },
}

export const KNN_RAW_DATA: KnnPoint[] = [
  { h: 158, w: 58, s: 'M' }, { h: 158, w: 59, s: 'M' }, { h: 158, w: 63, s: 'M' },
  { h: 160, w: 59, s: 'M' }, { h: 160, w: 60, s: 'M' }, { h: 163, w: 60, s: 'M' },
  { h: 163, w: 61, s: 'M' }, { h: 160, w: 64, s: 'L' }, { h: 163, w: 64, s: 'L' },
  { h: 165, w: 61, s: 'L' }, { h: 165, w: 62, s: 'L' }, { h: 165, w: 65, s: 'L' },
  { h: 168, w: 62, s: 'L' }, { h: 168, w: 63, s: 'L' }, { h: 168, w: 66, s: 'L' },
  { h: 170, w: 63, s: 'L' }, { h: 170, w: 64, s: 'L' }, { h: 170, w: 68, s: 'L' },
]
