import type { AbNode, BayesData, KernelPresetName, KnnPoint, Vec2 } from '../types'

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

// --- Alpha-Beta 剪枝：固定 3 层满二叉树（MAX→MIN→MAX→叶），坐标用于 SVG 布局 ---
export const AB_NODES: Record<string, AbNode> = {
  A: { id: 'A', type: 'max', parent: null, children: ['B', 'C'], x: 500, y: 60 },
  B: { id: 'B', type: 'min', parent: 'A', children: ['D', 'E'], x: 260, y: 170 },
  C: { id: 'C', type: 'min', parent: 'A', children: ['F', 'G'], x: 740, y: 170 },
  D: { id: 'D', type: 'max', parent: 'B', children: ['L1', 'L2'], x: 140, y: 280 },
  E: { id: 'E', type: 'max', parent: 'B', children: ['L3', 'L4'], x: 380, y: 280 },
  F: { id: 'F', type: 'max', parent: 'C', children: ['L5', 'L6'], x: 620, y: 280 },
  G: { id: 'G', type: 'max', parent: 'C', children: ['L7', 'L8'], x: 860, y: 280 },
  L1: { id: 'L1', type: 'leaf', parent: 'D', value: 3, x: 75, y: 400 },
  L2: { id: 'L2', type: 'leaf', parent: 'D', value: 5, x: 205, y: 400 },
  L3: { id: 'L3', type: 'leaf', parent: 'E', value: 2, x: 315, y: 400 },
  L4: { id: 'L4', type: 'leaf', parent: 'E', value: 4, x: 445, y: 400 },
  L5: { id: 'L5', type: 'leaf', parent: 'F', value: 6, x: 555, y: 400 },
  L6: { id: 'L6', type: 'leaf', parent: 'F', value: 1, x: 685, y: 400 },
  L7: { id: 'L7', type: 'leaf', parent: 'G', value: 0, x: 795, y: 400 },
  L8: { id: 'L8', type: 'leaf', parent: 'G', value: 7, x: 925, y: 400 },
}

export const AB_LEAF_IDS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'] as const

export const AB_PRESETS: Record<string, number[]> = {
  user: [3, 5, 2, 4, 6, 1, 0, 7], // 默认示例
  beta: [3, 5, 6, 2, 1, 4, 8, 9], // 演示单分支 Beta 剪枝
  alpha: [6, 7, 8, 9, 1, 2, 0, 4], // 演示子树级 Alpha 剪枝
}

// --- K-Means：三簇二维教学数据 ---
export const KMEANS_DATA: Vec2[] = [
  // 左下簇
  { x: 1.2, y: 1.5 }, { x: 1.8, y: 1.1 }, { x: 1.0, y: 2.2 }, { x: 2.1, y: 1.8 }, { x: 1.5, y: 1.0 }, { x: 2.3, y: 2.4 },
  // 右下簇
  { x: 7.2, y: 1.8 }, { x: 7.8, y: 1.2 }, { x: 8.1, y: 2.1 }, { x: 7.5, y: 2.6 }, { x: 8.4, y: 1.5 }, { x: 6.9, y: 1.1 },
  // 上中簇
  { x: 4.2, y: 7.5 }, { x: 4.8, y: 8.1 }, { x: 5.1, y: 7.2 }, { x: 4.5, y: 6.8 }, { x: 5.4, y: 8.0 }, { x: 3.9, y: 7.8 },
]

// 固定初始质心（教学可复现，刻意放偏以展示收敛过程）
export const KMEANS_INIT_CENTROIDS: Vec2[] = [
  { x: 2.0, y: 6.0 },
  { x: 5.0, y: 3.0 },
  { x: 6.0, y: 6.0 },
]

export const KNN_RAW_DATA: KnnPoint[] = [
  { h: 158, w: 58, s: 'M' }, { h: 158, w: 59, s: 'M' }, { h: 158, w: 63, s: 'M' },
  { h: 160, w: 59, s: 'M' }, { h: 160, w: 60, s: 'M' }, { h: 163, w: 60, s: 'M' },
  { h: 163, w: 61, s: 'M' }, { h: 160, w: 64, s: 'L' }, { h: 163, w: 64, s: 'L' },
  { h: 165, w: 61, s: 'L' }, { h: 165, w: 62, s: 'L' }, { h: 165, w: 65, s: 'L' },
  { h: 168, w: 62, s: 'L' }, { h: 168, w: 63, s: 'L' }, { h: 168, w: 66, s: 'L' },
  { h: 170, w: 63, s: 'L' }, { h: 170, w: 64, s: 'L' }, { h: 170, w: 68, s: 'L' },
]
