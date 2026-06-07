// --- 共享类型定义 ---

// KNN 模块
export type StudentClass = 'M' | 'L'

export interface KnnPoint {
  h: number
  w: number
  s: StudentClass
}

export interface KnnStats {
  meanH: number
  meanW: number
  stdH: number
  stdW: number
  minH: number
  maxH: number
  minW: number
  maxW: number
}

export interface TestPoint {
  h: number
  w: number
}

// 贝叶斯模块
export type BayesClass = 'yes' | 'no'
export type BayesFeature = 'BP' | 'Fever' | 'Diabetes' | 'Vomit'

export interface BayesData {
  priors: Record<BayesClass, number>
  counts: Record<BayesFeature, Record<string, Record<BayesClass, number>>>
  m_values: Record<BayesFeature, number>
}

export interface NaiveBayesStep {
  name: string
  val: number
  label: string
  formula?: string
}

export interface NaiveBayesClassResult {
  score: number
  raw: number
  steps: NaiveBayesStep[]
  prob: number
}

export type NaiveBayesResults = Record<BayesClass, NaiveBayesClassResult>

// 卷积模块
export type KernelPresetName = 'Sobel-X' | 'Sobel-Y' | 'Laplacian' | 'Identity'
export type ActivePreset = KernelPresetName | 'Custom'

// NumPy 模块
export type SliceType = 'none' | 'slice' | 'fancy' | 'mask'

// Alpha-Beta 剪枝模块（L10）
export type AbNodeType = 'max' | 'min' | 'leaf'

export interface AbNode {
  id: string
  type: AbNodeType
  parent: string | null
  children?: string[]
  value?: number // 仅叶子
  x: number
  y: number
}

export type AbStepType = 'enter' | 'leaf_eval' | 'traverse' | 'update' | 'prune' | 'exit'

export interface AbStep {
  type: AbStepType
  nodeId: string
  alpha: number
  beta: number
  nodeValues: Record<string, number | null>
  nodeAlphas: Record<string, number>
  nodeBetas: Record<string, number>
  visited: string[]
  prunedNodes: string[]
  prunedEdges: string[]
  activeEdge: string | null
  // 结构化解释字段（由组件按 i18n 渲染文案）
  childId?: string
  childResult?: number
  prevBest?: number
  newBest?: number
  prevBound?: number
  newBound?: number
  prunedChildren?: string[]
  leafValue?: number
  returnValue?: number
}

// K-Means 聚类模块（L4）
export interface Vec2 {
  x: number
  y: number
}

export interface KMeansIteration {
  centroids: Vec2[]
  assignments: number[] // 每个数据点所属簇索引
  inertia: number // 簇内平方和（WCSS）
  moved: number // 质心移动总距离
}

// App 导航
export type TabId = 'numpy' | 'backprop' | 'kernel' | 'bayesBasics' | 'naiveBayes' | 'knn' | 'alphabeta' | 'kmeans'
