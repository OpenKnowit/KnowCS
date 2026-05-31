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

// App 导航
export type TabId = 'numpy' | 'backprop' | 'kernel' | 'bayesBasics' | 'naiveBayes' | 'knn'
