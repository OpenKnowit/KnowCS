/**
 * D-fold cross-validation on an ordered dataset (lecture 3; the "unshuffled folds" trap of
 * 2023 Fall Q7, 2024 Spring Q7 and Final 2024 Q3b).
 *
 * Exam assumption: a test sample is classified correctly iff its class appears in the training folds.
 */

/** Deterministic PRNG so a "shuffle" looks the same on every render. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Labels for `perClass` samples of each of `classCount` classes, sorted by class (x0 … x19 = class 0, …). */
export const sortedLabels = (classCount: number, perClass: number): number[] =>
  Array.from({ length: classCount * perClass }, (_, i) => Math.floor(i / perClass))

export type Ordering = 'sorted' | 'shuffled' | 'stratified'

/** Order in which samples are laid out before being cut into contiguous folds. */
export function arrange(labels: number[], ordering: Ordering, d: number, seed = 7): number[] {
  const idx = labels.map((_, i) => i)
  if (ordering === 'sorted') return idx
  if (ordering === 'shuffled') {
    const rnd = mulberry32(seed)
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
    }
    return idx
  }
  // stratified: deal each class round-robin into folds, then concatenate the folds
  const folds: number[][] = Array.from({ length: d }, () => [])
  let k = 0
  const classes = [...new Set(labels)]
  for (const c of classes) for (const i of idx.filter((j) => labels[j] === c)) folds[k++ % d].push(i)
  return folds.flat()
}

/** Fold number for each position, like np.array_split: the first n % d folds get one extra sample. */
export function foldSizes(n: number, d: number): number[] {
  return Array.from({ length: d }, (_, f) => Math.floor(n / d) + (f < n % d ? 1 : 0))
}

export interface FoldResult {
  fold: number
  /** sample indices (into labels) in this fold, in layout order */
  members: number[]
  trainClasses: number[]
  testClasses: number[]
  correct: number
  accuracy: number
}

export function crossValidate(labels: number[], d: number, ordering: Ordering, seed = 7): { folds: FoldResult[]; mean: number; order: number[] } {
  const order = arrange(labels, ordering, d, seed)
  const sizes = foldSizes(labels.length, d)
  const groups: number[][] = []
  let at = 0
  for (const s of sizes) {
    groups.push(order.slice(at, at + s))
    at += s
  }
  const folds = groups.map((members, f) => {
    const train = new Set(groups.flatMap((g, k) => (k === f ? [] : g.map((i) => labels[i]))))
    const correct = members.filter((i) => train.has(labels[i])).length
    return {
      fold: f,
      members,
      trainClasses: [...train].sort((a, b) => a - b),
      testClasses: [...new Set(members.map((i) => labels[i]))].sort((a, b) => a - b),
      correct,
      accuracy: members.length ? correct / members.length : NaN,
    }
  })
  return { folds, mean: folds.reduce((a, f) => a + f.accuracy, 0) / folds.length, order }
}
