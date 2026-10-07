import { LabError } from './labError'

/** Gaussian Naive Bayes for numerical features (lecture 2 "Calculation for Continuous Variables"). */

export const gaussianPdf = (x: number, mu: number, sigma: number): number =>
  Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI))

/** Mean and standard deviation. ddof = 1 is the sample σ (÷ n − 1) used in COMP2211 exams. */
export function meanStd(values: number[], ddof: 0 | 1 = 1): { mu: number; sigma: number } {
  const n = values.length
  const mu = values.reduce((a, b) => a + b, 0) / n
  const ss = values.reduce((a, v) => a + (v - mu) ** 2, 0)
  return { mu, sigma: n - ddof > 0 ? Math.sqrt(ss / (n - ddof)) : NaN }
}

export interface GaussianClass {
  name: string
  prior: number
  /** Shown in tables, e.g. "3/6". */
  priorLabel: string
  /** One entry per feature. values is null when μ and σ were given directly. */
  stats: { mu: number; sigma: number; values: number[] | null }[]
}

export interface LabeledRow {
  label: string
  x: number[]
}

export function fitGaussianNb(rows: LabeledRow[], featureCount: number, ddof: 0 | 1 = 1): GaussianClass[] {
  const names = [...new Set(rows.map((r) => r.label))]
  return names.map((name) => {
    const mine = rows.filter((r) => r.label === name)
    return {
      name,
      prior: mine.length / rows.length,
      priorLabel: `${mine.length}/${rows.length}`,
      stats: Array.from({ length: featureCount }, (_, j) => {
        const values = mine.map((r) => r.x[j])
        return { ...meanStd(values, ddof), values }
      }),
    }
  })
}

export interface ScoredClass extends GaussianClass {
  likelihoods: number[]
  /** prior × Π likelihoods — the numerator of Bayes' rule. */
  score: number
  posterior: number
}

export function scoreGaussianNb(classes: GaussianClass[], x: number[]): ScoredClass[] {
  const scored = classes.map((c) => {
    const likelihoods = c.stats.map((s, j) => gaussianPdf(x[j], s.mu, s.sigma))
    return { ...c, likelihoods, score: likelihoods.reduce((a, b) => a * b, c.prior), posterior: 0 }
  })
  const total = scored.reduce((a, c) => a + c.score, 0)
  scored.forEach((c) => (c.posterior = total > 0 ? c.score / total : NaN))
  return scored
}

/** Parse "label, f1, f2" CSV with a header row. Throws a readable message on bad input. */
export function parseLabeledCsv(csv: string): { features: string[]; rows: LabeledRow[] } {
  const lines = csv.split('\n').map((l) => l.trim()).filter(Boolean)
  if (lines.length < 3) throw new LabError('csv_header', {}, 'Need a header and at least two rows.')
  const header = lines[0].split(',').map((s) => s.trim())
  const features = header.slice(1)
  if (!features.length) throw new LabError('csv_feature', {}, 'Need at least one numeric feature column.')
  const rows = lines.slice(1).map((line, i) => {
    const cells = line.split(',').map((s) => s.trim())
    if (cells.length !== header.length) throw new LabError('csv_row_len', { row: i + 1, got: cells.length, want: header.length }, `Row ${i + 1} has ${cells.length} values, expected ${header.length}.`)
    const x = cells.slice(1).map(Number)
    if (x.some((v) => !Number.isFinite(v))) throw new LabError('csv_nan', { row: i + 1 }, `Row ${i + 1} has a value that is not a number.`)
    return { label: cells[0], x }
  })
  return { features, rows }
}
