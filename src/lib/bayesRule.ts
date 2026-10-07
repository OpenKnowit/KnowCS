/** Bayes' rule with one binary belief B and one binary piece of evidence E (lecture 2 virus test). */
export interface BayesInput {
  prior: number // P(B)
  hit: number // P(E | B)
  falseAlarm: number // P(E | not B)
  observed: 'E' | 'notE'
}

export interface BayesResult {
  pB: number
  pNotB: number
  /** Likelihood of what was observed, given B / not B. */
  likeB: number
  likeNotB: number
  /** P(observed), by the total-probability rule. */
  evidence: number
  posterior: number // P(B | observed)
  likelihoodRatio: number
}

export function bayesPosterior({ prior, hit, falseAlarm, observed }: BayesInput): BayesResult {
  const pB = prior
  const pNotB = 1 - prior
  const likeB = observed === 'E' ? hit : 1 - hit
  const likeNotB = observed === 'E' ? falseAlarm : 1 - falseAlarm
  const evidence = likeB * pB + likeNotB * pNotB
  return {
    pB,
    pNotB,
    likeB,
    likeNotB,
    evidence,
    posterior: evidence > 0 ? (likeB * pB) / evidence : 0,
    likelihoodRatio: likeNotB > 0 ? likeB / likeNotB : Infinity,
  }
}

export interface NaturalCounts {
  tp: number // B and E
  fn: number // B and not E
  fp: number // not B and E
  tn: number // not B and not E
}

/** Whole-person counts out of n that add up to exactly n (largest-remainder rounding). */
export function naturalCounts(prior: number, hit: number, falseAlarm: number, n = 10000): NaturalCounts {
  const raw = [prior * hit, prior * (1 - hit), (1 - prior) * falseAlarm, (1 - prior) * (1 - falseAlarm)].map((p) => p * n)
  const out = raw.map(Math.floor)
  let left = n - out.reduce((a, b) => a + b, 0)
  raw
    .map((r, i) => [r - out[i], i] as const)
    .sort((a, b) => b[0] - a[0])
    .forEach(([, i]) => {
      if (left > 0) {
        out[i]++
        left--
      }
    })
  const [tp, fn, fp, tn] = out
  return { tp, fn, fp, tn }
}
