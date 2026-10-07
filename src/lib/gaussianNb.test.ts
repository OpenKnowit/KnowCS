import { describe, expect, it } from 'vitest'
import { fitGaussianNb, gaussianPdf, meanStd, parseLabeledCsv, scoreGaussianNb } from './gaussianNb'

describe('gaussianPdf', () => {
  it('matches the lecture 2 continuous example', () => {
    expect(gaussianPdf(66, 73, 6.2)).toBeCloseTo(0.0340187, 6)
    expect(gaussianPdf(97, 79, 10.2)).toBeCloseTo(0.00824276, 7)
    expect(gaussianPdf(66, 75, 7.9)).toBeCloseTo(0.0263909, 6)
    expect(gaussianPdf(97, 86, 9.7)).toBeCloseTo(0.0216215, 6)
  })
})

describe('meanStd', () => {
  it('uses n − 1 by default', () => {
    expect(meanStd([36, 36.5, 37]).sigma).toBeCloseTo(0.5, 10)
    expect(meanStd([36, 36.5, 37], 0).sigma).toBeCloseTo(Math.sqrt(1 / 6), 10)
  })

  it('matches the 2024 Spring midterm first-class honours σ (0.26904)', () => {
    const r = meanStd([8.5, 8.1, 7.9, 8.2, 8.0, 7.7, 7.8])
    expect(r.mu).toBeCloseTo(8.02857, 4)
    expect(r.sigma).toBeCloseTo(0.26904, 4)
  })
})

describe('fit + score', () => {
  it('reproduces the Final 2024 dengue answer (0.4839, 0.0252, posterior 0.9994)', () => {
    const { rows } = parseLabeledCsv('d,t,p\nD,40,60\nD,39,50\nD,38,70\nN,36,90\nN,36.5,75\nN,37,105')
    const classes = fitGaussianNb(rows, 2)
    const [, no] = scoreGaussianNb(classes, [36, 85])
    expect(no.likelihoods[0]).toBeCloseTo(0.4839, 4)
    expect(no.likelihoods[1]).toBeCloseTo(0.0252, 4)
    expect(no.score).toBeCloseTo(0.0061, 4)
    expect(no.score / (no.score + 0.0000036)).toBeCloseTo(0.9994, 4)
  })

  it('reports bad CSV rows', () => {
    expect(() => parseLabeledCsv('a,b\nx,1\ny')).toThrow(/Row 2/)
    expect(() => parseLabeledCsv('a,b\nx,1\ny,z')).toThrow(/not a number/)
  })
})
