import { describe, expect, it } from 'vitest'
import { answerMatches, fmt } from './format'

describe('answerMatches', () => {
  it('accepts fractions, rounding and unicode minus', () => {
    expect(answerMatches('5/68', '0.0735')).toBe(true)
    expect(answerMatches('0.48', '0.4839')).toBe(true)
    expect(answerMatches('−3', '-3')).toBe(true)
    expect(answerMatches('C1', 'c1')).toBe(true)
  })

  it('ignores spacing in text answers', () => {
    expect(answerMatches('torch.size([2,2])', 'torch.Size([2, 2])')).toBe(true)
    expect(answerMatches('torch.Size([2, 3])', 'torch.Size([2, 2])')).toBe(false)
  })

  it('grades a tail density that 4 decimals would round to 0', () => {
    expect(answerMatches('0.000018', '0.000018')).toBe(true)
    expect(answerMatches('0', '0.000018')).toBe(false)
  })

  it('grades tiny values relatively', () => {
    expect(answerMatches('0', '5.3e-7')).toBe(false)
    expect(answerMatches('5.3e-7', '5.342e-7')).toBe(true)
    expect(answerMatches('0.006', '0.0061')).toBe(true)
  })

  it('rejects wrong answers', () => {
    expect(answerMatches('0.6', '0.4')).toBe(false)
    expect(answerMatches('', '1')).toBe(false)
  })
})

describe('fmt', () => {
  it('drops trailing zeros and uses a real minus sign', () => {
    expect(fmt(0.5)).toBe('0.5')
    expect(fmt(-1.25, 1)).toBe('−1.3')
    expect(fmt(-0.0001)).toBe('0')
  })
})
