import { describe, expect, it } from 'vitest'
import { clipLine, halfPlane } from './geom2d'

describe('geom2d', () => {
  it('clips a diagonal line to the window', () => {
    expect(clipLine(1, 1, -1, 0, 1)).toEqual([[0, 1], [1, 0]])
    expect(clipLine(1, 0, -5, 0, 1)).toBeNull()
  })
  it('returns the half-plane polygon', () => {
    // x > 0.5 in [0,1]²: a rectangle of four corners
    const poly = halfPlane(1, 0, -0.5, 0, 1)
    expect(poly).toHaveLength(4)
    expect(poly).toContainEqual([1, 0])
    expect(poly).toContainEqual([0.5, 1])
    expect(halfPlane(1, 1, -5, 0, 1)).toEqual([])
    expect(halfPlane(1, 1, 5, 0, 1)).toHaveLength(4)
  })
})
