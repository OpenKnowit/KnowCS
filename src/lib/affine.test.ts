import { describe, expect, it } from 'vitest'
import { applyPoint, doableBy3x3, invert, properties, reach, reflectX, reflectY, rotate, scale, shear, translate, warpAffine } from './affine'

// 5 rows × 6 columns, values 1…30 — non-square so a rows/cols mix-up shows.
const img = Array.from({ length: 5 }, (_, r) => Array.from({ length: 6 }, (_, c) => r * 6 + c + 1))
const warp = (M: Parameters<typeof warpAffine>[1]) => warpAffine(img, M).out

// Expected outputs: cv2.warpAffine(img, M, (6, 5), flags=cv2.INTER_NEAREST), OpenCV 4.13.
describe('warpAffine matches OpenCV', () => {
  it('translates by (2, 1) and fills with 0', () => {
    expect(warp(translate(2, 1))).toEqual([[0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 4], [0, 0, 7, 8, 9, 10], [0, 0, 13, 14, 15, 16], [0, 0, 19, 20, 21, 22]])
  })
  it('reflects along the x-axis (upside down) and the y-axis (mirror)', () => {
    expect(warp(reflectX(5))).toEqual([[25, 26, 27, 28, 29, 30], [19, 20, 21, 22, 23, 24], [13, 14, 15, 16, 17, 18], [7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5, 6]])
    expect(warp(reflectY(6))).toEqual([[6, 5, 4, 3, 2, 1], [12, 11, 10, 9, 8, 7], [18, 17, 16, 15, 14, 13], [24, 23, 22, 21, 20, 19], [30, 29, 28, 27, 26, 25]])
  })
  it('rotates about (cols // 2, rows // 2) with the lecture matrix', () => {
    expect(warp(rotate(90, 3, 2))).toEqual([[0, 6, 12, 18, 24, 30], [0, 5, 11, 17, 23, 29], [0, 4, 10, 16, 22, 28], [0, 3, 9, 15, 21, 27], [0, 2, 8, 14, 20, 26]])
    expect(warp(rotate(30, 3, 2))).toEqual([[0, 0, 4, 5, 12, 0], [2, 3, 10, 11, 17, 18], [7, 8, 15, 16, 23, 24], [7, 14, 15, 22, 22, 29], [0, 19, 20, 27, 28, 0]])
  })
  it('scales and shears', () => {
    expect(warp(scale(2, 2, 3, 3))).toEqual([[15, 15, 16, 16, 17, 17], [15, 15, 16, 16, 17, 17], [21, 21, 22, 22, 23, 23], [21, 21, 22, 22, 23, 23], [27, 27, 28, 28, 29, 29]])
    expect(warp(shear(0.5))).toEqual([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [0, 13, 14, 15, 16, 17], [0, 19, 20, 21, 22, 23], [0, 0, 25, 26, 27, 28]])
  })
})

describe('matrices', () => {
  it('rotation equals cv2.getRotationMatrix2D((3, 2), 30, 1.0)', () => {
    const M = rotate(30, 3, 2)
    const cv = [[0.866025, 0.5, -0.598076], [-0.5, 0.866025, 1.767949]]
    M.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(cv[i][j], 5)))
  })
  it('maps points forward and back', () => {
    expect(applyPoint(reflectY(6), 0, 3)).toEqual([5, 3])
    const M = rotate(30, 3, 2)
    const back = applyPoint(invert(M)!, ...applyPoint(M, 1, 4))
    expect(back[0]).toBeCloseTo(1)
    expect(back[1]).toBeCloseTo(4)
    expect(invert([[1, 2, 0], [2, 4, 0]])).toBeNull()
  })
})

describe('what an affine map keeps (Final 2024 Q1(f): it *may* keep distances and angles)', () => {
  it('rigid maps keep distances and angles', () => {
    for (const M of [translate(3, -1), rotate(37, 2, 2), reflectX(5)]) expect(properties(M)).toMatchObject({ distances: true, angles: true })
  })
  it('uniform scaling keeps only angles; shear and stretch keep neither', () => {
    expect(properties(scale(2, 2, 0, 0))).toMatchObject({ distances: false, angles: true })
    expect(properties(scale(2, 1, 0, 0))).toMatchObject({ distances: false, angles: false })
    expect(properties(shear(0.5))).toMatchObject({ distances: false, angles: false })
  })
  it('reflections are mirrored, rotations are not', () => {
    expect(properties(reflectY(6)).mirrored).toBe(true)
    expect(properties(rotate(180, 3, 2)).mirrored).toBe(false)
  })
})

describe('local or global (Final 2022 Q1(f), Final 2024 Q5(e))', () => {
  it('a flip of a 64-wide image needs a pixel 63 columns away', () => {
    const wide = [Array.from({ length: 64 }, (_, i) => i)]
    expect(reach(warpAffine(wide, reflectY(64)).from)).toBe(63)
  })
  it('only a one-pixel shift fits in a 3 × 3 kernel', () => {
    expect(doableBy3x3(translate(1, -1))).toBe(true)
    expect(doableBy3x3(translate(2, 0))).toBe(false)
    expect(doableBy3x3(reflectY(3))).toBe(false)
    expect(doableBy3x3(rotate(90, 1, 1))).toBe(false)
  })
})
