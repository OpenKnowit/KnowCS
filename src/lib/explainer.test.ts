import { describe, expect, it } from 'vitest'
import { cueAt, ease, fmtClock, lerpColor, locate, seg, stopBetween, timeline, window01 } from './explainer'

const SCENES = [
  { dur: 1000, cues: [0, 0.5] },
  { dur: 2000, cues: [0], ponder: true },
  { dur: 500, cues: [0, 0.2, 0.8] },
]

describe('explainer timeline', () => {
  it('lays scenes end to end and records ponder stops', () => {
    const tl = timeline(SCENES)
    expect(tl.starts).toEqual([0, 1000, 3000])
    expect(tl.total).toBe(3500)
    expect(tl.stops).toEqual([3000])
  })

  it('locates the scene and its local progress', () => {
    const tl = timeline(SCENES)
    expect(locate(tl, SCENES, 0)).toEqual({ i: 0, p: 0 })
    expect(locate(tl, SCENES, 500)).toEqual({ i: 0, p: 0.5 })
    expect(locate(tl, SCENES, 1000)).toEqual({ i: 1, p: 0 })
    expect(locate(tl, SCENES, 3250)).toEqual({ i: 2, p: 0.5 })
    expect(locate(tl, SCENES, 3500)).toEqual({ i: 2, p: 1 })
    expect(locate(tl, SCENES, 9999)).toEqual({ i: 2, p: 1 })
    expect(locate(tl, SCENES, -5)).toEqual({ i: 0, p: 0 })
  })

  it('finds the caption and the stop crossed by a frame step', () => {
    expect(cueAt([0, 0.2, 0.8], 0.1)).toBe(0)
    expect(cueAt([0, 0.2, 0.8], 0.2)).toBe(1)
    expect(cueAt([0, 0.2, 0.8], 1)).toBe(2)
    expect(stopBetween([3000], 2990, 3010)).toBe(3000)
    expect(stopBetween([3000], 3000, 3010)).toBeNull()
  })
})

describe('explainer motion helpers', () => {
  it('clamps windows and eases symmetrically', () => {
    expect(seg(0.5, 0.2, 0.6)).toBeCloseTo(0.75)
    expect(seg(0.1, 0.2, 0.6)).toBe(0)
    expect(seg(0.7, 0.5, 0.5)).toBe(1)
    expect(ease(0)).toBe(0)
    expect(ease(1)).toBe(1)
    expect(ease(0.5)).toBeCloseTo(0.5)
    expect(ease(0.25) + ease(0.75)).toBeCloseTo(1)
    expect(window01(0.5, 0.2, 0.8)).toBe(1)
    expect(window01(0.9, 0.2, 0.8)).toBe(0)
  })

  it('blends colours and formats the clock', () => {
    expect(lerpColor('#000000', '#ffffff', 0.5)).toBe('#808080')
    expect(lerpColor('#f00', '#00f', 1)).toBe('#0000ff')
    expect(fmtClock(65_400)).toBe('1:05')
  })
})
