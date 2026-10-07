import { describe, expect, it } from 'vitest'
import { LECTURE_NET, XOR, backprop, forward, trainEpochs } from './mlp'

describe('lecture 6 XOR example (η = 0.5)', () => {
  it('round 1 step 1 matches the slides', () => {
    const s = backprop(LECTURE_NET, [0, 0], 0, 0.5)
    expect(s.fwd.o).toBeCloseTo(0.435364, 6)
    expect(s.deltaK).toBeCloseTo(0.107022, 6)
    expect(s.deltaJ[0]).toBeCloseTo(0.02301, 5)
    expect(s.deltaJ[1]).toBeCloseTo(-0.036923, 6)
    // the slides round every intermediate value to 6 decimals
    expect(s.after.w[4]).toBeCloseTo(0.833245, 5)
    expect(s.after.w[5]).toBeCloseTo(-1.406756, 5)
    expect(s.after.theta).toEqual([expect.closeTo(-0.011505, 5), expect.closeTo(0.018462, 5), expect.closeTo(-0.053511, 5)])
  })

  it('round 1 step 2 forward pass matches the slides', () => {
    const net = backprop(LECTURE_NET, [0, 0], 0, 0.5).after
    const f = forward(net, [0, 1])
    expect(f.h[0]).toBeCloseTo(0.652148, 6)
    expect(f.h[1]).toBeCloseTo(0.702339, 6)
    expect(f.o).toBeCloseTo(0.37798, 5)
  })

  it('after 10,000 rounds the outputs match the slides', () => {
    const net = trainEpochs(LECTURE_NET, 10000, 0.5)
    const outs = XOR.map((s) => forward(net, s.x).o)
    expect(outs[0]).toBeCloseTo(0.016973, 3)
    expect(outs[1]).toBeCloseTo(0.984139, 3)
    expect(outs[2]).toBeCloseTo(0.980513, 3)
    expect(outs[3]).toBeCloseTo(0.015179, 3)
  })
})

describe('step activation', () => {
  it('solves XOR with the hand-picked 2025 Spring weights', () => {
    const net = { w: [1, 1, -1, -1, 1, 1] as [number, number, number, number, number, number], theta: [-0.5, 1.5, -1] as [number, number, number] }
    expect(XOR.map((s) => forward(net, s.x, 'step').o)).toEqual([0, 1, 1, 0])
  })
})
