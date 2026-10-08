import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { MATPLOTLIB, cmapColor, histogram } from './pyPlot'
import type { PyEvent } from './pyEvents'

const run = (code: string) => runPython(`import numpy as np\nimport matplotlib.pyplot as plt\n${code}`, { libs: [MATPLOTLIB], maxSize: 65536 })
const figs = (r: ReturnType<typeof run>) => r.displays.flatMap((d) => (d.type === 'figure' ? [d.fig] : []))

describe('histogram (matches numpy.histogram)', () => {
  it('equal-width bins, last bin closed', () => {
    expect(histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 4)).toEqual({ counts: [1, 2, 3, 4], edges: [1, 1.75, 2.5, 3.25, 4] })
    expect(histogram([0, 0.5, 1], 2).counts).toEqual([1, 2])
    expect(histogram([0, 64, 128, 192, 255], 4, [0, 256]).counts).toEqual([1, 1, 1, 2])
  })

  it('plt.hist returns (n, bins, patches) like matplotlib', () => {
    const r = run('n, bins, patches = plt.hist([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], bins=4)\nprint(repr(n), repr(bins), patches)')
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('array([1., 2., 3., 4.]) array([1.  , 1.75, 2.5 , 3.25, 4.  ]) <BarContainer object of 4 artists>\n')
  })
})

describe('colormaps', () => {
  it('gray and viridis at the midpoint', () => {
    expect(cmapColor('gray', 0.5)).toBe('#808080')
    expect(cmapColor('viridis', 0.5)).toBe('#21918c')
    expect(cmapColor('gray_r', 0)).toBe('#ffffff')
  })
})

describe('pyplot', () => {
  it('subplots shapes and indexing follow matplotlib', () => {
    const r = run('fig, axes = plt.subplots(2, 3)\nprint(axes.shape, axes[0, 0])\nfig2, row = plt.subplots(1, 3)\nprint(row.shape)\nfor ax in axes.flat:\n    ax.plot([1, 2])\nprint(len(fig.axes))')
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('(2, 3) <Axes: >\n(3,)\n6\n')
  })

  it('records a figure after each plotting call, focused on the new artist', () => {
    const r = run("x = np.arange(5)\nplt.plot(x, x ** 2, 'ro--', label='x²')\nplt.title('square')\nplt.legend()")
    const ev = r.events.filter((e): e is PyEvent & { type: 'figure' } => e.type === 'figure')
    expect(ev.map((e) => e.line)).toEqual([4, 5, 6])
    const line = ev[0].fig.axes[0].artists[0]
    expect(line).toMatchObject({ kind: 'line', color: '#ff0000', marker: 'o', dash: '6 4', label: 'x²', y: [0, 1, 4, 9, 16] })
    expect(ev[0].focus).toEqual({ ax: 0, artist: 0 })
    expect(figs(r)[0].axes[0].title).toBe('square')
  })

  it('imshow: viridis and auto range unless cmap / vmin / vmax are given', () => {
    const r = run('img = np.array([[0.2, 0.4], [0.6, 0.8]])\nplt.imshow(img)\nplt.figure()\nplt.imshow(img, cmap="gray", vmin=0, vmax=1)')
    expect(r.error).toBeNull()
    const [a, b] = figs(r).map((f) => f.axes[0].artists[0])
    expect(a).toMatchObject({ kind: 'image', cmap: 'viridis', auto: true, vmin: 0.2, vmax: 0.8 })
    expect(a.kind === 'image' && a.pixels[0]).toBe('#440154')
    expect(b.kind === 'image' && b.pixels).toEqual(['#333333', '#666666', '#999999', '#cccccc'])
    expect(r.events.filter((e) => e.type === 'note').map((e) => (e as { key: string }).key)).toEqual(['default_cmap', 'auto_range'])
  })

  it('scatter colours points by label through the colormap', () => {
    const r = run("pts = np.array([[1, 1], [2, 1], [8, 9]])\nlabels = np.array([0, 0, 1])\nplt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='viridis')")
    const s = figs(r)[0].axes[0].artists[0]
    expect(s.kind === 'scatter' && s.colors).toEqual(['#440154', '#440154', '#fde725'])
  })

  it('show() ends a figure; the next call starts a new one', () => {
    const r = run('plt.plot([1, 2])\nplt.show()\nplt.bar(["a", "b"], [3, 5])\nprint(plt.figure())')
    expect(r.stdout).toBe('Figure(640x480)\n')
    expect(figs(r).length).toBe(2)
    expect(figs(r)[1].axes[0].artists[0]).toMatchObject({ kind: 'bar', tickLabels: ['a', 'b'], heights: [3, 5] })
  })

  it('bad format strings and shapes give matplotlib errors', () => {
    expect(run("plt.plot([1, 2], 'q')").error?.message).toMatch("'q' is not a valid format string")
    expect(run('plt.plot([1, 2, 3], [1, 2])').error?.message).toBe('x and y must have same first dimension, but have shapes (3,) and (2,)')
    expect(run('plt.imshow(np.arange(4))').error?.message).toBe('Invalid shape (4,) for image data')
  })
})
