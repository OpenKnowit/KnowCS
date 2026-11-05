import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { MATPLOTLIB, boxStats, cmapColor, contourLevels, histogram, maxNLocator } from './pyPlot'
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
    expect(r.stdout).toBe('(2, 3) Axes(0.125,0.53;0.227941x0.35)\n(3,)\n6\n')
  })

  it('records a figure after each plotting call, focused on the new artist', () => {
    const r = run("x = np.arange(5)\nplt.plot(x, x ** 2, 'ro--', label='x²')\nplt.title('square')\nplt.legend()")
    const ev = r.events.filter((e): e is PyEvent & { type: 'figure' } => e.type === 'figure')
    expect(ev.map((e) => e.line)).toEqual([4, 5, 6])
    const line = ev[0].fig.axes[0].artists[0]
    expect(line).toMatchObject({ kind: 'line', color: '#ff0000', marker: 'o', dash: [3.7, 1.6], label: 'x²', y: [0, 1, 4, 9, 16] })
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
    expect(figs(r)[1].axes[0].artists[0]).toMatchObject({ kind: 'bar', heights: [3, 5] })
    expect(figs(r)[1].axes[0].xticks).toMatchObject({ at: [0, 1], labels: ['a', 'b'] })
  })

  it('bad format strings and shapes give matplotlib errors', () => {
    expect(run("plt.plot([1, 2], 'q')").error?.message).toMatch("'q' is not a valid format string")
    expect(run('plt.plot([1, 2, 3], [1, 2])').error?.message).toBe('x and y must have same first dimension, but have shapes (3,) and (2,)')
    expect(run('plt.imshow(np.arange(4))').error?.message).toBe('Invalid shape (4,) for image data')
  })
})

// Checked against matplotlib 3.10.8.
describe('contourf, boxplot, pie, errorbar', () => {
  it('contour levels follow MaxNLocator(levels + 1) with matplotlib\'s default steps', () => {
    expect(contourLevels([0, 1], null).map((v) => +v.toFixed(10))).toEqual([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05])
    expect(contourLevels([0, 6], null).map((v) => +v.toFixed(10))).toEqual([0, 0.8, 1.6, 2.4, 3.2, 4, 4.8, 5.6, 6.4])
    expect(contourLevels([-10, 10], null)).toEqual([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10])
    expect(contourLevels([0, 6], 3)).toEqual([0, 1.5, 3, 4.5, 6])
    expect(+maxNLocator(0, 1, 8)[1].toFixed(10)).toBe(0.15)
  })
  it('contourf fills bands of the level range, each coloured by its midpoint, without margins', () => {
    const r = run('xx, yy = np.meshgrid(np.linspace(0, 4, 5), np.linspace(0, 2, 3))\nplt.contourf(xx, yy, (xx > 2).astype(int))')
    expect(r.error).toBeNull()
    const ax = figs(r)[0].axes[0]
    expect(ax.xlim).toEqual([0, 4])
    expect(ax.ylim).toEqual([0, 2])
    const bands = ax.artists[0]
    expect(bands.kind).toBe('poly')
    if (bands.kind !== 'poly') return
    // first band (0–0.15) is viridis(0.075 / 1.05)
    expect(bands.fcs[0]).toBe(cmapColor('viridis', 0.075 / 1.05))
  })
  it('boxplot statistics: quartiles, whiskers within 1.5 IQR, fliers', () => {
    expect(boxStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 30], 1)).toEqual({ pos: 1, q1: 3.25, med: 5.5, q3: 7.75, lo: 1, hi: 9, fliers: [30] })
    expect(boxStats([2, 3, 3, 4], 2)).toMatchObject({ q1: 2.75, med: 3, q3: 3.25, lo: 2, hi: 4, fliers: [] })
    const r = run('plt.boxplot([[1, 2, 3], [2, 3, 4]])')
    const box = figs(r)[0].axes[0].artists[0]
    expect(box.kind === 'line' && box.x).toEqual([0.925, 1.075, 1.075, 0.925, 0.925])
  })
  it('pie: wedges with their texts and fixed limits', () => {
    const r = run("plt.pie([1, 2, 3], labels=['a', 'b', 'c'], autopct='%1.1f%%')")
    expect(r.error).toBeNull()
    const ax = figs(r)[0].axes[0]
    expect(ax.artists.flatMap((a) => (a.kind === 'text' ? [a.text] : []))).toEqual(['a', '16.7%', 'b', '33.3%', 'c', '50.0%'])
    expect(ax.artists.filter((a) => a.kind === 'poly').length).toBe(3)
    expect(ax.xlim).toEqual([-1.25, 1.25])
  })
  it('errorbar takes a scalar, one value per point, or [below, above]', () => {
    const r = run('plt.errorbar([1, 2], [3, 4], yerr=[0.5, 1], capsize=3)')
    expect(r.error).toBeNull()
    const e = figs(r)[0].axes[0].artists.find((a) => a.kind === 'segs')
    expect(e && e.kind === 'segs' && e.paths).toEqual([[1, 2.5, 1, 3.5], [2, 3, 2, 5]])
  })
})

describe('the figure description', () => {
  it('a call that adds several artists focuses on all of them', () => {
    const r = run("plt.pie([1, 2], labels=['a', 'b'])")
    const ev = r.events.filter((e): e is PyEvent & { type: 'figure' } => e.type === 'figure')
    expect(ev[ev.length - 1].focus).toEqual({ ax: 0, artist: 0, last: 3 })
  })
  it('drawing a step does not settle the autoscale a later patch still counts for', () => {
    const r = run("from matplotlib.patches import Circle\nplt.scatter([0, 1], [0, 1])\nplt.gca().add_patch(Circle((5, 5), 1))\nprint([float(v) for v in plt.xlim()])")
    expect(r.stdout).toBe('[-0.30000000000000004, 6.3]\n')
  })
  it('uint8 images are integers in 0..255, in colour and as data', () => {
    const r = run("img = np.array([[[255, 0, 0], [0, 128, 0]]], dtype=np.uint8)\nim = plt.imshow(img)\nprint(im.get_array().dtype)\nplt.figure()\nb = plt.imshow(np.array([[0, 255]], dtype=np.uint8), cmap='gray')\nprint(b.get_clim())")
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('uint8\n(0.0, 255.0)\n')
    const [a, g] = figs(r).map((f) => f.axes[0].artists[0])
    expect(a.kind === 'image' && a.pixels).toEqual(['#ff0000', '#008000'])
    expect(g.kind === 'image' && g.pixels).toEqual(['#000000', '#ffffff'])
  })
  it('there are no image files to read, and saving is a no-op with a note', () => {
    const e = run("import matplotlib.image as mpimg\nmpimg.imread('cat.png')").error
    expect(e?.type).toBe('FileNotFoundError')
    expect(e?.message).toMatch("No such file or directory: 'cat.png'")
    expect(run("plt.imsave('a.png', np.zeros((2, 2)))").stdout).toMatch('a.png')
    expect(run("plt.plot([1])\nplt.savefig('b.png')").stdout).toMatch('b.png')
  })
  it('a parameter that would change the result and is not drawn says so', () => {
    for (const code of ["plt.subplot(projection='polar')", "plt.pcolormesh([[1, 2]], shading='gouraud')", 'plt.pie([1], shadow=True)', "plt.gca().set_xscale('symlog')", 'plt.hexbin([1], [1], C=[1])']) {
      expect(run(code).error?.type).toBe('NotImplementedError')
    }
  })
  it('3D axes project onto the 2D plane and keep their depth order', () => {
    const r = run("ax = plt.figure().add_subplot(projection='3d')\nax.scatter([0, 1], [0, 1], [0, 1], s=9)\nax.set_zlabel('z')")
    expect(r.error).toBeNull()
    const ax = figs(r)[0].axes[0]
    const s = ax.artists[0]
    expect(s.kind).toBe('scatter')
    expect(ax.artists.some((a) => a.kind === 'text' && a.text === 'z')).toBe(true)
    if (s.kind === 'scatter') expect(s.x.every((v) => Math.abs(v) < 0.1)).toBe(true)
  })
})
