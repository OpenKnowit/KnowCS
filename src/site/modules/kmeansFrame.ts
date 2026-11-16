// The k-means plot's pixel frame: the read-only plot (KMeansTable) and the drag practice (KMeansDrag) draw the same picture.

export function plotFrame(pts: number[][], centroids: number[][], next: number[][]) {
  const oneD = pts[0].length === 1
  const W = 560, H = oneD ? 160 : 360, P = 34
  const all = [...pts, ...centroids, ...next]
  const pad = (a: number[]) => {
    const lo = Math.min(...a), hi = Math.max(...a)
    const m = (hi - lo || 2) * 0.12
    return [lo - m, hi + m]
  }
  const [x0, x1] = pad(all.map((p) => p[0]))
  const [y0, y1] = oneD ? [-1, 1] : pad(all.map((p) => p[1]))
  const X = (v: number) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P)
  const Y = (v: number) => (oneD ? H / 2 : H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P))
  const at = (p: number[]): [number, number] => [X(p[0]), Y(oneD ? 0 : p[1])]
  return { oneD, W, H, P, x0, x1, y0, y1, X, Y, at }
}
