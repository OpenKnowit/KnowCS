/**
 * Geometry behind contour plots: iso-lines and filled bands on a grid or on triangles, Delaunay triangulation, and the
 * small statistics helpers plotting needs (kernel density). Paths are flat [x0, y0, x1, y1, …] arrays.
 */

export type Path = number[]

interface V { x: number; y: number; z: number }

/** stitch unordered segments (each end keyed by the mesh edge it lies on) into polylines */
const stitch = (segs: { a: string; b: string; pa: [number, number]; pb: [number, number] }[]): Path[] => {
  const ends = new Map<string, number[]>()
  const add = (k: string, i: number) => {
    const l = ends.get(k)
    if (l) l.push(i)
    else ends.set(k, [i])
  }
  segs.forEach((s, i) => {
    add(s.a, i)
    add(s.b, i)
  })
  const used = new Array<boolean>(segs.length).fill(false)
  const out: Path[] = []
  const walk = (start: string, first: number): Path => {
    const pts: Path = []
    let key = start
    let i = first
    for (;;) {
      used[i] = true
      const s = segs[i]
      const forward = s.a === key
      const [p, q] = forward ? [s.pa, s.pb] : [s.pb, s.pa]
      if (!pts.length) pts.push(p[0], p[1])
      pts.push(q[0], q[1])
      key = forward ? s.b : s.a
      const next = (ends.get(key) ?? []).find((j) => !used[j])
      if (next === undefined) break
      i = next
    }
    return pts
  }
  // open lines first (an end touched once), then closed loops
  segs.forEach((s, i) => {
    if (used[i]) return
    if (ends.get(s.a)?.length === 1) out.push(walk(s.a, i))
    else if (ends.get(s.b)?.length === 1) out.push(walk(s.b, i))
  })
  segs.forEach((s, i) => {
    if (!used[i]) out.push(walk(s.a, i))
  })
  return out
}

/** iso-lines at one level on a grid; X, Y, Z are row-major (ny × nx); cells touching NaN are skipped */
export const gridLines = (X: number[], Y: number[], Z: number[], nx: number, ny: number, level: number): Path[] => {
  const segs: { a: string; b: string; pa: [number, number]; pb: [number, number] }[] = []
  const at = (i: number, j: number) => i * nx + j
  const cross = (p: number, q: number): [number, number] => {
    const t = (level - Z[p]) / (Z[q] - Z[p])
    return [X[p] + t * (X[q] - X[p]), Y[p] + t * (Y[q] - Y[p])]
  }
  const key = (p: number, q: number) => (p < q ? `${p}_${q}` : `${q}_${p}`)
  for (let i = 0; i < ny - 1; i++) {
    for (let j = 0; j < nx - 1; j++) {
      const c = [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)]
      const zs = c.map((k) => Z[k])
      if (zs.some((z) => !Number.isFinite(z))) continue
      const code = (zs[0] > level ? 1 : 0) | (zs[1] > level ? 2 : 0) | (zs[2] > level ? 4 : 0) | (zs[3] > level ? 8 : 0)
      if (code === 0 || code === 15) continue
      const edge = (e: number): [string, [number, number]] => {
        const [p, q] = [c[e], c[(e + 1) % 4]]
        return [key(p, q), cross(p, q)]
      }
      const link = (e0: number, e1: number) => {
        const [ka, pa] = edge(e0)
        const [kb, pb] = edge(e1)
        segs.push({ a: ka, b: kb, pa, pb })
      }
      const mid = (zs[0] + zs[1] + zs[2] + zs[3]) / 4 > level
      switch (code) {
        case 1: case 14: link(3, 0); break
        case 2: case 13: link(0, 1); break
        case 3: case 12: link(3, 1); break
        case 4: case 11: link(1, 2); break
        case 6: case 9: link(0, 2); break
        case 7: case 8: link(2, 3); break
        case 5: if (mid) { link(0, 1); link(2, 3) } else { link(3, 0); link(1, 2) } break
        case 10: if (mid) { link(3, 0); link(1, 2) } else { link(0, 1); link(2, 3) } break
      }
    }
  }
  return stitch(segs)
}

/** part of a convex polygon (with a value at each vertex) where lo ≤ z ≤ hi */
const clipBand = (poly: V[], lo: number, hi: number): V[] => {
  const clip = (input: V[], keep: (z: number) => boolean, bound: number): V[] => {
    const out: V[] = []
    input.forEach((a, i) => {
      const b = input[(i + 1) % input.length]
      const ka = keep(a.z)
      const kb = keep(b.z)
      if (ka) out.push(a)
      if (ka !== kb) {
        const t = (bound - a.z) / (b.z - a.z)
        out.push({ x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y), z: bound })
      }
    })
    return out
  }
  const low = clip(poly, (z) => z >= lo, lo)
  if (low.length < 3) return []
  const high = clip(low, (z) => z <= hi, hi)
  return high.length < 3 ? [] : high
}

const flat = (ps: V[]): Path => ps.flatMap((p) => [p.x, p.y])

/**
 * Filled bands on a grid: one list of polygons per band [levels[k], levels[k + 1]]. A cell inside one band stays whole
 * (runs along a row are merged when the grid is rectilinear); a crossed cell is split into four triangles, which also
 * resolves saddles.
 */
export const gridBands = (X: number[], Y: number[], Z: number[], nx: number, ny: number, levels: number[]): Path[][] => {
  const bands: Path[][] = levels.slice(1).map(() => [])
  const nb = bands.length
  const bandOf = (z: number) => {
    if (z < levels[0] || z > levels[nb]) return -1
    let k = 0
    while (k < nb - 1 && z > levels[k + 1]) k++
    return k
  }
  for (let i = 0; i < ny - 1; i++) {
    let run = -1
    let runStart = 0
    const flush = (j: number) => {
      if (run < 0) return
      const a = i * nx + runStart
      const b = i * nx + j
      const c = (i + 1) * nx + j
      const d = (i + 1) * nx + runStart
      bands[run].push([X[a], Y[a], X[b], Y[b], X[c], Y[c], X[d], Y[d]])
      run = -1
    }
    for (let j = 0; j < nx - 1; j++) {
      const idx = [i * nx + j, i * nx + j + 1, (i + 1) * nx + j + 1, (i + 1) * nx + j]
      const zs = idx.map((k) => Z[k])
      if (zs.some((z) => !Number.isFinite(z))) {
        flush(j)
        continue
      }
      const b0 = bandOf(zs[0])
      if (b0 >= 0 && zs.every((z) => bandOf(z) === b0) && Y[idx[0]] === Y[idx[1]] && Y[idx[3]] === Y[idx[2]]) {
        if (run !== b0) {
          flush(j)
          run = b0
          runStart = j
        }
        continue
      }
      flush(j)
      const corners = idx.map((k) => ({ x: X[k], y: Y[k], z: Z[k] }))
      const centre: V = { x: corners.reduce((s, p) => s + p.x, 0) / 4, y: corners.reduce((s, p) => s + p.y, 0) / 4, z: (zs[0] + zs[1] + zs[2] + zs[3]) / 4 }
      for (let e = 0; e < 4; e++) {
        const tri = [corners[e], corners[(e + 1) % 4], centre]
        for (let k = 0; k < nb; k++) {
          const p = clipBand(tri, levels[k], levels[k + 1])
          if (p.length) bands[k].push(flat(p))
        }
      }
    }
    flush(nx - 1)
  }
  return bands
}

/** iso-lines on a triangulation */
export const triLines = (x: number[], y: number[], z: number[], tris: number[][], level: number): Path[] => {
  const segs: { a: string; b: string; pa: [number, number]; pb: [number, number] }[] = []
  for (const t of tris) {
    const hits: [string, [number, number]][] = []
    for (let e = 0; e < 3; e++) {
      const p = t[e]
      const q = t[(e + 1) % 3]
      if (z[p] > level !== z[q] > level) {
        const s = (level - z[p]) / (z[q] - z[p])
        hits.push([p < q ? `${p}_${q}` : `${q}_${p}`, [x[p] + s * (x[q] - x[p]), y[p] + s * (y[q] - y[p])]])
      }
    }
    if (hits.length === 2) segs.push({ a: hits[0][0], b: hits[1][0], pa: hits[0][1], pb: hits[1][1] })
  }
  return stitch(segs)
}

export const triBands = (x: number[], y: number[], z: number[], tris: number[][], levels: number[]): Path[][] => {
  const bands: Path[][] = levels.slice(1).map(() => [])
  for (const t of tris) {
    const poly = t.map((i) => ({ x: x[i], y: y[i], z: z[i] }))
    for (let k = 0; k < bands.length; k++) {
      const p = clipBand(poly, levels[k], levels[k + 1])
      if (p.length) bands[k].push(flat(p))
    }
  }
  return bands
}

/** Delaunay triangulation (Bowyer–Watson); triangles are vertex index triples, counter-clockwise */
export const delaunay = (xs: number[], ys: number[]): number[][] => {
  const n = xs.length
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const d = Math.max(maxX - minX, maxY - minY) || 1
  const mx = (minX + maxX) / 2
  const my = (minY + maxY) / 2
  const px = [...xs, mx - 2e4 * d, mx, mx + 2e4 * d]
  const py = [...ys, my - 2e4 * d, my + 2e4 * d, my - 2e4 * d]
  interface T { a: number; b: number; c: number; cx: number; cy: number; r2: number }
  const make = (a0: number, b0: number, c0: number): T => {
    const a = a0
    let [b, c] = [b0, c0]
    const area = (px[b] - px[a]) * (py[c] - py[a]) - (py[b] - py[a]) * (px[c] - px[a])
    if (area < 0) [b, c] = [c, b]
    const bx = px[b] - px[a]
    const by = py[b] - py[a]
    const cx = px[c] - px[a]
    const cy = py[c] - py[a]
    const dd = 2 * (bx * cy - by * cx)
    const ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / dd
    const uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / dd
    return { a, b, c, cx: ux + px[a], cy: uy + py[a], r2: ux * ux + uy * uy }
  }
  let tris: T[] = [make(n, n + 1, n + 2)]
  for (let p = 0; p < n; p++) {
    const bad: T[] = []
    const keep: T[] = []
    for (const t of tris) {
      const dx = px[p] - t.cx
      const dy = py[p] - t.cy
      if (dx * dx + dy * dy < t.r2) bad.push(t)
      else keep.push(t)
    }
    const edges = new Map<string, [number, number]>()
    for (const t of bad) {
      for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
        const rev = `${v}_${u}`
        if (edges.has(rev)) edges.delete(rev)
        else edges.set(`${u}_${v}`, [u, v])
      }
    }
    tris = keep
    for (const [u, v] of edges.values()) tris.push(make(u, v, p))
  }
  return tris.filter((t) => t.a < n && t.b < n && t.c < n).map((t) => [t.a, t.b, t.c])
}

/** scipy-style Gaussian KDE with Scott's bandwidth, as violinplot uses (mlab.GaussianKDE) */
export const gaussianKde = (data: number[], points: number[]): number[] => {
  const n = data.length
  const mean = data.reduce((s, v) => s + v, 0) / n
  const variance = data.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)
  const factor = n ** -0.2
  const cov = variance * factor * factor
  const norm = Math.sqrt(2 * Math.PI * cov) * n
  return points.map((p) => data.reduce((s, v) => s + Math.exp(-((p - v) ** 2) / (2 * cov)), 0) / norm)
}
