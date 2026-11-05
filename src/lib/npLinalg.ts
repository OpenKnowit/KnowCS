// Linear algebra on plain rows of numbers, for np.linalg: solve, QR, Cholesky, SVD and eigenvalues.
// QR follows LAPACK's Householder steps, so Q and R match NumPy. The eigenvalue code is the classic EISPACK
// hqr2 (by way of JAMA), which finds them in the order LAPACK usually does. The signs of eigenvectors and
// singular vectors are a free choice that libraries make differently: A = U S Vᵀ holds whichever is picked.

export type Mat = number[][]

const zeros = (r: number, c: number): Mat => Array.from({ length: r }, () => new Array<number>(c).fill(0))
export const identity = (n: number): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
export const transposeM = (a: Mat): Mat => (a.length ? a[0].map((_, j) => a.map((row) => row[j])) : [])
export const mul = (a: Mat, b: Mat): Mat => {
  const c = b[0]?.length ?? 0
  return a.map((row) => {
    const out = new Array<number>(c).fill(0)
    row.forEach((x, k) => {
      if (x !== 0) for (let j = 0; j < c; j++) out[j] += x * b[k][j]
    })
    return out
  })
}

/** LU with partial pivoting: row swaps recorded, null if a pivot is exactly zero (what LAPACK calls singular) */
export const lu = (a: Mat): { lu: Mat; perm: number[]; sign: number; singular: boolean } => {
  const m = a.map((r) => [...r])
  const n = m.length
  const perm = m.map((_, i) => i)
  let sign = 1
  let singular = false
  for (let k = 0; k < n; k++) {
    let p = k
    for (let i = k + 1; i < n; i++) if (Math.abs(m[i][k]) > Math.abs(m[p][k])) p = i
    if (p !== k) {
      ;[m[p], m[k]] = [m[k], m[p]]
      ;[perm[p], perm[k]] = [perm[k], perm[p]]
      sign = -sign
    }
    if (m[k][k] === 0) {
      singular = true
      continue
    }
    for (let i = k + 1; i < n; i++) {
      m[i][k] /= m[k][k]
      for (let j = k + 1; j < n; j++) m[i][j] -= m[i][k] * m[k][j]
    }
  }
  return { lu: m, perm, sign, singular }
}

/** solve A x = B for each column of B; null if A is singular */
export const solveM = (a: Mat, b: Mat): Mat | null => {
  const { lu: m, perm, singular } = lu(a)
  if (singular) return null
  const n = a.length
  const out = perm.map((p) => [...b[p]])
  const cols = b[0]?.length ?? 0
  for (let c = 0; c < cols; c++) {
    for (let i = 1; i < n; i++) for (let j = 0; j < i; j++) out[i][c] -= m[i][j] * out[j][c]
    for (let i = n - 1; i >= 0; i--) {
      for (let j = i + 1; j < n; j++) out[i][c] -= m[i][j] * out[j][c]
      out[i][c] /= m[i][i]
    }
  }
  return out
}

/** the lower triangle L with L Lᵀ = A; null if A is not positive definite */
export const cholesky = (a: Mat): Mat | null => {
  const n = a.length
  const l = zeros(n, n)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      if (i === j) {
        if (s <= 0) return null
        l[i][i] = Math.sqrt(s)
      } else l[i][j] = s / l[j][j]
    }
  }
  return l
}

/** Householder QR as LAPACK's dgeqr2 does it: complete Q (m × m) and R (m × n) */
export const qr = (a: Mat): { q: Mat; r: Mat } => {
  const m = a.length
  const n = a[0]?.length ?? 0
  const r = a.map((row) => [...row])
  const steps: { v: number[]; tau: number; k: number }[] = []
  for (let k = 0; k < Math.min(m, n); k++) {
    let xnorm = 0
    for (let i = k + 1; i < m; i++) xnorm = Math.hypot(xnorm, r[i][k])
    if (xnorm === 0) continue
    const alpha = r[k][k]
    const beta = -Math.sign(alpha || 1) * Math.hypot(alpha, xnorm)
    const tau = (beta - alpha) / beta
    const v = new Array<number>(m).fill(0)
    v[k] = 1
    for (let i = k + 1; i < m; i++) v[i] = r[i][k] / (alpha - beta)
    for (let j = k; j < n; j++) {
      let s = 0
      for (let i = k; i < m; i++) s += v[i] * r[i][j]
      s *= tau
      for (let i = k; i < m; i++) r[i][j] -= s * v[i]
    }
    for (let i = k + 1; i < m; i++) r[i][k] = 0
    steps.push({ v, tau, k })
  }
  const q = identity(m)
  for (const { v, tau, k } of steps.reverse()) {
    for (let j = 0; j < m; j++) {
      let s = 0
      for (let i = k; i < m; i++) s += v[i] * q[i][j]
      s *= tau
      for (let i = k; i < m; i++) q[i][j] -= s * v[i]
    }
  }
  return { q, r }
}

/** symmetric eigenproblem by cyclic Jacobi: ascending eigenvalues, unit eigenvectors as columns */
export const eigh = (a: Mat): { values: number[]; vectors: Mat } => {
  const n = a.length
  // the lower triangle is the input, as in np.linalg.eigh
  const m = a.map((row, i) => row.map((_, j) => (i >= j ? a[i][j] : a[j][i])))
  const v = identity(n)
  for (let sweep = 0; sweep < 64; sweep++) {
    let off = 0
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) off += m[i][j] * m[i][j]
    if (off < 1e-300) break
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (m[p][q] === 0) continue
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q])
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.hypot(theta, 1))
        const c = 1 / Math.hypot(t, 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const x = m[k][p]
          const y = m[k][q]
          m[k][p] = c * x - s * y
          m[k][q] = s * x + c * y
        }
        for (let k = 0; k < n; k++) {
          const x = m[p][k]
          const y = m[q][k]
          m[p][k] = c * x - s * y
          m[q][k] = s * x + c * y
        }
        for (let k = 0; k < n; k++) {
          const x = v[k][p]
          const y = v[k][q]
          v[k][p] = c * x - s * y
          v[k][q] = s * x + c * y
        }
      }
    }
  }
  const order = m.map((_, i) => i).sort((x, y) => m[x][x] - m[y][y])
  const vectors = zeros(n, n)
  order.forEach((from, to) => {
    // each column's last non-zero entry is positive, so a result does not depend on the sweep order
    let flip = 1
    for (let i = n - 1; i >= 0; i--) {
      if (Math.abs(v[i][from]) > 1e-12) {
        flip = v[i][from] < 0 ? -1 : 1
        break
      }
    }
    for (let i = 0; i < n; i++) vectors[i][to] = flip * v[i][from]
  })
  return { values: order.map((i) => m[i][i]), vectors }
}

/** thin SVD by one-sided Jacobi: singular values descending, U m × k, V n × k */
export const svdThin = (a: Mat): { u: Mat; s: number[]; v: Mat } => {
  const m = a.length
  const n = a[0]?.length ?? 0
  if (m < n) {
    const t = svdThin(transposeM(a))
    return { u: t.v, s: t.s, v: t.u }
  }
  const w = a.map((row) => [...row])
  const v = identity(n)
  for (let sweep = 0; sweep < 64; sweep++) {
    let rotated = false
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < m; i++) {
          alpha += w[i][p] * w[i][p]
          beta += w[i][q] * w[i][q]
          gamma += w[i][p] * w[i][q]
        }
        if (Math.abs(gamma) <= 1e-16 * Math.sqrt(alpha * beta) || gamma === 0) continue
        rotated = true
        const zeta = (beta - alpha) / (2 * gamma)
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.hypot(zeta, 1))
        const c = 1 / Math.hypot(t, 1)
        const s = c * t
        for (let i = 0; i < m; i++) {
          const x = w[i][p]
          w[i][p] = c * x - s * w[i][q]
          w[i][q] = s * x + c * w[i][q]
        }
        for (let i = 0; i < n; i++) {
          const x = v[i][p]
          v[i][p] = c * x - s * v[i][q]
          v[i][q] = s * x + c * v[i][q]
        }
      }
    }
    if (!rotated) break
  }
  const norms = Array.from({ length: n }, (_, j) => Math.sqrt(w.reduce((t, row) => t + row[j] * row[j], 0)))
  const order = norms.map((_, j) => j).sort((x, y) => norms[y] - norms[x])
  const big = norms[order[0]] ?? 0
  const u = zeros(m, n)
  const vs = zeros(n, n)
  order.forEach((from, to) => {
    const positive = norms[from] > big * 1e-14
    // LAPACK's Householder steps usually leave a column of positive data pointing negative
    let total = 0
    if (positive) for (let i = 0; i < m; i++) total += w[i][from]
    const flip = total > 0 ? -1 : 1
    for (let i = 0; i < m; i++) u[i][to] = positive ? (flip * w[i][from]) / norms[from] : 0
    for (let i = 0; i < n; i++) vs[i][to] = flip * v[i][from]
  })
  completeColumns(u, order.map((from) => norms[from] > big * 1e-14))
  return { u, s: order.map((j) => norms[j]), v: vs }
}

/** fill the columns not marked done with unit vectors orthogonal to the others (Gram–Schmidt on the axes) */
const completeColumns = (u: Mat, done: boolean[]): void => {
  const m = u.length
  const cols = u[0]?.length ?? 0
  const have = done.map((d, j) => (d ? j : -1)).filter((j) => j >= 0)
  let axis = 0
  for (let j = 0; j < cols; j++) {
    if (done[j]) continue
    for (; axis < m; axis++) {
      const x = new Array<number>(m).fill(0)
      x[axis] = 1
      for (let pass = 0; pass < 2; pass++) {
        for (const h of have) {
          let d = 0
          for (let i = 0; i < m; i++) d += x[i] * u[i][h]
          for (let i = 0; i < m; i++) x[i] -= d * u[i][h]
        }
      }
      const len = Math.sqrt(x.reduce((t, y) => t + y * y, 0))
      if (len > 1e-8) {
        for (let i = 0; i < m; i++) u[i][j] = x[i] / len
        have.push(j)
        axis++
        break
      }
    }
  }
}

/** the full SVD: U is m × m and V is n × n */
export const svdFull = (a: Mat): { u: Mat; s: number[]; v: Mat } => {
  const m = a.length
  const n = a[0]?.length ?? 0
  const t = svdThin(a)
  const widen = (x: Mat, size: number): Mat => {
    const out = zeros(size, size)
    const have = x[0]?.length ?? 0
    for (let i = 0; i < size; i++) for (let j = 0; j < have; j++) out[i][j] = x[i][j]
    completeColumns(out, Array.from({ length: size }, (_, j) => j < have))
    return out
  }
  return { u: widen(t.u, m), s: t.s, v: widen(t.v, n) }
}

/** eigenvalues of any real square matrix (imaginary parts in im), and unit eigenvectors when they are all real */
export const eigGeneral = (a: Mat): { re: number[]; im: number[]; vectors: Mat | null } => {
  const nn = a.length
  const H = a.map((r) => [...r])
  const V = identity(nn)
  const d = new Array<number>(nn).fill(0)
  const e = new Array<number>(nn).fill(0)
  const ort = new Array<number>(nn).fill(0)
  const high = nn - 1
  // reduce to Hessenberg form
  for (let m = 1; m <= high - 1; m++) {
    let scale = 0
    for (let i = m; i <= high; i++) scale += Math.abs(H[i][m - 1])
    if (scale === 0) continue
    let h = 0
    for (let i = high; i >= m; i--) {
      ort[i] = H[i][m - 1] / scale
      h += ort[i] * ort[i]
    }
    let g = Math.sqrt(h)
    if (ort[m] > 0) g = -g
    h -= ort[m] * g
    ort[m] -= g
    for (let j = m; j < nn; j++) {
      let f = 0
      for (let i = high; i >= m; i--) f += ort[i] * H[i][j]
      f /= h
      for (let i = m; i <= high; i++) H[i][j] -= f * ort[i]
    }
    for (let i = 0; i <= high; i++) {
      let f = 0
      for (let j = high; j >= m; j--) f += ort[j] * H[i][j]
      f /= h
      for (let j = m; j <= high; j++) H[i][j] -= f * ort[j]
    }
    ort[m] *= scale
    H[m][m - 1] = scale * g
  }
  for (let m = high - 1; m >= 1; m--) {
    if (H[m][m - 1] === 0) continue
    for (let i = m + 1; i <= high; i++) ort[i] = H[i][m - 1]
    for (let j = m; j <= high; j++) {
      let g = 0
      for (let i = m; i <= high; i++) g += ort[i] * V[i][j]
      g = g / ort[m] / H[m][m - 1]
      for (let i = m; i <= high; i++) V[i][j] += g * ort[i]
    }
  }
  // Francis double-shift QR on the Hessenberg matrix
  const eps = 2 ** -52
  let exshift = 0
  let p = 0
  let q = 0
  let r = 0
  let s = 0
  let z = 0
  let norm = 0
  for (let i = 0; i < nn; i++) for (let j = Math.max(i - 1, 0); j < nn; j++) norm += Math.abs(H[i][j])
  let n = nn - 1
  let iter = 0
  while (n >= 0) {
    if (iter > 300) throw new Error('Eigenvalues did not converge')
    let l = n
    while (l > 0) {
      s = Math.abs(H[l - 1][l - 1]) + Math.abs(H[l][l])
      if (s === 0) s = norm
      if (Math.abs(H[l][l - 1]) < eps * s) break
      l--
    }
    if (l === n) {
      H[n][n] += exshift
      d[n] = H[n][n]
      e[n] = 0
      n--
      iter = 0
    } else if (l === n - 1) {
      const w = H[n][n - 1] * H[n - 1][n]
      p = (H[n - 1][n - 1] - H[n][n]) / 2
      q = p * p + w
      z = Math.sqrt(Math.abs(q))
      H[n][n] += exshift
      H[n - 1][n - 1] += exshift
      let x = H[n][n]
      if (q >= 0) {
        z = p >= 0 ? p + z : p - z
        d[n - 1] = x + z
        d[n] = d[n - 1]
        if (z !== 0) d[n] = x - w / z
        e[n - 1] = 0
        e[n] = 0
        x = H[n][n - 1]
        s = Math.abs(x) + Math.abs(z)
        p = x / s
        q = z / s
        r = Math.sqrt(p * p + q * q)
        p /= r
        q /= r
        for (let j = n - 1; j < nn; j++) {
          z = H[n - 1][j]
          H[n - 1][j] = q * z + p * H[n][j]
          H[n][j] = q * H[n][j] - p * z
        }
        for (let i = 0; i <= n; i++) {
          z = H[i][n - 1]
          H[i][n - 1] = q * z + p * H[i][n]
          H[i][n] = q * H[i][n] - p * z
        }
        for (let i = 0; i <= high; i++) {
          z = V[i][n - 1]
          V[i][n - 1] = q * z + p * V[i][n]
          V[i][n] = q * V[i][n] - p * z
        }
      } else {
        d[n - 1] = x + p
        d[n] = x + p
        e[n - 1] = z
        e[n] = -z
      }
      n -= 2
      iter = 0
    } else {
      let x = H[n][n]
      let y = H[n - 1][n - 1]
      let w = H[n][n - 1] * H[n - 1][n]
      if (iter === 10) {
        exshift += x
        for (let i = 0; i <= n; i++) H[i][i] -= x
        s = Math.abs(H[n][n - 1]) + Math.abs(H[n - 1][n - 2])
        x = y = 0.75 * s
        w = -0.4375 * s * s
      }
      if (iter === 30) {
        s = (y - x) / 2
        s = s * s + w
        if (s > 0) {
          s = Math.sqrt(s)
          if (y < x) s = -s
          s = x - w / ((y - x) / 2 + s)
          for (let i = 0; i <= n; i++) H[i][i] -= s
          exshift += s
          x = y = w = 0.964
        }
      }
      iter++
      let m = n - 2
      while (m >= l) {
        z = H[m][m]
        r = x - z
        s = y - z
        p = (r * s - w) / H[m + 1][m] + H[m][m + 1]
        q = H[m + 1][m + 1] - z - r - s
        r = H[m + 2][m + 1]
        s = Math.abs(p) + Math.abs(q) + Math.abs(r)
        p /= s
        q /= s
        r /= s
        if (m === l) break
        if (Math.abs(H[m][m - 1]) * (Math.abs(q) + Math.abs(r)) < eps * (Math.abs(p) * (Math.abs(H[m - 1][m - 1]) + Math.abs(z) + Math.abs(H[m + 1][m + 1])))) break
        m--
      }
      for (let i = m + 2; i <= n; i++) {
        H[i][i - 2] = 0
        if (i > m + 2) H[i][i - 3] = 0
      }
      for (let k = m; k <= n - 1; k++) {
        const notlast = k !== n - 1
        if (k !== m) {
          p = H[k][k - 1]
          q = H[k + 1][k - 1]
          r = notlast ? H[k + 2][k - 1] : 0
          x = Math.abs(p) + Math.abs(q) + Math.abs(r)
          if (x === 0) continue
          p /= x
          q /= x
          r /= x
        }
        s = Math.sqrt(p * p + q * q + r * r)
        if (p < 0) s = -s
        if (s === 0) continue
        if (k !== m) H[k][k - 1] = -s * x
        else if (l !== m) H[k][k - 1] = -H[k][k - 1]
        p += s
        x = p / s
        y = q / s
        z = r / s
        q /= p
        r /= p
        for (let j = k; j < nn; j++) {
          p = H[k][j] + q * H[k + 1][j]
          if (notlast) {
            p += r * H[k + 2][j]
            H[k + 2][j] -= p * z
          }
          H[k][j] -= p * x
          H[k + 1][j] -= p * y
        }
        for (let i = 0; i <= Math.min(n, k + 3); i++) {
          p = x * H[i][k] + y * H[i][k + 1]
          if (notlast) {
            p += z * H[i][k + 2]
            H[i][k + 2] -= p * r
          }
          H[i][k] -= p
          H[i][k + 1] -= p * q
        }
        for (let i = 0; i <= high; i++) {
          p = x * V[i][k] + y * V[i][k + 1]
          if (notlast) {
            p += z * V[i][k + 2]
            V[i][k + 2] -= p * r
          }
          V[i][k] -= p
          V[i][k + 1] -= p * q
        }
      }
    }
  }
  if (e.some((x) => x !== 0)) return { re: d, im: e, vectors: null }
  // back-substitute to the eigenvectors of the triangular form, then transform back
  if (norm !== 0) {
    for (n = nn - 1; n >= 0; n--) {
      p = d[n]
      H[n][n] = 1
      let l = n
      for (let i = n - 1; i >= 0; i--) {
        const w = H[i][i] - p
        r = 0
        for (let j = l; j <= n; j++) r += H[i][j] * H[j][n]
        l = i
        H[i][n] = w !== 0 ? -r / w : -r / (eps * norm)
        const t = Math.abs(H[i][n])
        if (eps * t * t > 1) for (let j = i; j <= n; j++) H[j][n] /= t
      }
    }
    for (let j = nn - 1; j >= 0; j--) {
      for (let i = 0; i <= high; i++) {
        z = 0
        for (let k = 0; k <= Math.min(j, high); k++) z += V[i][k] * H[k][j]
        V[i][j] = z
      }
    }
  }
  for (let j = 0; j < nn; j++) {
    const len = Math.sqrt(V.reduce((t, row) => t + row[j] * row[j], 0))
    if (len > 0) for (let i = 0; i < nn; i++) V[i][j] /= len
  }
  return { re: d, im: e, vectors: V }
}
