// Assemble captured page trees into one .pen document.
// node scripts/pencil-sync/assemble.mjs <captureDir> <out.pen> <row>:<key>,<key>... [<row>:...]
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'

const [capDir, outPen, ...rows] = process.argv.slice(2)
const penDir = dirname(outPen)
const GAP = 240

const pages = []
const palette = {}
for (const row of rows) {
  const [title, keys] = row.split(':')
  pages.push({ title, items: keys.split(',').map((k) => JSON.parse(readFileSync(join(capDir, `${k}.json`), 'utf8'))) })
}
for (const r of pages) for (const p of r.items) Object.assign(palette, p.palette)

// hex → variable name (first Tailwind name wins; keep white/black)
const byHex = new Map()
const order = Object.keys(palette).sort((a, b) => (/^(slate|blue|indigo)/.test(b) ? 1 : 0) - (/^(slate|blue|indigo)/.test(a) ? 1 : 0))
for (const name of order) if (!byHex.has(palette[name])) byHex.set(palette[name], name)
const used = new Set()
const tok = (c) => {
  if (typeof c !== 'string' || !c.startsWith('#')) return c
  const n = byHex.get(c.toUpperCase())
  if (!n) return c
  used.add(n)
  return `$${n}`
}

const fixPaint = (f) => {
  if (Array.isArray(f)) return f.map(fixPaint)
  if (typeof f === 'string') return tok(f)
  if (f && f.type === 'gradient') return { ...f, colors: f.colors.map((s) => ({ ...s, color: tok(s.color) })) }
  return f
}

const assetsOut = join(penDir, 'assets')
const prep = (node, key, akey = key) => {
  delete node.__tag
  node.id = `${key.replace(/[^a-z0-9]/gi, '')}_${node.id}`
  if (node.fill) {
    if (node.fill.type === 'image') {
      if (!node.fill.url.startsWith('__ASSET__/')) return node
      const src = join(capDir, 'assets', akey, node.fill.url.replace('__ASSET__/', ''))
      const dst = join(assetsOut, akey, node.fill.url.replace('__ASSET__/', ''))
      if (existsSync(src)) { mkdirSync(dirname(dst), { recursive: true }); copyFileSync(src, dst) }
      node.fill = { ...node.fill, url: './' + relative(penDir, dst) }
    } else node.fill = fixPaint(node.fill)
  }
  if (node.stroke) node.stroke = fixPaint(node.stroke)
  if (node.effect) {
    const fx = (e) => (e.color ? { ...e, color: tok(e.color) } : e)
    node.effect = Array.isArray(node.effect) ? node.effect.map(fx) : fx(node.effect)
  }
  if (node.children) {
    node.children.forEach((c) => prep(c, key, akey))
    if (!node.children.length) delete node.children
  }
  return node
}

// --- shared chrome (header / footer) → reusable components + instances with overrides
const shape = (n) => `${n.type}(${(n.children || []).map(shape).join(',')})`
const SKIP = new Set(['id', 'children', 'name', 'x', 'y', '__tag'])
const diff = (a, b, out, path) => {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  const d = {}
  for (const k of keys) if (!SKIP.has(k) && JSON.stringify(a[k]) !== JSON.stringify(b[k])) d[k] = b[k] === undefined ? null : b[k]
  for (const k of ['x', 'y']) if (path && a[k] !== b[k]) d[k] = b[k]
  if (path && Object.keys(d).length) out[path] = d
  ;(a.children || []).forEach((c, i) => diff(c, b.children[i], out, c.id))
  return d
}
const findTag = (n, tag) => {
  for (const [idx, c] of (n.children || []).entries()) {
    if (c.__tag === tag) return { parent: n, idx }
    const r = findTag(c, tag)
    if (r) return r
  }
  return null
}
const components = []
for (const tag of ['HEADER', 'FOOTER']) {
  const groups = new Map()
  for (const row of pages) for (const p of row.items) {
    const hit = findTag(p.tree, tag)
    if (!hit) continue
    const sig = shape(hit.parent.children[hit.idx])
    if (!groups.has(sig)) groups.set(sig, [])
    groups.get(sig).push({ p, parent: hit.parent, idx: hit.idx })
  }
  let v = 0
  for (const members of [...groups.values()].sort((a, b) => b.length - a.length)) {
    if (members.length < 2) continue
    v++
    const first = members[0]
    const comp = prep(structuredClone(first.parent.children[first.idx]), `${tag.toLowerCase()}${v}`, first.p.key)
    comp.reusable = true
    comp.name = `Site ${tag === 'HEADER' ? 'Header' : 'Footer'}${v > 1 ? ` ${v}` : ''}`
    components.push(comp)
    for (const { p, parent, idx } of members) {
      const node = prep(structuredClone(parent.children[idx]), `${tag.toLowerCase()}${v}`, p.key)
      // ids in node now mirror comp ids (same prefix & same capture ids only for the first member) → align by position
      const relabel = (a, b) => { b.id = a.id; (a.children || []).forEach((c, i) => relabel(c, b.children[i])) }
      relabel(comp, node)
      const descendants = {}
      const rootDiff = diff(comp, node, descendants, '')
      parent.children[idx] = { type: 'ref', ref: comp.id, id: `${p.key.replace(/[^a-z0-9]/gi, '')}_${tag.toLowerCase()}`, name: comp.name, x: node.x, y: node.y, ...rootDiff, ...(Object.keys(descendants).length ? { descendants } : {}) }
    }
  }
}

const pageNodes = []
let y = 0
for (const row of pages) {
  let x = 0
  let rowH = 0
  for (const p of row.items) {
    const t = prep(p.tree, p.key)
    t.name = `${row.title} / ${p.key}`
    t.x = x
    t.y = y
    t.context = `Synced from https://knowcs.online/${p.path} (English, 1440px, light).`
    pageNodes.push(t)
    x += t.width + GAP
    rowH = Math.max(rowH, t.height)
  }
  y += rowH + GAP * 2
}

const children = []
let boardH = 0
// row 0: tokens board + components
{
  const sw = []
  const names = [...used].sort((a, b) => a.localeCompare(b, 'en', { numeric: true }))
  const fams = [...new Set(names.map((n) => n.replace(/-\d+$/, '')))]
  let fy = 72
  for (const fam of fams) {
    const row = names.filter((n) => n.replace(/-\d+$/, '') === fam)
    sw.push({ type: 'text', id: `tok_l_${fam}`, name: fam, content: fam, x: 32, y: fy + 14, fontFamily: 'Inter', fontSize: 12, fontWeight: '700', fill: '#334155' })
    row.forEach((n, i) => {
      sw.push({ type: 'rectangle', id: `tok_s_${n}`, name: n, x: 140 + i * 76, y: fy, width: 64, height: 40, cornerRadius: 8, fill: `$${n}`, stroke: '#E2E8F0', strokeWidth: 1, strokeAlignment: 'inner' })
      sw.push({ type: 'text', id: `tok_t_${n}`, name: `${n} label`, content: n.replace(`${fam}-`, '') || n, x: 140 + i * 76, y: fy + 44, fontFamily: 'JetBrains Mono', fontSize: 10, fill: '#64748B' })
    })
    fy += 72
  }
  const typo = [['Display · 48 / 900', 48, '900'], ['Heading · 30 / 900', 30, '900'], ['Title · 18 / 800', 18, '800'], ['Body · 16 / 400', 16, '400'], ['Small · 13 / 500', 13, '500'], ['LABEL · 11 / 800 · tracking', 11, '800']]
  let ty = 72
  for (const [t, s, w] of typo) { sw.push({ type: 'text', id: `tok_ty_${s}`, name: t, content: t, x: 1100, y: ty, fontFamily: 'Inter', fontSize: s, fontWeight: w, fill: '#0F172A' }); ty += s * 1.5 + 16 }
  sw.push({ type: 'text', id: 'tok_ty_mono', name: 'Mono', content: 'Mono · JetBrains Mono · np.argmin(d, axis=1)', x: 1100, y: ty, fontFamily: 'JetBrains Mono', fontSize: 14, fill: '#0F172A' })
  sw.push({ type: 'text', id: 'tok_title', name: 'Title', content: 'KnowCS · Design tokens (Tailwind colours used on knowcs.online)', x: 32, y: 24, fontFamily: 'Inter', fontSize: 20, fontWeight: '800', fill: '#0F172A' })
  const board = { type: 'frame', id: 'tokens', name: 'Design Tokens', x: 0, y: 0, width: 1700, height: Math.max(fy, ty + 60) + 24, layout: 'none', fill: '#FFFFFF', cornerRadius: 16, stroke: '#E2E8F0', strokeWidth: 1, children: sw }
  children.push(board)
  let cx = board.width + GAP
  let ch = board.height
  for (const comp of components) { comp.x = cx; comp.y = 0; children.push(comp); cx += comp.width + GAP; ch = Math.max(ch, comp.height) }
  boardH = ch + GAP * 2
}
for (const t of pageNodes) { t.y += boardH; children.push(t) }

const variables = {}
for (const n of [...used].sort()) variables[n] = { type: 'color', value: palette[n] }
writeFileSync(outPen, JSON.stringify({ version: '2.13', children, variables }, null, 2))
let count = 0
const cnt = (n) => { count++; (n.children || []).forEach(cnt) }
children.forEach(cnt)
console.log(`wrote ${outPen}: ${children.length} pages, ${count} nodes, ${Object.keys(variables).length} colour variables`)
