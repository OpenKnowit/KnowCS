// Runs inside the page (via CDP Runtime.evaluate). Walks the rendered DOM and returns a .pen node tree
// with absolute positions, plus a list of regions that must be captured as images (canvas, img, KaTeX, sliders…).
// Returns { tree, rasters, palette }.
(() => {
  const RASTER_SEL = 'img,canvas,video,iframe,picture,.katex,input[type=range],input[type=checkbox],input[type=radio],input[type=color],select,progress,meter'
  const SEMANTIC = new Set(['HEADER', 'NAV', 'MAIN', 'FOOTER', 'SECTION', 'ARTICLE', 'ASIDE'])
  let seq = 0
  const nid = (p) => `${p}${(seq++).toString(36)}`
  const rasters = []

  // --- colour: any CSS colour (oklch, lab, rgb…) → #RRGGBB[AA] via a 1×1 canvas
  const cv = document.createElement('canvas'); cv.width = cv.height = 1
  const cx = cv.getContext('2d', { willReadFrequently: true })
  const colorCache = new Map()
  const hex = (css) => {
    if (!css || css === 'transparent' || css === 'none') return null
    if (colorCache.has(css)) return colorCache.get(css)
    cx.clearRect(0, 0, 1, 1); cx.fillStyle = '#000'; cx.fillStyle = css; cx.fillRect(0, 0, 1, 1)
    const [r, g, b, a] = cx.getImageData(0, 0, 1, 1).data
    // getImageData premultiplies poorly for low alpha; recover from the parsed alpha instead
    let out = null
    if (a > 0) {
      const h = (v) => v.toString(16).padStart(2, '0').toUpperCase()
      // re-read without premultiplication: draw on white and on black is overkill; use fillStyle normalisation
      const norm = cx.fillStyle // '#rrggbb' or 'rgba(r, g, b, a)'
      if (norm.startsWith('#')) out = norm.toUpperCase()
      else {
        const m = norm.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/)
        if (m) out = '#' + h(+m[1]) + h(+m[2]) + h(+m[3]) + (m[4] !== undefined && +m[4] < 1 ? h(Math.round(+m[4] * 255)) : '')
        else out = '#' + h(r) + h(g) + h(b) + (a < 255 ? h(a) : '')
      }
    }
    colorCache.set(css, out)
    return out
  }
  const COLOR_RE = /(?:oklch|oklab|lab|lch|rgba?|hsla?|color)\([^()]*(?:\([^()]*\)[^()]*)*\)|#[0-9a-fA-F]{3,8}\b|\btransparent\b|\bwhite\b|\bblack\b/g
  const withAlpha = (h, a) => {
    if (!h || a >= 1) return h
    const base = h.slice(0, 7); const old = h.length === 9 ? parseInt(h.slice(7), 16) / 255 : 1
    const v = Math.round(old * a * 255)
    return v >= 255 ? base : base + v.toString(16).padStart(2, '0').toUpperCase()
  }

  // linear-gradient(...) → pencil gradient fill
  const gradient = (bgImage) => {
    const m = bgImage.match(/(linear|radial)-gradient\((.*)\)\s*$/)
    if (!m) return null
    const body = m[2]
    let rot = 180
    const dir = body.match(/^\s*(to [a-z ]+|-?[\d.]+deg)/)
    if (dir) {
      const d = dir[1]
      const map = { 'to top': 0, 'to right': 90, 'to bottom': 180, 'to left': 270, 'to top right': 45, 'to right top': 45, 'to bottom right': 135, 'to right bottom': 135, 'to bottom left': 225, 'to left bottom': 225, 'to top left': 315, 'to left top': 315 }
      const css = d.endsWith('deg') ? parseFloat(d) : (map[d.trim().replace(/\s+in\s+.*/, '')] ?? 180)
      rot = (360 - css) % 360
    }
    const cols = [...body.matchAll(COLOR_RE)].map((x) => hex(x[0]) ?? '#00000000')
    if (cols.length < 2) return cols.length ? cols[0] : null
    return {
      type: 'gradient', gradientType: m[1] === 'radial' ? 'radial' : 'linear', rotation: rot,
      colors: cols.map((c, i) => ({ color: c, position: +(i / (cols.length - 1)).toFixed(3) })),
    }
  }

  const shadow = (bs) => {
    if (!bs || bs === 'none') return null
    const out = []
    for (const part of bs.split(/,(?![^()]*\))/)) {
      if (/inset/.test(part)) continue
      const col = part.match(COLOR_RE); const nums = part.replace(COLOR_RE, '').trim().split(/\s+/).map(parseFloat)
      const c = col ? hex(col[0]) : '#00000040'
      if (!c || /00$/.test(c) && c.length === 9) continue
      if (nums.every((n) => !n)) continue
      out.push({ type: 'shadow', shadowType: 'outer', offset: { x: nums[0] || 0, y: nums[1] || 0 }, blur: nums[2] || 0, spread: nums[3] || 0, color: c })
    }
    return out.length ? (out.length === 1 ? out[0] : out) : null
  }

  const SANS = 'Inter', MONO = 'JetBrains Mono'
  const family = (ff) => (/mono|courier|menlo|consolas|sf mono|jetbrains/i.test(ff) ? MONO : SANS)
  const r1 = (v) => Math.round(v * 10) / 10

  const vis = (el) => el.checkVisibility ? el.checkVisibility({ opacityProperty: true, visibilityProperty: true, contentVisibilityAuto: true }) : true

  const label = (el) => {
    const a = el.getAttribute('aria-label') || el.getAttribute('title')
    if (a) return a.slice(0, 40)
    const h = el.querySelector?.(':scope h1,:scope h2,:scope h3,:scope h4')
    const t = (h?.textContent || el.textContent || '').replace(/\s+/g, ' ').trim()
    const tag = el.tagName.toLowerCase()
    return t ? `${tag} · ${t.slice(0, 32)}` : tag
  }

  // --- frame for an element with visible box decoration
  const boxProps = (el, cs) => {
    const p = {}
    let fill = null
    const bgc = hex(cs.backgroundColor)
    if (cs.backgroundImage && cs.backgroundImage !== 'none' && cs.backgroundClip !== 'text') {
      const g = gradient(cs.backgroundImage)
      if (g) fill = bgc && typeof g === 'object' ? [bgc, g] : g
    }
    if (!fill && bgc && cs.backgroundClip !== 'text') fill = bgc
    if (fill) p.fill = fill
    const sides = ['Top', 'Right', 'Bottom', 'Left'].map((s) => ({ w: parseFloat(cs[`border${s}Width`]) || 0, c: cs[`border${s}Style`] === 'none' ? null : hex(cs[`border${s}Color`]) }))
    const vis = sides.filter((s) => s.w > 0 && s.c)
    if (vis.length) {
      p.stroke = vis[0].c
      const ws = sides.map((s) => (s.c ? s.w : 0))
      p.strokeWidth = ws.every((w) => w === ws[0]) ? ws[0] : { top: ws[0], right: ws[1], bottom: ws[2], left: ws[3] }
      p.strokeAlignment = 'inner'
    }
    const rad = ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft'].map((c) => parseFloat(cs[`border${c}Radius`]) || 0)
    const r = el.getBoundingClientRect()
    const cap = Math.min(r.width, r.height) / 2
    const rr = rad.map((v) => Math.min(v, cap))
    if (rr.some((v) => v)) p.cornerRadius = rr.every((v) => v === rr[0]) ? r1(rr[0]) : rr.map(r1)
    const sh = shadow(cs.boxShadow)
    if (sh) p.effect = sh
    if (+cs.opacity < 1) p.opacity = +(+cs.opacity).toFixed(2)
    if (cs.overflow !== 'visible' && (p.fill || p.stroke || p.cornerRadius)) p.clip = true
    return p
  }

  // --- text: one text node → one or more pencil text nodes (absolute)
  const textNodesFor = (tn, pel, ox, oy) => {
    const cs = getComputedStyle(pel)
    const raw = tn.data
    if (!raw.trim()) return []
    const pre = /^pre/.test(cs.whiteSpace) || cs.whiteSpace === 'break-spaces'
    let color = hex(cs.color)
    let fill = color
    if (cs.backgroundClip === 'text' || (cs.webkitBackgroundClip === 'text')) {
      fill = gradient(cs.backgroundImage) || color || '#0F172A'
    } else if (!color) return []
    const fs = parseFloat(cs.fontSize)
    const lhPx = cs.lineHeight === 'normal' ? fs * 1.2 : parseFloat(cs.lineHeight)
    const tt = cs.textTransform
    const xf = (s) => (tt === 'uppercase' ? s.toUpperCase() : tt === 'lowercase' ? s.toLowerCase() : tt === 'capitalize' ? s.replace(/\b\w/g, (c) => c.toUpperCase()) : s)
    const style = {
      fill, fontFamily: family(cs.fontFamily), fontSize: r1(fs), fontWeight: String(cs.fontWeight),
      lineHeight: +(lhPx / fs).toFixed(3),
    }
    if (cs.fontStyle === 'italic') style.fontStyle = 'italic'
    const ls = parseFloat(cs.letterSpacing); if (ls) style.letterSpacing = r1(ls)
    if (/underline/.test(cs.textDecorationLine)) style.underline = true
    if (/line-through/.test(cs.textDecorationLine)) style.strikethrough = true
    const pOp = +cs.opacity < 1 ? +cs.opacity : 1

    // group characters into rendered lines
    const range = document.createRange()
    const lines = []
    let cur = null
    for (let i = 0; i < raw.length; i++) {
      range.setStart(tn, i); range.setEnd(tn, i + 1)
      const rects = range.getClientRects()
      const ch = raw[i]
      if (!rects.length) { if (cur && /\s/.test(ch) && !pre) cur.pendingSpace = true; continue }
      const rc = rects[0]
      if (rc.width === 0 && /\s/.test(ch)) { if (cur) cur.pendingSpace = true; continue }
      const mid = rc.top + rc.height / 2
      if (!cur || Math.abs(mid - cur.mid) > lhPx * 0.5 || (ch === '\n' && pre)) {
        if (ch === '\n') { cur = null; continue }
        cur = { mid, l: rc.left, r: rc.right, t: rc.top, b: rc.bottom, s: '' }
        lines.push(cur)
      } else if (cur.pendingSpace) { cur.s += ' ' }
      cur.pendingSpace = false
      cur.s += /\s/.test(ch) && !pre ? ' ' : ch
      cur.l = Math.min(cur.l, rc.left); cur.r = Math.max(cur.r, rc.right)
    }
    const out = []
    const clean = (s) => (pre ? s.replace(/\s+$/, '') : s.replace(/ +/g, ' ').trim())
    const mk = (ln, extra) => {
      const node = {
        type: 'text', id: nid('t'), name: clean(ln.s).slice(0, 40) || 'text', content: xf(clean(ln.s)),
        x: r1(ln.l - ox), y: r1(ln.mid - lhPx / 2 - oy), ...style, ...extra,
      }
      if (pOp < 1) node.opacity = +pOp.toFixed(2)
      return node
    }
    const ls2 = lines.filter((l) => clean(l.s))
    if (!ls2.length) return []
    const left0 = ls2[0].l
    const block = ls2.length > 1 && ls2.every((l) => Math.abs(l.l - left0) < 1.5) && !pre
    if (block) {
      const maxR = Math.max(...ls2.map((l) => l.r))
      const n = mk(ls2[0], { textGrowth: 'fixed-width', width: r1(maxR - left0 + 3) })
      n.content = xf(clean(ls2.map((l) => l.s).join(' ')))
      n.name = n.content.slice(0, 40)
      if (cs.textAlign === 'justify') n.textAlign = 'justify'
      out.push(n)
    } else {
      for (const ln of ls2) out.push(mk(ln))
    }
    return out
  }

  // --- svg → frame of shapes (or a lucide icon)
  const svgNode = (svg, ox, oy) => {
    const r = svg.getBoundingClientRect()
    const cls = svg.getAttribute('class') || ''
    const lm = cls.match(/lucide-([a-z0-9-]+)/)
    if (lm && /\blucide\b/.test(cls)) {
      const cs = getComputedStyle(svg)
      return { type: 'icon', id: nid('i'), name: lm[1], library: 'lucide', icon: lm[1], x: r1(r.left - ox), y: r1(r.top - oy), width: r1(r.width), height: r1(r.height), fill: hex(cs.stroke !== 'none' ? cs.stroke : cs.color) || hex(cs.color) }
    }
    const frame = { type: 'frame', id: nid('s'), name: svg.getAttribute('aria-label') || 'svg', x: r1(r.left - ox), y: r1(r.top - oy), width: r1(r.width), height: r1(r.height), layout: 'none', children: [] }
    const sm = svg.getScreenCTM()
    const paint = (el, n) => {
      const cs = getComputedStyle(el)
      const op = (+cs.opacity || 1)
      const f = cs.fill && cs.fill !== 'none' && !/url\(/.test(cs.fill) ? withAlpha(hex(cs.fill), op * (+cs.fillOpacity || 1)) : null
      let fUrl = null
      if (/url\(/.test(cs.fill)) { const id = cs.fill.match(/#([^")]+)/)?.[1]; const st = id && svg.querySelector(`#${CSS.escape(id)} stop`); fUrl = st ? hex(getComputedStyle(st).stopColor) : null }
      const s = cs.stroke && cs.stroke !== 'none' && !/url\(/.test(cs.stroke) ? withAlpha(hex(cs.stroke), op * (+cs.strokeOpacity || 1)) : null
      n.fill = f || fUrl || '#00000000'
      if (s) {
        const ctm = el.getScreenCTM(); const sc = ctm ? Math.sqrt(Math.abs(ctm.a * ctm.d - ctm.b * ctm.c)) : 1
        n.stroke = s; n.strokeWidth = r1((parseFloat(cs.strokeWidth) || 1) * sc)
        if (cs.strokeLinecap !== 'butt') n.strokeLinecap = cs.strokeLinecap
        if (cs.strokeLinejoin !== 'miter') n.strokeLinejoin = cs.strokeLinejoin
      }
      return n
    }
    const walkSvg = (el) => {
      for (const c of el.children) {
        const tag = c.tagName.toLowerCase()
        if (['defs', 'clippath', 'mask', 'lineargradient', 'radialgradient', 'marker', 'title', 'desc', 'style', 'pattern', 'filter'].includes(tag)) continue
        if (!vis(c)) continue
        if (tag === 'g' || tag === 'a' || tag === 'svg') { walkSvg(c); continue }
        if (tag === 'foreignobject') { walk(c, frame, r.left, r.top); continue }
        if (tag === 'text') {
          for (const t of c.querySelectorAll('*')) void t
          const tw = document.createTreeWalker(c, NodeFilter.SHOW_TEXT)
          let tn; while ((tn = tw.nextNode())) {
            const rg = document.createRange(); rg.selectNodeContents(tn); const rc = rg.getBoundingClientRect()
            if (!rc.width || !tn.data.trim()) continue
            const cs = getComputedStyle(tn.parentElement)
            const fs = parseFloat(cs.fontSize)
            frame.children.push({ type: 'text', id: nid('t'), name: tn.data.trim().slice(0, 30), content: tn.data.trim(), x: r1(rc.left - r.left), y: r1(rc.top - r.top), fontFamily: family(cs.fontFamily), fontSize: r1(fs * (sm ? Math.abs(sm.a) : 1) || fs), fontWeight: String(cs.fontWeight), lineHeight: +(rc.height / (fs || 1)).toFixed(2) > 0.5 ? +(rc.height / fs).toFixed(2) : 1.2, fill: hex(cs.fill) || hex(cs.color) || '#000000' })
          }
          continue
        }
        let bb; try { bb = c.getBBox() } catch { continue }
        const sr = c.getBoundingClientRect()
        if (!sr.width && !sr.height) continue
        const base = { id: nid('v'), name: tag, x: r1(sr.left - r.left), y: r1(sr.top - r.top), width: r1(Math.max(sr.width, 0.5)), height: r1(Math.max(sr.height, 0.5)) }
        if (tag === 'circle' || tag === 'ellipse') { frame.children.push(paint(c, { type: 'ellipse', ...base })); continue }
        if (tag === 'rect') {
          const n = paint(c, { type: 'rectangle', ...base }); const rx = parseFloat(c.getAttribute('rx') || c.getAttribute('ry') || 0)
          const ctm = c.getScreenCTM(); if (rx) n.cornerRadius = r1(rx * (ctm ? Math.abs(ctm.a) : 1))
          frame.children.push(n); continue
        }
        let d = null
        if (tag === 'path') d = c.getAttribute('d')
        else if (tag === 'line') d = `M${c.getAttribute('x1') || 0} ${c.getAttribute('y1') || 0}L${c.getAttribute('x2') || 0} ${c.getAttribute('y2') || 0}`
        else if (tag === 'polyline' || tag === 'polygon') { const pts = (c.getAttribute('points') || '').trim().split(/[\s,]+/).map(Number); let s = ''; for (let i = 0; i + 1 < pts.length; i += 2) s += (i ? 'L' : 'M') + pts[i] + ' ' + pts[i + 1]; d = s + (tag === 'polygon' ? 'Z' : '') }
        if (!d) continue
        // dashed axis-aligned lines: Pencil has no dash pattern, so emit one rectangle per dash
        const dcs = getComputedStyle(c)
        if (dcs.strokeDasharray && dcs.strokeDasharray !== 'none' && (sr.width < 3 || sr.height < 3) && /^M[^MLHVCSQTAZ]*[LHV][^MLHVCSQTAZ]*$/i.test(d.trim())) {
          const ctm0 = c.getScreenCTM(); const sc = ctm0 ? Math.sqrt(Math.abs(ctm0.a * ctm0.d)) : 1
          const pat = dcs.strokeDasharray.split(/[\s,]+/).map((v) => parseFloat(v) * sc).filter((v) => v > 0)
          const sw = (parseFloat(dcs.strokeWidth) || 1) * sc
          const col = hex(dcs.stroke)
          if (pat.length && col) {
            const horiz = sr.width >= sr.height
            const len = horiz ? sr.width : sr.height
            const full = pat.length % 2 ? [...pat, ...pat] : pat
            let pos = 0, k = 0, n = 0
            const cx0 = sr.left + sr.width / 2 - r.left, cy0 = sr.top + sr.height / 2 - r.top
            while (pos < len && n < 400) {
              const seg = Math.min(full[k % full.length], len - pos)
              if (k % 2 === 0 && seg > 0.2) {
                frame.children.push(horiz
                  ? { type: 'rectangle', id: nid('v'), name: 'dash', x: r1(sr.left - r.left + pos), y: r1(cy0 - sw / 2), width: r1(seg), height: r1(sw), fill: withAlpha(col, +dcs.strokeOpacity || 1) }
                  : { type: 'rectangle', id: nid('v'), name: 'dash', x: r1(cx0 - sw / 2), y: r1(sr.top - r.top + pos), width: r1(sw), height: r1(seg), fill: withAlpha(col, +dcs.strokeOpacity || 1) })
                n++
              }
              pos += full[k % full.length]; k++
            }
            continue
          }
        }
        const ctm = c.getScreenCTM(); const sx = ctm ? Math.abs(ctm.a) || 1 : 1, sy = ctm ? Math.abs(ctm.d) || 1 : 1
        // geometry box in screen px (without stroke)
        const gx = ctm ? ctm.a * bb.x + ctm.e : sr.left, gy = ctm ? ctm.d * bb.y + ctm.f : sr.top
        const vw = Math.max(bb.width, 0.5 / sx), vh = Math.max(bb.height, 0.5 / sy)
        frame.children.push(paint(c, { type: 'path', id: base.id, name: tag, geometry: d, viewBox: [r1(bb.x), r1(bb.y), +vw.toFixed(3), +vh.toFixed(3)], x: r1(gx - r.left), y: r1(gy - r.top), width: r1(vw * sx), height: r1(vh * sy) }))
      }
    }
    walkSvg(svg)
    if (!frame.children.length) return null
    return frame
  }

  const raster = (el, ox, oy, name) => {
    const r = el.getBoundingClientRect()
    if (r.width < 1 || r.height < 1) return null
    const id = nid('r')
    rasters.push({ id, x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height })
    const cs = getComputedStyle(el)
    const n = { type: 'rectangle', id, name, x: r1(r.left - ox), y: r1(r.top - oy), width: r1(r.width), height: r1(r.height), fill: { type: 'image', url: `__ASSET__/${id}.png`, mode: 'stretch' } }
    const rad = parseFloat(cs.borderTopLeftRadius); if (rad) n.cornerRadius = r1(Math.min(rad, r.width / 2, r.height / 2))
    return n
  }

  // --- main walk: `parent` is the pencil frame children attach to; ox/oy its viewport origin
  const walk = (el, parent, ox, oy) => {
    for (const c of el.childNodes) {
      if (c.nodeType === 3) { if (c.parentElement && vis(c.parentElement)) parent.children.push(...textNodesFor(c, c.parentElement, ox, oy)); continue }
      if (c.nodeType !== 1) continue
      if (!vis(c)) continue
      const cs = getComputedStyle(c)
      if (cs.display === 'none') continue
      const r = c.getBoundingClientRect()
      const tag = c.tagName
      if (r.width < 0.5 && r.height < 0.5 && cs.overflow !== 'visible') continue
      if (r.width <= 1 && r.height <= 1 && cs.position === 'absolute') continue // sr-only
      if (c.matches(RASTER_SEL)) { const n = raster(c, ox, oy, tag === 'SPAN' ? 'math' : tag.toLowerCase()); if (n) parent.children.push(n); continue }
      if (tag === 'svg' || tag === 'SVG' || c instanceof SVGSVGElement) { const n = svgNode(c, ox, oy); if (n) parent.children.push(n); continue }
      if (tag === 'INPUT' || tag === 'TEXTAREA') {
        const f = { type: 'frame', id: nid('f'), name: tag.toLowerCase(), x: r1(r.left - ox), y: r1(r.top - oy), width: r1(r.width), height: r1(r.height), layout: 'none', ...boxProps(c, cs), children: [] }
        const val = c.value || c.placeholder
        if (val) {
          const pcs = c.value ? cs : getComputedStyle(c, '::placeholder')
          const fs = parseFloat(cs.fontSize)
          f.children.push({ type: 'text', id: nid('t'), name: 'value', content: val, x: parseFloat(cs.paddingLeft) + parseFloat(cs.borderLeftWidth), y: r1((r.height - fs * 1.4) / 2), fontFamily: family(cs.fontFamily), fontSize: r1(fs), fontWeight: String(cs.fontWeight), lineHeight: 1.4, fill: hex(pcs.color) || '#94A3B8' })
        }
        parent.children.push(f); continue
      }
      // list markers
      if (cs.display === 'list-item' && cs.listStyleType !== 'none' && (cs.listStylePosition === 'outside')) {
        const fs = parseFloat(cs.fontSize); const lh = cs.lineHeight === 'normal' ? fs * 1.2 : parseFloat(cs.lineHeight)
        let mark = '•'
        if (cs.listStyleType === 'decimal') mark = [...c.parentElement.children].filter((x) => x.tagName === 'LI').indexOf(c) + 1 + '.'
        parent.children.push({ type: 'text', id: nid('t'), name: 'marker', content: String(mark), x: r1(r.left - ox - fs * (cs.listStyleType === 'decimal' ? 1.4 : 1)), y: r1(r.top - oy + parseFloat(cs.paddingTop)), fontFamily: SANS, fontSize: r1(fs), fontWeight: '400', lineHeight: +(lh / fs).toFixed(3), fill: hex(cs.color) || '#475569' })
      }
      const bp = boxProps(c, cs)
      const decorated = bp.fill || bp.stroke || bp.effect || bp.opacity !== undefined
      if (decorated || SEMANTIC.has(tag)) {
        const f = { type: 'frame', id: nid('f'), name: label(c), x: r1(r.left - ox), y: r1(r.top - oy), width: r1(r.width), height: r1(r.height), layout: 'none', ...bp, children: [] }
        f.__tag = tag
        walk(c, f, r.left, r.top)
        if (!f.children.length && !decorated) continue
        parent.children.push(f)
      } else {
        walk(c, parent, ox, oy)
      }
    }
  }

  const body = document.body
  const bcs = getComputedStyle(body)
  const hcs = getComputedStyle(document.documentElement)
  const W = document.documentElement.clientWidth, H = document.documentElement.scrollHeight
  scrollTo(0, 0)
  const root = { type: 'frame', id: nid('p'), name: document.title, x: 0, y: 0, width: W, height: H, layout: 'none', clip: true, fill: hex(bcs.backgroundColor) || hex(hcs.backgroundColor) || '#FFFFFF', children: [] }
  walk(body, root, 0, 0)

  // palette: Tailwind colour variables → hex
  const palette = {}
  for (const sheet of document.styleSheets) {
    let rules; try { rules = sheet.cssRules } catch { continue }
    for (const rule of rules) {
      const scan = (rl) => { if (rl.style) for (const prop of rl.style) if (prop.startsWith('--color-')) { const v = hcs.getPropertyValue(prop).trim(); const h = v && hex(v); if (h) palette[prop.slice(8)] = h } ; if (rl.cssRules) for (const x of rl.cssRules) scan(x) }
      scan(rule)
    }
  }
  return { tree: root, rasters, palette }
})()
