import type { ReactElement, ReactNode } from 'react'
import type { ModuleId } from '../lib/sitemap'

// The picture on each Visualize card: a small drawing of what the lab is about, readable at a glance.
// Every drawing shares a 320 × 200 canvas (the card's 8:5), one palette, and symbols instead of words,
// so it reads the same in every language. No <defs>/ids: the grid and chapter views both mount a copy.

const C = {
  blue: '#2563eb', blueM: '#93c5fd', blueL: '#dbeafe',
  rose: '#e11d48', roseM: '#fda4af', roseL: '#ffe4e6',
  amber: '#d97706', amberM: '#fbbf24', amberL: '#fef3c7',
  green: '#059669', greenM: '#34d399', greenL: '#d1fae5',
  violet: '#7c3aed', violetM: '#c4b5fd', violetL: '#ede9fe',
  ink: '#0f172a', text: '#334155', muted: '#64748b', soft: '#94a3b8', line: '#cbd5e1', faint: '#e2e8f0', paper: '#ffffff',
}
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

type TxtProps = { x: number; y: number; s?: number; c?: string; a?: 'start' | 'middle' | 'end'; m?: boolean; w?: number; i?: boolean; o?: number; children: ReactNode }
const Txt = ({ x, y, s = 11, c = C.text, a = 'middle', m = false, w, i, o, children }: TxtProps) => (
  <text x={x} y={y} fontSize={s} fill={c} textAnchor={a} dominantBaseline="central" fontFamily={m ? MONO : undefined} fontWeight={w} fontStyle={i ? 'italic' : undefined} opacity={o}>
    {children}
  </text>
)

/** A straight (or, with q, curved) arrow whose head is drawn by hand, so it needs no marker id. */
function Arrow({ x1, y1, x2, y2, c = C.muted, w = 2, dash, q }: { x1: number; y1: number; x2: number; y2: number; c?: string; w?: number; dash?: string; q?: [number, number] }) {
  const [fx, fy] = q ?? [x1, y1]
  const a = Math.atan2(y2 - fy, x2 - fx)
  const h = 5 + w * 1.6
  const bx = x2 - h * Math.cos(a)
  const by = y2 - h * Math.sin(a)
  const side = (k: number) => `${(bx + k * h * 0.6 * Math.sin(a)).toFixed(1)},${(by - k * h * 0.6 * Math.cos(a)).toFixed(1)}`
  const d = q ? `M${x1} ${y1}Q${q[0]} ${q[1]} ${bx.toFixed(1)} ${by.toFixed(1)}` : `M${x1} ${y1}L${bx.toFixed(1)} ${by.toFixed(1)}`
  return (
    <g>
      <path d={d} fill="none" stroke={c} strokeWidth={w} strokeDasharray={dash} strokeLinecap="round" />
      <polygon points={`${x2},${y2} ${side(1)} ${side(-1)}`} fill={c} />
    </g>
  )
}

type Cell = { f?: string; st?: string; t?: ReactNode; tc?: string; dash?: string; hi?: boolean; o?: number } | null
/** A grid of square cells; highlighted (hi) cells are drawn last so their outline sits on top. */
function Cells({ x, y, rows, cols, s, cell, ts = 10, gap = 0, rx = 0 }: { x: number; y: number; rows: number; cols: number; s: number; cell: (r: number, c: number) => Cell; ts?: number; gap?: number; rx?: number }) {
  const back: ReactNode[] = []
  const front: ReactNode[] = []
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const k = cell(r, c)
      if (!k) continue
      const cx = x + c * (s + gap)
      const cy = y + r * (s + gap)
      const to = k.hi ? front : back
      to.push(<rect key={`r${r}-${c}`} x={cx} y={cy} width={s} height={s} rx={rx} fill={k.f ?? C.paper} stroke={k.st ?? (gap ? 'none' : C.line)} strokeWidth={k.hi ? 2 : 1} strokeDasharray={k.dash} opacity={k.o} />)
      if (k.t !== undefined)
        to.push(
          <Txt key={`t${r}-${c}`} x={cx + s / 2} y={cy + s / 2 + 0.5} s={ts} c={k.tc ?? C.text} m o={k.o}>
            {k.t}
          </Txt>,
        )
    }
  return (
    <g>
      {back}
      {front}
    </g>
  )
}

const Dot = ({ x, y, c, r = 6 }: { x: number; y: number; c: string; r?: number }) => <circle cx={x} cy={y} r={r} fill={c} stroke={C.paper} strokeWidth={1.5} />
const Cross = ({ x, y, c, s = 8, dash }: { x: number; y: number; c: string; s?: number; dash?: boolean }) => {
  const d = `M${x - s} ${y - s}L${x + s} ${y + s}M${x + s} ${y - s}L${x - s} ${y + s}`
  return dash ? (
    <path d={d} stroke={c} strokeWidth={2} strokeLinecap="round" opacity={0.45} />
  ) : (
    <g strokeLinecap="round">
      <path d={d} stroke={C.paper} strokeWidth={7} />
      <path d={d} stroke={c} strokeWidth={3.5} />
    </g>
  )
}
const bell = (mu: number, sd: number, h: number, base: number, x0: number, x1: number) => {
  let d = ''
  for (let i = 0; i <= 64; i++) {
    const x = x0 + ((x1 - x0) * i) / 64
    d += `${i ? 'L' : 'M'}${x.toFixed(1)} ${(base - h * Math.exp(-((x - mu) ** 2) / (2 * sd * sd))).toFixed(1)}`
  }
  return d
}

const FLAME = 'M0 -16C4 -8 11 -3 11 5A11 11 0 0 1 -11 5C-11 -3 -4 -8 0 -16Z'
const FLAME_IN = 'M0 -1C2 3 5 5 5 8A5 5 0 0 1 -5 8C-5 5 -2 3 0 -1Z'
const Flame = ({ x, y, k = 1 }: { x: number; y: number; k?: number }) => (
  <g transform={`translate(${x} ${y}) scale(${k})`}>
    <path d={FLAME} fill="#f97316" />
    <path d={FLAME_IN} fill="#fde047" />
  </g>
)
const Cloud = ({ x, y, k = 1 }: { x: number; y: number; k?: number }) => (
  <g transform={`translate(${x} ${y}) scale(${k})`} fill="#94a3b8">
    <circle cx={-10} cy={3} r={7} />
    <circle cx={0} cy={-3} r={9.5} />
    <circle cx={10} cy={3} r={7} />
    <rect x={-10} y={3} width={20} height={7} />
  </g>
)

/** A box seen slightly from the side: w = thickness (channels), h = height and depth (spatial size). */
function Cuboid({ x, yc, w, h, d = h, col }: { x: number; yc: number; w: number; h: number; d?: number; col: [string, string, string] }) {
  const dx = d * 0.42
  const dy = -d * 0.3
  const y0 = yc - h / 2 - dy / 2
  const y1 = y0 + h
  const p = (pts: number[][]) => pts.map((q) => q.map((v) => v.toFixed(1)).join(',')).join(' ')
  return (
    <g stroke="#475569" strokeWidth={1} strokeLinejoin="round">
      <polygon points={p([[x, y0], [x + w, y0], [x + w + dx, y0 + dy], [x + dx, y0 + dy]])} fill={col[1]} />
      <polygon points={p([[x + w, y0], [x + w + dx, y0 + dy], [x + w + dx, y1 + dy], [x + w, y1]])} fill={col[2]} />
      <rect x={x} y={y0} width={w} height={h} fill={col[0]} />
    </g>
  )
}

const ART: Record<ModuleId, () => ReactElement> = {
  // a slice picks cells out of the array: which ones, and what comes out
  numpy: () => {
    const hit = (r: number, c: number) => r >= 1 && r < 3 && c % 2 === 0
    return (
      <>
        <Txt x={22} y={28} a="start" s={15} m w={700} c={C.ink}>
          a<tspan fill={C.blue}>[1:3, ::2]</tspan>
        </Txt>
        <Cells x={22} y={48} rows={4} cols={5} s={24} ts={11} cell={(r, c) => (hit(r, c) ? { f: C.blueL, st: C.blue, t: r * 5 + c, tc: C.blue, hi: true } : { t: r * 5 + c, tc: C.soft })} />
        <Arrow x1={154} y1={96} x2={200} y2={96} c={C.blue} w={2.5} />
        <Cells x={212} y={72} rows={2} cols={3} s={24} ts={11} cell={(r, c) => ({ f: C.blue, st: C.blue, t: [5, 7, 9, 10, 12, 14][r * 3 + c], tc: C.paper })} />
        <Txt x={82} y={162} m c={C.muted}>
          (4, 5)
        </Txt>
        <Txt x={248} y={136} m c={C.muted}>
          (2, 3)
        </Txt>
      </>
    )
  },

  // fire inside smoke: the overlap is P(fire | smoke)
  'bayes-basics': () => (
    <>
      <circle cx={184} cy={86} r={62} fill="#f1f5f9" />
      <circle cx={122} cy={86} r={48} fill={C.roseL} />
      <path d="M140.6 41.7A48 48 0 0 1 140.6 130.3A62 62 0 0 1 140.6 41.7Z" fill={C.roseM} />
      <circle cx={184} cy={86} r={62} fill="none" stroke={C.soft} strokeWidth={2} />
      <circle cx={122} cy={86} r={48} fill="none" stroke={C.rose} strokeWidth={2} />
      <Flame x={98} y={84} k={1.25} />
      <Cloud x={214} y={84} k={1.3} />
      <rect x={110} y={162} width={100} height={28} rx={14} fill={C.roseL} stroke={C.roseM} />
      <Txt x={134} y={176} a="end" s={14} m w={700} c={C.ink}>
        P(
      </Txt>
      <Flame x={143} y={176} k={0.62} />
      <Txt x={156} y={176} s={14} m w={700} c={C.ink}>
        |
      </Txt>
      <Cloud x={172} y={175} k={0.62} />
      <Txt x={182} y={176} a="start" s={14} m w={700} c={C.ink}>
        )
      </Txt>
    </>
  ),

  // 200 people: most positive tests (amber) come from the healthy, only a sliver (rose) are sick
  'bayes-virus': () => {
    const sick = new Set([47, 112])
    const falsePos = new Set([5, 18, 22, 30, 41, 58, 66, 73, 81, 90, 99, 104, 118, 125, 133, 141, 150, 157])
    return (
      <>
        <Cells x={22} y={22} rows={8} cols={20} s={11} gap={3} rx={2} cell={(r, c) => ({ f: sick.has(r * 20 + c) ? C.rose : falsePos.has(r * 20 + c) ? C.amberM : C.faint })} />
        <circle cx={34} cy={168} r={11} fill={C.ink} />
        <Txt x={34} y={167.5} s={16} w={700} c={C.paper}>
          +
        </Txt>
        <rect x={54} y={161} width={206} height={14} fill={C.amberM} />
        <rect x={54} y={161} width={21} height={14} fill={C.rose} />
        <Txt x={270} y={168} a="start" s={15} w={800} c={C.rose}>
          10%
        </Txt>
      </>
    )
  },

  // count a table of yes/no features, multiply, compare the two classes
  'naive-bayes': () => {
    const rows = [
      [1, 0, 1, 1],
      [1, 1, 0, 0],
      [0, 1, 1, 0],
      [1, 1, 1, 1],
    ]
    return (
      <>
        <rect x={16} y={26} width={138} height={124} rx={8} fill={C.paper} stroke={C.line} />
        <line x1={16} x2={154} y1={52} y2={52} stroke={C.line} />
        <line x1={118} x2={118} y1={26} y2={150} stroke={C.line} />
        {['x₁', 'x₂', 'x₃', 'y'].map((h, i) => (
          <Txt key={h} x={[38, 66, 94, 136][i]} y={39} m c={C.muted}>
            {h}
          </Txt>
        ))}
        {rows.map((row, r) =>
          row.map((v, c) =>
            c < 3 ? (
              <circle key={`${r}-${c}`} cx={[38, 66, 94][c]} cy={68 + r * 24} r={5.5} fill={v ? C.text : C.paper} stroke={v ? C.text : C.soft} strokeWidth={1.5} />
            ) : (
              <Dot key={`${r}-${c}`} x={136} y={68 + r * 24} c={v ? C.blue : C.rose} r={7} />
            ),
          ),
        )}
        <Arrow x1={162} y1={92} x2={180} y2={92} />
        <Txt x={244} y={38} m s={12} w={600} c={C.ink}>
          P(y)·∏ P(x|y)
        </Txt>
        {[
          [C.blue, 0.2, '20%'],
          [C.rose, 0.8, '80%'],
        ].map(([col, v, label], i) => (
          <g key={i}>
            <Dot x={194} y={76 + i * 32} c={col as string} r={7} />
            <rect x={206} y={70 + i * 32} width={66} height={12} rx={6} fill={C.faint} />
            <rect x={206} y={70 + i * 32} width={66 * (v as number)} height={12} rx={6} fill={col as string} />
            <Txt x={306} y={76 + i * 32} a="end" m s={11} w={700} c={col as string}>
              {label}
            </Txt>
          </g>
        ))}
        <Txt x={244} y={144} m s={11} c={C.muted}>
          (n+1) / (N+k)
        </Txt>
      </>
    )
  },

  // one bell curve per class; the likelihood of x is how tall each curve is above it
  'gaussian-nb': () => (
    <>
      <path d={`${bell(118, 30, 108, 160, 18, 302)}L302 160L18 160Z`} fill={C.blue} opacity={0.1} />
      <path d={`${bell(208, 34, 86, 160, 18, 302)}L302 160L18 160Z`} fill={C.rose} opacity={0.1} />
      <path d={bell(118, 30, 108, 160, 18, 302)} fill="none" stroke={C.blue} strokeWidth={2.5} />
      <path d={bell(208, 34, 86, 160, 18, 302)} fill="none" stroke={C.rose} strokeWidth={2.5} />
      <line x1={14} x2={306} y1={160} y2={160} stroke={C.soft} strokeWidth={2} />
      <line x1={166} x2={166} y1={34} y2={160} stroke={C.muted} strokeDasharray="4 4" />
      <line x1={160} x2={160} y1={160} y2={130} stroke={C.blue} strokeWidth={6} />
      <line x1={172} x2={172} y1={160} y2={120} stroke={C.rose} strokeWidth={6} />
      <Dot x={166} y={130} c={C.blue} r={5.5} />
      <Dot x={166} y={120} c={C.rose} r={5.5} />
      <Txt x={166} y={176} m s={15} i w={700} c={C.ink}>
        x
      </Txt>
      <Txt x={304} y={24} a="end" m s={12} c={C.muted}>
        p(x | y)
      </Txt>
    </>
  ),

  // the 3 nearest points vote on the new one
  knn: () => {
    const blue = [[52, 58], [78, 38], [96, 76], [62, 98], [114, 46], [140, 78], [138, 120], [84, 132], [106, 104]]
    const rose = [[178, 124], [206, 70], [234, 112], [252, 62], [216, 154], [264, 140], [238, 36], [284, 96], [196, 172]]
    const near = [[140, 78], [138, 120], [178, 124]]
    return (
      <>
        <circle cx={160} cy={100} r={40} fill={C.blueL} fillOpacity={0.35} stroke={C.muted} strokeWidth={1.5} strokeDasharray="5 4" />
        {near.map(([x, y]) => (
          <line key={`${x}`} x1={160} y1={100} x2={x} y2={y} stroke={C.muted} strokeWidth={1.5} />
        ))}
        {blue.map(([x, y]) => (
          <Dot key={`b${x}`} x={x} y={y} c={C.blue} />
        ))}
        {rose.map(([x, y]) => (
          <Dot key={`r${x}`} x={x} y={y} c={C.rose} />
        ))}
        <circle cx={160} cy={100} r={10} fill={C.paper} stroke={C.ink} strokeWidth={2.5} />
        <Txt x={160} y={100.5} s={12} w={800} c={C.ink}>
          ?
        </Txt>
        <Txt x={18} y={178} a="start" m s={15} w={700} c={C.ink}>
          K = 3
        </Txt>
      </>
    )
  },

  // a 2 × 2 confusion matrix and the scores read off it
  evaluation: () => {
    const cells = [
      ['TP', 42, '#10b981', C.paper],
      ['FN', 8, C.roseL, C.rose],
      ['FP', 5, C.amberL, C.amber],
      ['TN', 45, C.greenL, C.green],
    ] as const
    return (
      <>
        <Txt x={98} y={20} m s={12} c={C.muted}>
          ŷ
        </Txt>
        <Txt x={20} y={96} m s={12} c={C.muted}>
          y
        </Txt>
        {['+', '−'].map((h, i) => (
          <g key={h}>
            <Txt x={73 + i * 50} y={36} m w={700} c={C.muted}>
              {h}
            </Txt>
            <Txt x={38} y={71 + i * 50} m w={700} c={C.muted}>
              {h}
            </Txt>
          </g>
        ))}
        {cells.map(([label, n, f, tc], i) => {
          const x = 48 + (i % 2) * 50
          const y = 46 + Math.floor(i / 2) * 50
          return (
            <g key={label}>
              <rect x={x} y={y} width={50} height={50} fill={f} stroke={C.paper} strokeWidth={2} />
              <Txt x={x + 6} y={y + 10} a="start" s={9} w={700} c={tc} o={0.85}>
                {label}
              </Txt>
              <Txt x={x + 25} y={y + 29} s={18} w={800} c={tc}>
                {n}
              </Txt>
            </g>
          )
        })}
        {[
          ['P', 0.89, C.amber],
          ['R', 0.84, C.rose],
          ['F1', 0.86, C.green],
        ].map(([label, v, col], i) => (
          <g key={label as string}>
            <Txt x={176} y={64 + i * 34} a="start" m s={13} w={800} c={col as string}>
              {label}
            </Txt>
            <rect x={204} y={58 + i * 34} width={66} height={12} rx={6} fill={C.faint} />
            <rect x={204} y={58 + i * 34} width={66 * (v as number)} height={12} rx={6} fill={col as string} />
            <Txt x={306} y={64 + i * 34} a="end" m s={11} w={700} c={C.text}>
              {(v as number).toFixed(2).slice(1)}
            </Txt>
          </g>
        ))}
      </>
    )
  },

  // sorted data, no shuffle: each test fold is a class the model never trained on
  'cross-validation': () => {
    const cols = [C.blue, C.green, C.amberM]
    return (
      <>
        <Txt x={282} y={26} m s={11} c={C.muted}>
          acc
        </Txt>
        {[0, 1, 2].map((i) => (
          <g key={i}>
            <Txt x={20} y={62 + i * 44} m s={12} w={700} c={C.muted}>
              {i + 1}
            </Txt>
            <Cells x={34} y={50 + i * 44} rows={1} cols={15} s={12} gap={2} rx={2} cell={(_, c) => ({ f: cols[Math.floor(c / 5)], o: Math.floor(c / 5) === i ? 1 : 0.22 })} />
            <rect x={30 + i * 70} y={46 + i * 44} width={74} height={20} rx={5} fill="none" stroke={C.ink} strokeWidth={2} />
            <Txt x={282} y={57 + i * 44} s={16} w={800} c={C.rose}>
              0%
            </Txt>
          </g>
        ))}
      </>
    )
  },

  // points take the colour of the nearest centre, centres move to the middle; inset: the elbow curve
  kmeans: () => {
    const offs = [[-20, -8], [-8, -18], [10, -12], [-22, 10], [2, 2], [18, 8], [-6, 18], [22, -4], [8, 20]]
    const groups = [
      { c: C.blue, at: [78, 70], old: [120, 98], f: (o: number[]) => o },
      { c: C.rose, at: [100, 152], old: [140, 124], f: (o: number[]) => [o[1], o[0] * 0.8] },
      { c: C.green, at: [204, 128], old: [172, 92], f: (o: number[]) => [-o[0], -o[1]] },
    ]
    return (
      <>
        {groups.map((g) =>
          offs.map((o, i) => {
            const [dx, dy] = g.f(o)
            return <Dot key={`${g.c}${i}`} x={g.at[0] + dx} y={g.at[1] + dy} c={g.c} r={5} />
          }),
        )}
        {groups.map((g) => (
          <g key={g.c}>
            <Cross x={g.old[0]} y={g.old[1]} c={g.c} s={6} dash />
            <Arrow x1={g.old[0]} y1={g.old[1]} x2={g.at[0] + (g.old[0] - g.at[0]) * 0.28} y2={g.at[1] + (g.old[1] - g.at[1]) * 0.28} c={g.c} w={1.8} dash="4 3" />
            <Cross x={g.at[0]} y={g.at[1]} c={g.c} />
          </g>
        ))}
        <rect x={234} y={14} width={72} height={54} rx={6} fill={C.paper} stroke={C.line} />
        <polyline points="244,22 255,39 266,51 277,54 288,56 298,57" fill="none" stroke={C.muted} strokeWidth={1.5} />
        {[[244, 22], [255, 39], [277, 54], [288, 56], [298, 57]].map(([x, y]) => (
          <circle key={x} cx={x} cy={y} r={2} fill={C.muted} />
        ))}
        <circle cx={266} cy={51} r={4} fill={C.rose} />
      </>
    )
  },

  // the exam table: distance to each centre, the smaller wins, then the centre moves
  'kmeans-table': () => {
    const rows = [
      ['A', '1.4', '5.0', 0],
      ['B', '2.2', '3.6', 0],
      ['C', '4.1', '1.0', 1],
      ['D', '5.8', '0.6', 1],
    ] as const
    const col = [C.blue, C.rose]
    const light = [C.blueL, C.roseL]
    return (
      <>
        <rect x={14} y={28} width={178} height={112} rx={8} fill={C.paper} stroke={C.line} />
        <line x1={14} x2={192} y1={50} y2={50} stroke={C.line} />
        <Txt x={32} y={39} m c={C.muted}>
          pt
        </Txt>
        <Txt x={75} y={39} m w={700} c={C.blue}>
          d₁
        </Txt>
        <Txt x={125} y={39} m w={700} c={C.rose}>
          d₂
        </Txt>
        <Txt x={170} y={39} m c={C.muted}>
          →
        </Txt>
        {rows.map(([p, d1, d2, k], r) => {
          const y = 62 + r * 22
          return (
            <g key={p}>
              <rect x={k ? 107 : 57} y={y - 8.5} width={36} height={17} rx={4} fill={light[k]} />
              <Txt x={32} y={y} m w={700} c={C.ink}>
                {p}
              </Txt>
              <Txt x={75} y={y} m w={k ? 400 : 700} c={k ? C.soft : C.blue}>
                {d1}
              </Txt>
              <Txt x={125} y={y} m w={k ? 700 : 400} c={k ? C.rose : C.soft}>
                {d2}
              </Txt>
              <circle cx={170} cy={y} r={8} fill={col[k]} />
              <Txt x={170} y={y + 0.5} s={10} w={700} c={C.paper}>
                {k + 1}
              </Txt>
            </g>
          )
        })}
        <Txt x={103} y={162} m s={11} c={C.muted}>
          μ₁ = mean(A, B)
        </Txt>
        <rect x={206} y={28} width={100} height={112} rx={8} fill={C.paper} stroke={C.line} />
        {[
          ['A', 222, 118, 0],
          ['B', 244, 102, 0],
          ['C', 272, 66, 1],
          ['D', 292, 48, 1],
        ].map(([p, x, y, k]) => (
          <g key={p as string}>
            <Dot x={x as number} y={y as number} c={col[k as number]} r={5} />
            <Txt x={(x as number) - 9} y={(y as number) + 1} a="end" m s={9} c={C.muted}>
              {p}
            </Txt>
          </g>
        ))}
        <Cross x={252} y={126} c={C.blue} s={5} dash />
        <Arrow x1={252} y1={126} x2={237} y2={113} c={C.blue} w={1.5} dash="3 3" />
        <Cross x={233} y={110} c={C.blue} s={6} />
        <Cross x={262} y={92} c={C.rose} s={5} dash />
        <Arrow x1={262} y1={92} x2={279} y2={63} c={C.rose} w={1.5} dash="3 3" />
        <Cross x={282} y={57} c={C.rose} s={6} />
      </>
    )
  },

  // the line moves after each mistake until it separates the two classes; the table holds w and θ
  perceptron: () => (
    <>
      <rect x={16} y={22} width={160} height={156} rx={6} fill={C.paper} stroke={C.line} />
      <polygon points="17,23 175,23 175,177" fill={C.blue} opacity={0.07} />
      <polygon points="17,23 17,177 175,177" fill={C.rose} opacity={0.07} />
      <line x1={16} y1={92} x2={176} y2={124} stroke={C.soft} strokeWidth={1.5} strokeDasharray="4 4" />
      <line x1={62} y1={22} x2={128} y2={178} stroke={C.soft} strokeWidth={1.5} strokeDasharray="4 4" />
      <line x1={16} y1={22} x2={176} y2={178} stroke={C.ink} strokeWidth={2.5} />
      <Arrow x1={96} y1={100} x2={122} y2={73} c={C.ink} w={2} />
      <Txt x={130} y={66} m s={13} i w={700} c={C.ink}>
        w
      </Txt>
      {[[118, 44], [152, 86], [100, 66], [158, 42], [146, 130]].map(([x, y]) => (
        <Dot key={`b${x}`} x={x} y={y} c={C.blue} />
      ))}
      {[[44, 142], [72, 120], [38, 100], [90, 162], [56, 166]].map(([x, y]) => (
        <Dot key={`r${x}`} x={x} y={y} c={C.rose} />
      ))}
      <rect x={190} y={22} width={116} height={118} rx={8} fill={C.paper} stroke={C.line} />
      <line x1={190} x2={306} y1={44} y2={44} stroke={C.line} />
      {['w₁', 'w₂', 'θ'].map((h, i) => (
        <Txt key={h} x={216 + i * 34} y={33} m c={C.muted}>
          {h}
        </Txt>
      ))}
      <rect x={194} y={108} width={108} height={22} rx={5} fill={C.blueL} />
      {[
        ['0', '0', '0'],
        ['1', '1', '1'],
        ['1', '−1', '0'],
        ['2', '−1', '0'],
      ].map((row, r) =>
        row.map((v, i) => (
          <Txt key={`${r}${i}`} x={216 + i * 34} y={56 + r * 20 + (r === 3 ? 3 : 0)} m s={12} w={r === 3 ? 700 : 400} c={r === 3 ? C.blue : C.muted}>
            {v}
          </Txt>
        )),
      )}
      <Txt x={248} y={162} m s={12} w={600} c={C.ink}>
        w·x − θ ≥ 0
      </Txt>
    </>
  ),

  // forward pass left to right (blue), gradients flow back right to left (rose), multiplied by the chain rule
  backprop: () => (
    <>
      <circle cx={40} cy={78} r={18} fill={C.paper} stroke={C.soft} strokeWidth={2} />
      <Txt x={40} y={78} m s={14} i w={700} c={C.ink}>
        x
      </Txt>
      <circle cx={154} cy={78} r={30} fill={C.paper} stroke={C.blue} strokeWidth={2.5} />
      <path d="M134 90C150 90 158 66 174 66" fill="none" stroke={C.blue} strokeWidth={2.5} />
      <rect x={250} y={60} width={44} height={36} rx={8} fill={C.roseL} stroke={C.rose} strokeWidth={2} />
      <Txt x={272} y={78} s={16} i w={800} c={C.rose}>
        E
      </Txt>
      <Arrow x1={60} y1={78} x2={120} y2={78} c={C.blue} w={2.5} />
      <Arrow x1={186} y1={78} x2={246} y2={78} c={C.blue} w={2.5} />
      <Txt x={90} y={64} m s={13} i w={700} c={C.ink}>
        w
      </Txt>
      <Txt x={216} y={64} m s={13} i w={700} c={C.ink}>
        o
      </Txt>
      <Arrow x1={262} y1={100} q={[222, 140]} x2={180} y2={102} c={C.rose} w={2} dash="5 4" />
      <Arrow x1={130} y1={102} q={[96, 136]} x2={88} y2={88} c={C.rose} w={2} dash="5 4" />
      <Txt x={226} y={132} m s={11} w={700} c={C.rose}>
        ∂E/∂o
      </Txt>
      <Txt x={64} y={126} m s={11} w={700} c={C.rose}>
        ∂E/∂w
      </Txt>
      <rect x={20} y={160} width={280} height={28} rx={8} fill={C.paper} stroke={C.line} />
      <Txt x={160} y={174} m s={11} c={C.text}>
        ∂E/∂w = ∂E/∂o · ∂o/∂net · ∂net/∂w
      </Txt>
    </>
  ),

  // XOR's corners can't be split by one line; two hidden units draw two lines and the band between them works
  'xor-mlp': () => {
    const edges = [
      [184, 70, 240, 70, C.blue],
      [184, 130, 240, 70, C.blue],
      [184, 70, 240, 130, C.blue],
      [184, 130, 240, 130, C.blue],
      [240, 70, 292, 100, C.blue],
      [240, 130, 292, 100, C.rose],
    ] as const
    const nodes = [
      [184, 70, 'x₁', C.soft],
      [184, 130, 'x₂', C.soft],
      [240, 70, 'h₁', C.violet],
      [240, 130, 'h₂', C.amber],
      [292, 100, 'o', C.ink],
    ] as const
    return (
      <>
        <Txt x={80} y={28} m s={13} w={800} c={C.ink}>
          XOR
        </Txt>
        <rect x={24} y={44} width={112} height={112} rx={6} fill={C.paper} stroke={C.line} />
        <polygon points="24,84 96,156 136,156 136,116 64,44 24,44" fill={C.blue} opacity={0.12} />
        <line x1={24} y1={84} x2={96} y2={156} stroke={C.violet} strokeWidth={2.5} />
        <line x1={64} y1={44} x2={136} y2={116} stroke={C.amber} strokeWidth={2.5} />
        <Dot x={40} y={140} c={C.rose} r={8} />
        <Dot x={120} y={60} c={C.rose} r={8} />
        <Dot x={40} y={60} c={C.blue} r={8} />
        <Dot x={120} y={140} c={C.blue} r={8} />
        {edges.map(([x1, y1, x2, y2, col], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={2.5} opacity={0.7} />
        ))}
        {nodes.map(([x, y, label, col]) => (
          <g key={label}>
            <circle cx={x} cy={y} r={14} fill={C.paper} stroke={col} strokeWidth={2.5} />
            <Txt x={x} y={y} m s={11} w={700} c={C.ink}>
              {label}
            </Txt>
          </g>
        ))}
      </>
    )
  },

  // image ∗ Laplacian kernel = only the edges light up
  kernel: () => {
    const img = ['........', '..####..', '.######.', '.######.', '.######.', '.######.', '..####..', '........']
    const on = (r: number, c: number) => img[r]?.[c] === '#'
    const edge = (r: number, c: number) => on(r, c) && !(on(r - 1, c) && on(r + 1, c) && on(r, c - 1) && on(r, c + 1))
    const k = [0, 1, 0, 1, -4, 1, 0, 1, 0]
    return (
      <>
        <Cells x={12} y={56} rows={8} cols={8} s={11} cell={(r, c) => ({ f: on(r, c) ? '#e2e8f0' : '#1e293b', st: '#334155' })} />
        <Txt x={113} y={99} s={22} w={700} c={C.muted}>
          ∗
        </Txt>
        <Cells x={127} y={73} rows={3} cols={3} s={18} ts={11} cell={(r, c) => ({ f: k[r * 3 + c] < 0 ? C.roseL : C.paper, t: String(k[r * 3 + c]).replace('-', '−'), tc: k[r * 3 + c] < 0 ? C.rose : k[r * 3 + c] ? C.ink : C.soft })} />
        <Txt x={201} y={100} s={20} w={700} c={C.muted}>
          =
        </Txt>
        <Cells x={220} y={56} rows={8} cols={8} s={11} cell={(r, c) => ({ f: edge(r, c) ? '#fbbf24' : '#1e293b', st: '#334155' })} />
      </>
    )
  },

  // a 3 × 3 window slides over the zero-padded input; each stop is one output cell
  convolution: () => {
    const v = [3, 1, 2, 0, 1, 4, 0, 2, 2, 0, 1, 3, 0, 2, 3, 1]
    const k = ['1', '0', '−1', '2', '0', '−2', '1', '0', '−1']
    return (
      <>
        <Txt x={304} y={24} a="end" m s={11} c={C.muted}>
          pad = 1 · stride = 1
        </Txt>
        <Cells
          x={16}
          y={48}
          rows={6}
          cols={6}
          s={17}
          ts={10}
          cell={(r, c) => {
            const pad = r === 0 || c === 0 || r === 5 || c === 5
            const win = r < 3 && c < 3
            return pad ? { f: win ? C.blueL : C.paper, t: '0', tc: C.soft, dash: '2 2', st: C.soft } : { f: win ? C.blueL : '#f1f5f9', t: v[(r - 1) * 4 + (c - 1)], tc: C.ink }
          }}
        />
        <rect x={16} y={48} width={51} height={51} fill="none" stroke={C.blue} strokeWidth={2.5} />
        <Txt x={132} y={99} s={22} w={700} c={C.muted}>
          ∗
        </Txt>
        <Txt x={168} y={70} m s={13} c={C.violet}>
          ↻
        </Txt>
        <Cells x={147} y={80} rows={3} cols={3} s={14} ts={9} cell={(r, c) => ({ f: C.violetL, st: C.violetM, t: k[r * 3 + c], tc: C.violet })} />
        <Txt x={205} y={100} s={20} w={700} c={C.muted}>
          =
        </Txt>
        <Cells x={220} y={62} rows={4} cols={4} s={20} ts={12} cell={(r, c) => (r + c === 0 ? { f: C.blue, st: C.blue, t: 6, tc: C.paper, hi: true } : { f: C.paper })} />
        <Txt x={260} y={160} m s={10} c={C.muted}>
          (4+2−3)/1+1 = 4
        </Txt>
      </>
    )
  },

  // a two-hump histogram; Otsu's T splits it, and the photo becomes black and white
  otsu: () => {
    const T = 15
    const h = Array.from({ length: 32 }, (_, i) => 66 * Math.exp(-((i - 8) ** 2) / 18) + 92 * Math.exp(-((i - 23) ** 2) / 26) + 5)
    const px = (r: number, c: number) => 52 + 176 * Math.exp(-((r - 3.6) ** 2 + (c - 4.2) ** 2) / 7) + ((r * 7 + c * 3) % 5) * 7
    const g = (v: number) => `rgb(${v},${v},${v})`
    return (
      <>
        {h.map((v, i) => (
          <rect key={i} x={16 + i * 6} y={146 - v} width={5} height={v} fill={i < T ? '#334155' : '#a5b4c8'} />
        ))}
        <line x1={14} x2={210} y1={146} y2={146} stroke={C.soft} strokeWidth={1.5} />
        {h.map((_, i) => (
          <rect key={`g${i}`} x={16 + i * 6} y={152} width={6} height={8} fill={g(Math.round((i / 31) * 255))} />
        ))}
        <line x1={16 + T * 6 - 0.5} x2={16 + T * 6 - 0.5} y1={34} y2={162} stroke={C.rose} strokeWidth={2.5} strokeDasharray="5 3" />
        <Txt x={16 + T * 6 - 0.5} y={24} m s={14} w={800} c={C.rose}>
          T
        </Txt>
        <Cells x={232} y={18} rows={8} cols={8} s={9} cell={(r, c) => ({ f: g(Math.round(Math.min(255, px(r, c)))), st: 'none' })} />
        <Txt x={268} y={100} s={14} c={C.muted}>
          ↓
        </Txt>
        <Cells x={232} y={110} rows={8} cols={8} s={9} cell={(r, c) => ({ f: px(r, c) > 120 ? '#ffffff' : '#0f172a', st: 'none' })} />
        <rect x={232} y={110} width={72} height={72} fill="none" stroke={C.line} />
      </>
    )
  },

  // one 2 × 3 matrix moves every pixel: here a quarter turn of an F
  affine: () => {
    const F = ['.......', '.#####.', '.#.....', '.####..', '.#.....', '.#.....', '.......']
    const on = (r: number, c: number) => F[r]?.[c] === '#'
    const nums = [
      ['0', '−1', '6'],
      ['1', '0', '0'],
    ]
    return (
      <>
        <Cells x={16} y={50} rows={7} cols={7} s={13} cell={(r, c) => ({ f: on(r, c) ? C.blue : C.paper, st: C.faint })} />
        <Cells x={213} y={50} rows={7} cols={7} s={13} cell={(r, c) => ({ f: on(6 - c, r) ? C.violet : on(r, c) ? C.blueL : C.paper, st: C.faint })} />
        <path d="M128 70H122V112H128M190 70H196V112H190" fill="none" stroke={C.ink} strokeWidth={2} />
        {nums.map((row, r) =>
          row.map((n, i) => (
            <Txt key={`${r}${i}`} x={[138, 159, 181][i]} y={82 + r * 20} m s={13} w={700} c={i === 2 ? C.amber : C.ink}>
              {n}
            </Txt>
          )),
        )}
        <Arrow x1={124} y1={134} x2={198} y2={134} c={C.violet} w={2.5} />
        <Txt x={160} y={28} m s={12} c={C.muted}>
          [x′ y′] = A·[x y 1]
        </Txt>
      </>
    )
  },

  // feature maps shrink in space and grow in channels, then flatten into a vector
  'cnn-shapes': () => {
    const slate: [string, string, string] = ['#cbd5e1', '#e2e8f0', '#94a3b8']
    const blue: [string, string, string] = ['#93c5fd', '#dbeafe', '#60a5fa']
    const violet: [string, string, string] = ['#c4b5fd', '#ede9fe', '#a78bfa']
    const amber: [string, string, string] = ['#fcd34d', '#fef3c7', '#fbbf24']
    const layers = [
      { x: 14, w: 5, h: 92, col: slate, top: 'in', dim: '32²×3', cx: 38 },
      { x: 76, w: 10, h: 80, col: blue, top: 'conv', dim: '28²×6', cx: 102 },
      { x: 140, w: 10, h: 46, col: violet, top: 'pool', dim: '14²×6', cx: 158 },
      { x: 188, w: 22, h: 32, col: blue, top: 'conv', dim: '10²×16', cx: 206 },
      { x: 244, w: 10, h: 84, d: 14, col: amber, top: 'fc', dim: '120', cx: 252 },
      { x: 282, w: 10, h: 32, d: 14, col: amber, top: 'out', dim: '10', cx: 290 },
    ]
    return (
      <>
        {layers.map((l, i) => (
          <g key={i}>
            <Cuboid x={l.x} yc={98} w={l.w} h={l.h} d={l.d} col={l.col} />
            <Txt x={l.cx} y={22} m s={10} c={C.muted}>
              {l.top}
            </Txt>
            <Txt x={l.cx} y={172} m s={10.5} w={600} c={C.text}>
              {l.dim}
            </Txt>
          </g>
        ))}
      </>
    )
  },

  // the five lines of every training step, going round
  pytorch: () => {
    const steps = [
      ['zero_grad()', C.soft, C.text],
      ['model(x)', C.blue, C.blue],
      ['loss(ŷ, y)', C.blue, C.blue],
      ['backward()', C.rose, C.rose],
      ['step()', C.green, C.green],
    ]
    const at = (deg: number) => [160 + 104 * Math.cos((deg * Math.PI) / 180), 100 + 66 * Math.sin((deg * Math.PI) / 180)]
    return (
      <>
        <ellipse cx={160} cy={100} rx={104} ry={66} fill="none" stroke={C.line} strokeWidth={2} />
        {[-54, 18, 90, 162, 234].map((deg) => {
          const [x, y] = at(deg)
          const t = (deg * Math.PI) / 180
          const a = Math.atan2(66 * Math.cos(t), -104 * Math.sin(t))
          const tip = (k: number, s: number) => `${(x + k * Math.cos(a) - s * Math.sin(a)).toFixed(1)},${(y + k * Math.sin(a) + s * Math.cos(a)).toFixed(1)}`
          return <polygon key={deg} points={`${tip(6, 0)} ${tip(-5, 5)} ${tip(-5, -5)}`} fill={C.soft} />
        })}
        {steps.map(([label, stroke, color], i) => {
          const [x, y] = at(-90 + i * 72)
          const w = label.length * 7 + 18
          return (
            <g key={label}>
              <rect x={x - w / 2} y={y - 12} width={w} height={24} rx={12} fill={C.paper} stroke={stroke} strokeWidth={2} />
              <Txt x={x} y={y} m s={11.5} w={700} c={color}>
                {label}
              </Txt>
            </g>
          )
        })}
        <Cells x={139} y={82} rows={3} cols={3} s={14} cell={(r, c) => ({ f: ['#fde68a', '#fdba74', '#fca5a5', '#fdba74', '#fca5a5', '#f87171', '#fca5a5', '#f87171', '#ef4444'][r * 3 + c], st: C.paper })} />
        <Txt x={160} y={136} m s={10} c={C.muted}>
          .grad
        </Txt>
      </>
    )
  },

  // minimax tree: MAX takes the larger child, MIN the smaller; once α ≥ β the rest of a branch is cut off
  alphabeta: () => {
    const mins = [64, 160, 256]
    const leaves = [
      [3, 12, 8],
      [2, 4, 6],
      [14, 5, 2],
    ]
    const minVal = ['3', '2', '2']
    const up = (x: number, y: number) => `${x},${y - 16} ${x + 18},${y + 13} ${x - 18},${y + 13}`
    const down = (x: number, y: number) => `${x - 18},${y - 13} ${x + 18},${y - 13} ${x},${y + 16}`
    return (
      <>
        {mins.map((mx, i) => (
          <line key={mx} x1={160} y1={44} x2={mx} y2={80} stroke={i === 0 ? C.blue : C.soft} strokeWidth={i === 0 ? 3 : 2} />
        ))}
        {mins.map((mx, i) =>
          leaves[i].map((v, j) => {
            const lx = mx - 32 + j * 32
            const cut = i === 1 && j > 0
            const best = i === 0 && j === 0
            return (
              <g key={`${i}${j}`}>
                <line x1={mx} y1={104} x2={lx} y2={140} stroke={best ? C.blue : cut ? C.line : C.soft} strokeWidth={best ? 3 : 2} strokeDasharray={cut ? '4 4' : undefined} />
                {cut && <path d={`M${(mx + lx) / 2 - 7} ${118}L${(mx + lx) / 2 + 7} ${126}`} stroke={C.rose} strokeWidth={2.5} strokeLinecap="round" />}
                <rect x={lx - 12} y={140} width={24} height={24} rx={4} fill={cut ? '#f8fafc' : C.paper} stroke={cut ? C.line : C.soft} strokeWidth={1.5} strokeDasharray={cut ? '3 3' : undefined} />
                <Txt x={lx} y={152} m s={11} w={700} c={cut ? C.line : best ? C.blue : C.ink}>
                  {v}
                </Txt>
              </g>
            )
          }),
        )}
        <polygon points={up(160, 34)} fill={C.blueL} stroke={C.blue} strokeWidth={2} strokeLinejoin="round" />
        <Txt x={160} y={39} m s={11} w={800} c={C.blue}>
          3
        </Txt>
        {mins.map((mx, i) => (
          <g key={mx}>
            <polygon points={down(mx, 92)} fill={i === 0 ? C.blueL : C.paper} stroke={i === 0 ? C.blue : C.text} strokeWidth={2} strokeLinejoin="round" />
            <Txt x={mx} y={86} m s={11} w={800} c={i === 0 ? C.blue : C.ink}>
              {minVal[i]}
            </Txt>
          </g>
        ))}
        <Txt x={14} y={36} a="start" m s={9} c={C.muted}>
          MAX
        </Txt>
        <Txt x={14} y={92} a="start" m s={9} c={C.muted}>
          MIN
        </Txt>
        <Txt x={196} y={30} a="start" m s={11} w={700} c={C.blue}>
          α = 3
        </Txt>
        <Txt x={176} y={180} m s={11} w={700} c={C.rose}>
          β = 2 ≤ α
        </Txt>
      </>
    )
  },
}

export default function LabArt({ id }: { id: ModuleId }) {
  return (
    <svg viewBox="0 0 320 200" className="h-full w-full" aria-hidden="true" focusable="false">
      <rect width={320} height={200} fill="#f8fafc" />
      {ART[id]()}
    </svg>
  )
}
