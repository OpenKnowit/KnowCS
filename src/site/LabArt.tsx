import type { ReactElement, ReactNode } from 'react'
import type { ModuleId } from '../lib/sitemap'

// The picture on each Visualize card: a small drawing of the lab's own main view (its dark panels, markers and colours),
// simplified so what the lab is about reads at a glance. Every drawing shares a 320 × 200 canvas (the card's 8:5).
// No <defs>/ids: the grid and chapter views both mount a copy.

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

// the dark panels several labs draw on (slate-900 cards, as on the pages themselves)
const D = { bg: '#0f172a', panel: '#1e293b', edge: '#334155', dim: '#64748b', text: '#cbd5e1', bright: '#f1f5f9', blue: '#60a5fa', rose: '#fb7185', amber: '#fbbf24', green: '#6ee7b7', violet: '#c4b5fd', sky: '#7dd3fc', orange: '#fdba74' }
const Panel = ({ x = 10, y = 12, w = 300, h = 176 }: { x?: number; y?: number; w?: number; h?: number }) => <rect x={x} y={y} width={w} height={h} rx={12} fill={D.bg} />
/** a centroid as the labs draw it: a ring with a cross */
const Centroid = ({ x, y, c, faded }: { x: number; y: number; c: string; faded?: boolean }) => (
  <g opacity={faded ? 0.35 : 1} strokeLinecap="round">
    <circle cx={x} cy={y} r={9} fill={C.paper} stroke={c} strokeWidth={2.5} />
    <path d={`M${x - 4.5} ${y - 4.5}L${x + 4.5} ${y + 4.5}M${x + 4.5} ${y - 4.5}L${x - 4.5} ${y + 4.5}`} stroke={c} strokeWidth={2.5} />
  </g>
)
/** a line of syntax-coloured code: [text, colour] pieces */
const Code = ({ x, y, n, parts, s = 11 }: { x: number; y: number; n?: number; parts: [string, string][]; s?: number }) => (
  <g>
    {n !== undefined && (
      <Txt x={x - 12} y={y} a="end" m s={s - 1} c={D.dim}>
        {n}
      </Txt>
    )}
    <text x={x} y={y} fontSize={s} fontFamily={MONO} dominantBaseline="central" xmlSpace="preserve">
      {parts.map(([t, c], i) => (
        <tspan key={i} fill={c}>
          {t}
        </tspan>
      ))}
    </text>
  </g>
)

const ART: Record<ModuleId, () => ReactElement> = {
  // the lab's code editor and the cells the slice picks
  numpy: () => {
    const hit = (r: number, c: number) => r >= 1 && r < 3 && c % 2 === 0
    const P = D.bright
    return (
      <>
        <Panel x={10} y={14} w={186} h={172} />
        <Code x={38} y={40} n={1} parts={[['import', D.violet], [' numpy ', P], ['as', D.violet], [' np', P]]} />
        <Code x={38} y={62} n={2} parts={[['a = np.', P], ['arange', D.sky], ['(', P], ['16', D.orange], [')', P]]} />
        <Code x={38} y={84} n={3} parts={[['a = a.', P], ['reshape', D.sky], ['(', P], ['4', D.orange], [', ', P], ['4', D.orange], [')', P]]} />
        <Code x={38} y={106} n={4} parts={[['a[', P], ['1', D.orange], [':', P], ['3', D.orange], [', ::', P], ['2', D.orange], [']', P]]} />
        <rect x={22} y={156} width={46} height={20} rx={5} fill={C.blue} />
        <Txt x={45} y={166} s={10} w={700} c={C.paper}>
          ▷ Run
        </Txt>
        <rect x={206} y={14} width={104} height={172} rx={10} fill={C.paper} stroke={C.faint} />
        <Cells x={218} y={30} rows={4} cols={4} s={20} ts={10} cell={(r, c) => (hit(r, c) ? { f: C.blueL, st: C.blue, t: r * 4 + c, tc: C.blue, hi: true } : { t: r * 4 + c, tc: C.soft })} />
        <rect x={218} y={126} width={80} height={20} rx={4} fill={D.bg} />
        <Txt x={258} y={136} m s={10} w={700} c={C.paper}>
          a[1:3, ::2]
        </Txt>
        <rect x={238} y={154} width={40} height={18} rx={4} fill="#22c55e" />
        <Txt x={258} y={163} s={9} w={800} c={C.paper}>
          VIEW
        </Txt>
      </>
    )
  },

  // the lab's grids: a blue column plus an orange row, each stretched (dashed copies) into a green 3 × 4 result
  'numpy-broadcast': () => (
    <>
      <Cells x={14} y={56} rows={3} cols={4} s={20} gap={3} rx={3} ts={10} cell={(r, c) => (c === 0 ? { f: '#3b82f6', st: '#2563eb', t: r, tc: C.paper } : { f: '#eff6ff', st: '#93c5fd', dash: '3 2', t: r, tc: '#93c5fd' })} />
      <Txt x={114} y={89} s={20} w={700} c={C.soft}>
        +
      </Txt>
      <Cells x={126} y={56} rows={3} cols={4} s={20} gap={3} rx={3} ts={10} cell={(r, c) => (r === 0 ? { f: '#f97316', st: '#ea580c', t: (c + 1) * 10, tc: C.paper } : { f: '#fff7ed', st: '#fdba74', dash: '3 2', t: (c + 1) * 10, tc: '#fdba74' })} />
      <Txt x={226} y={89} s={20} w={700} c={C.soft}>
        =
      </Txt>
      <Cells x={238} y={58} rows={3} cols={4} s={17} gap={2} rx={3} ts={8.5} cell={(r, c) => ({ f: '#10b981', st: '#059669', t: r + (c + 1) * 10, tc: C.paper })} />
      <Txt x={59} y={38} m s={12} w={700} c="#2563eb">
        (3, 1)
      </Txt>
      <Txt x={171} y={38} m s={12} w={700} c="#ea580c">
        (1, 4)
      </Txt>
      <Txt x={274} y={38} m s={12} w={700} c="#059669">
        (3, 4)
      </Txt>
      <Txt x={160} y={168} m s={11} c={C.muted}>
        (3, 1) + (1, 4) → (3, 4)
      </Txt>
    </>
  ),

  // the sliders on the left, the dark result card with the posterior on the right
  'bayes-basics': () => (
    <>
      {[
        ['P(F)', 0.25, '1%'],
        ['P(S|F)', 0.9, '90%'],
        ['P(S)', 0.45, '10%'],
      ].map(([label, v, val], i) => (
        <g key={i}>
          <Txt x={16} y={52 + i * 44} a="start" m s={10} c={C.muted}>
            {label}
          </Txt>
          <Txt x={130} y={52 + i * 44} a="end" m s={10} w={700} c={C.rose}>
            {val}
          </Txt>
          <rect x={16} y={64 + i * 44} width={114} height={5} rx={2.5} fill={C.faint} />
          <rect x={16} y={64 + i * 44} width={114 * (v as number)} height={5} rx={2.5} fill="#8b5cf6" />
          <circle cx={16 + 114 * (v as number)} cy={66.5 + i * 44} r={6} fill="#8b5cf6" stroke={C.paper} strokeWidth={2} />
        </g>
      ))}
      <Panel x={146} y={18} w={164} h={164} />
      <Flame x={198} y={46} k={0.55} />
      <Txt x={228} y={46} m s={12} w={700} c={D.dim}>
        |
      </Txt>
      <Cloud x={256} y={45} k={0.6} />
      <Txt x={228} y={92} s={34} w={800} c="#f43f5e">
        9.00%
      </Txt>
      <rect x={176} y={126} width={104} height={38} rx={8} fill={D.panel} stroke={D.edge} />
      <Txt x={228} y={139} m s={9} c={D.dim}>
        P(F|S) =
      </Txt>
      <Txt x={228} y={153} m s={10} w={700} c={D.text}>
        0.01×0.9 / 0.1
      </Txt>
    </>
  ),

  // one square per person: the positives (amber, a few red) and how few of them are really ill
  'bayes-virus': () => {
    const sick = new Set([47, 112])
    const falsePos = new Set([5, 18, 22, 30, 41, 58, 66, 73, 81, 90, 99, 104, 118, 125, 133, 141, 150, 157])
    return (
      <>
        <Cells x={22} y={18} rows={8} cols={20} s={11} gap={3} rx={2} cell={(r, c) => ({ f: sick.has(r * 20 + c) ? '#ef4444' : falsePos.has(r * 20 + c) ? '#f59e0b' : '#e2e8f0' })} />
        <Txt x={22} y={160} a="start" s={28} w={800} c="#e11d48">
          1.94%
        </Txt>
        <Txt x={124} y={163} a="start" m s={12} c={C.muted}>
          = 10 / 510
        </Txt>
        <rect x={22} y={180} width={9} height={9} rx={2} fill="#ef4444" />
        <rect x={86} y={180} width={9} height={9} rx={2} fill="#f59e0b" />
        <Txt x={36} y={185} a="start" m s={9} c={C.muted}>
          virus +
        </Txt>
        <Txt x={100} y={185} a="start" m s={9} c={C.muted}>
          no virus +
        </Txt>
      </>
    )
  },

  // the lab's dark result card: each class's share after multiplying the table's probabilities
  'naive-bayes': () => (
    <>
      <Panel />
      <Txt x={30} y={36} a="start" m s={10} c={D.dim}>
        P(y)·∏ P(x|y)
      </Txt>
      {[
        ['y = yes', '20.46%', 0.2046, '0.00529', '#3b82f6', D.blue],
        ['y = no', '79.54%', 0.7954, '0.02057', '#94a3b8', D.text],
      ].map(([label, pct, v, raw, bar, tc], i) => (
        <g key={i}>
          <Txt x={30} y={70 + i * 58} a="start" s={11} w={800} c={D.bright}>
            {label}
          </Txt>
          <Txt x={290} y={68 + i * 58} a="end" m s={16} c={tc as string}>
            {pct}
          </Txt>
          <rect x={30} y={84 + i * 58} width={260} height={6} rx={3} fill={D.edge} />
          <rect x={30} y={84 + i * 58} width={260 * (v as number)} height={6} rx={3} fill={bar as string} />
          <Txt x={30} y={102 + i * 58} a="start" m s={9} c={D.dim}>
            = {raw}
          </Txt>
        </g>
      ))}
    </>
  ),

  // one bell curve per class; the likelihood of x is each curve's height there
  'gaussian-nb': () => (
    <>
      <line x1={14} x2={306} y1={160} y2={160} stroke={C.line} />
      <path d={bell(118, 30, 112, 160, 14, 306)} fill="none" stroke="#e11d48" strokeWidth={2} />
      <path d={bell(206, 38, 82, 160, 14, 306)} fill="none" stroke="#2563eb" strokeWidth={2} />
      <line x1={136} x2={136} y1={34} y2={160} stroke={C.text} strokeDasharray="4 3" />
      <Txt x={136} y={24} m s={11} w={700} c={C.ink}>
        x = 36.6
      </Txt>
      <circle cx={136} cy={160 - 112 * Math.exp(-(18 ** 2) / 1800)} r={4.5} fill="#e11d48" />
      <circle cx={136} cy={160 - 82 * Math.exp(-(70 ** 2) / 2888)} r={4.5} fill="#2563eb" />
      <Txt x={144} y={62} a="start" m s={10} w={700} c="#e11d48">
        0.0013
      </Txt>
      <Txt x={128} y={150} a="end" m s={10} w={700} c="#2563eb">
        0.00041
      </Txt>
      <circle cx={20} cy={180} r={4} fill="#e11d48" />
      <Txt x={28} y={180} a="start" s={10} c={C.text}>
        y = 1
      </Txt>
      <circle cx={74} cy={180} r={4} fill="#2563eb" />
      <Txt x={82} y={180} a="start" s={10} c={C.text}>
        y = 0
      </Txt>
    </>
  ),

  // the test sample (green), the dashed circle round its K nearest, everything else faded
  knn: () => {
    const far = [[52, 58], [78, 38], [96, 76], [62, 98], [114, 46], [84, 132], [106, 104], [206, 70], [234, 112], [252, 62], [216, 154], [264, 140], [238, 36], [284, 96], [196, 172]]
    const near = [[140, 78, '#3b82f6', '#1'], [138, 122, '#3b82f6', '#2'], [182, 120, '#ef4444', '#3']] as const
    return (
      <>
        {[60, 120, 180, 240].map((x) => (
          <line key={x} x1={x} x2={x} y1={10} y2={190} stroke="#f1f5f9" />
        ))}
        {far.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={4} fill={x < 160 ? '#93c5fd' : '#fca5a5'} opacity={0.7} />
        ))}
        <circle cx={160} cy={100} r={42} fill="#3b82f6" fillOpacity={0.05} stroke="#60a5fa" strokeWidth={1.5} strokeDasharray="4 4" />
        {near.map(([x, y]) => (
          <line key={x} x1={160} y1={100} x2={x} y2={y} stroke="#10b981" strokeWidth={2} />
        ))}
        {near.map(([x, y, c, label]) => (
          <g key={label}>
            <circle cx={x} cy={y} r={8} fill={c} stroke={C.paper} strokeWidth={1.5} />
            <Txt x={x} y={y + 15} s={8.5} w={800} c={C.ink}>
              {label}
            </Txt>
          </g>
        ))}
        <circle cx={160} cy={100} r={6.5} fill="#10b981" stroke={C.paper} strokeWidth={2} />
        <Txt x={170} y={88} a="start" s={8.5} w={800} c="#059669">
          TEST
        </Txt>
        <Txt x={18} y={180} a="start" m s={14} w={700} c={C.ink}>
          K = 3
        </Txt>
      </>
    )
  },

  // the lab's confusion matrix: rounded cells with coloured borders, then the scores
  evaluation: () => {
    const cells = [
      ['TP', 42, '#dcfce7', '#22c55e'],
      ['FN', 8, '#ffe4e6', '#f43f5e'],
      ['FP', 5, '#fef9c3', '#eab308'],
      ['TN', 45, '#f1f5f9', '#94a3b8'],
    ] as const
    return (
      <>
        <Txt x={100} y={20} s={9} w={800} c={C.blue}>
          PREDICTED →
        </Txt>
        <text x={0} y={0} fontSize={9} fontWeight={800} fill={C.blue} textAnchor="middle" transform="translate(24 96) rotate(-90)">
          ACTUAL →
        </text>
        {cells.map(([label, n, f, st], i) => {
          const x = 46 + (i % 2) * 56
          const y = 40 + Math.floor(i / 2) * 56
          return (
            <g key={label}>
              <rect x={x} y={y} width={50} height={50} rx={9} fill={f} stroke={st} strokeWidth={2} />
              <Txt x={x + 7} y={y + 10} a="start" s={8} w={800} c={st}>
                {label}
              </Txt>
              <Txt x={x + 25} y={y + 29} m s={17} w={800} c={C.ink}>
                {n}
              </Txt>
            </g>
          )
        })}
        <rect x={174} y={40} width={134} height={106} rx={10} fill={C.paper} stroke={C.faint} />
        {[
          ['Precision', '.89'],
          ['Recall', '.84'],
          ['F1', '.86'],
        ].map(([label, v], i) => (
          <g key={label}>
            <Txt x={186} y={64 + i * 29} a="start" m s={10.5} c={C.text}>
              {label}
            </Txt>
            <Txt x={296} y={64 + i * 29} a="end" m s={11} w={800} c={C.blue}>
              {v}
            </Txt>
          </g>
        ))}
        {[
          ['#22c55e', 'TP'],
          ['#f43f5e', 'FN'],
          ['#eab308', 'FP'],
        ].map(([c, l], i) => (
          <g key={l}>
            <circle cx={50 + i * 44} cy={170} r={3.5} fill={c} />
            <Txt x={57 + i * 44} y={170} a="start" s={9} w={700} c={C.text}>
              {l}
            </Txt>
          </g>
        ))}
      </>
    )
  },

  // sorted samples cut into folds, then each round: the test fold's classes never appear in training
  'cross-validation': () => {
    const col = ['#2563eb', '#e11d48', '#16a34a', '#ea580c']
    return (
      <>
        {[0, 1].map((f) => (
          <g key={f}>
            <Txt x={22 + f * 146} y={22} a="start" s={9} w={800} c={C.muted}>
              FOLD {f + 1}
            </Txt>
            <rect x={16 + f * 146} y={30} width={138} height={52} rx={6} fill={C.paper} stroke={C.faint} />
            {Array.from({ length: 40 }, (_, i) => (
              <circle key={i} cx={28 + f * 146 + (i % 10) * 12.5} cy={42 + Math.floor(i / 10) * 10} r={3.6} fill={col[f * 2 + (i % 10 < 5 ? 0 : 1)]} />
            ))}
          </g>
        ))}
        {[0, 1].map((r) => (
          <g key={r}>
            <Txt x={16} y={116 + r * 40} a="start" m s={10} w={700} c={C.text}>
              test {r + 1}
            </Txt>
            {col.map((c, i) => (
              <rect key={i} x={70 + i * 46} y={109 + r * 40} width={46} height={14} fill={c} opacity={Math.floor(i / 2) === r ? 1 : 0.18} />
            ))}
            <Txt x={300} y={116 + r * 40} a="end" s={14} w={800} c="#e11d48">
              0%
            </Txt>
          </g>
        ))}
      </>
    )
  },

  // the animation's frame: soft points, ⊗ centroids moving to their cluster's middle, the dark elbow card
  kmeans: () => {
    const offs = [[-20, -8], [-8, -18], [10, -12], [-22, 10], [2, 2], [18, 8], [-6, 18], [22, -4], [8, 20]]
    const groups = [
      { c: '#3b82f6', p: '#60a5fa', at: [70, 70], old: [110, 100], f: (o: number[]) => o },
      { c: '#ef4444', p: '#f87171', at: [96, 154], old: [140, 132], f: (o: number[]) => [o[1], o[0] * 0.8] },
      { c: '#10b981', p: '#34d399', at: [196, 128], old: [166, 92], f: (o: number[]) => [-o[0], -o[1]] },
    ]
    return (
      <>
        {groups.map((g) =>
          offs.map((o, i) => {
            const [dx, dy] = g.f(o)
            return <circle key={`${g.c}${i}`} cx={g.at[0] + dx} cy={g.at[1] + dy} r={4.5} fill={g.p} />
          }),
        )}
        {groups.map((g) => (
          <g key={g.c}>
            <Centroid x={g.old[0]} y={g.old[1]} c={g.c} faded />
            <Arrow x1={g.old[0]} y1={g.old[1]} x2={g.at[0] + (g.old[0] - g.at[0]) * 0.3} y2={g.at[1] + (g.old[1] - g.at[1]) * 0.3} c={g.c} w={1.5} dash="4 3" />
            <Centroid x={g.at[0]} y={g.at[1]} c={g.c} />
          </g>
        ))}
        <Panel x={236} y={12} w={74} h={100} />
        <Txt x={246} y={26} a="start" s={7.5} w={800} c={D.dim}>
          ELBOW
        </Txt>
        {[40, 62, 84].map((y) => (
          <line key={y} x1={246} x2={302} y1={y} y2={y} stroke={D.edge} strokeDasharray="2 3" />
        ))}
        <polyline points="248,38 259,66 270,84 281,89 292,92 302,94" fill="none" stroke="#3b82f6" strokeWidth={2} />
        <circle cx={270} cy={84} r={3.5} fill={D.amber} />
      </>
    )
  },

  // the exam table next to the lab's plot: ringed centroids C1 / C2 and the point they claim
  'kmeans-table': () => {
    const rows = [
      ['A', '1.4', '5.0', 0],
      ['B', '2.2', '3.6', 0],
      ['C', '4.1', '1.0', 1],
      ['D', '5.8', '0.6', 1],
    ] as const
    const col = ['#2563eb', '#e11d48']
    const light = [C.blueL, C.roseL]
    return (
      <>
        <rect x={12} y={20} width={168} height={128} rx={10} fill={C.paper} stroke={C.faint} />
        <line x1={12} x2={180} y1={44} y2={44} stroke={C.faint} />
        {['pt', 'd(C1)', 'd(C2)', '→'].map((h, i) => (
          <Txt key={h} x={[30, 70, 118, 160][i]} y={32} m s={9.5} w={700} c={i === 1 ? col[0] : i === 2 ? col[1] : C.muted}>
            {h}
          </Txt>
        ))}
        {rows.map(([p, d1, d2, k], r) => {
          const y = 58 + r * 24
          return (
            <g key={p}>
              <rect x={k ? 101 : 53} y={y - 8.5} width={34} height={17} rx={4} fill={light[k]} />
              <Txt x={30} y={y} m w={700} c={C.ink}>
                {p}
              </Txt>
              <Txt x={70} y={y} m w={k ? 400 : 700} c={k ? C.soft : col[0]}>
                {d1}
              </Txt>
              <Txt x={118} y={y} m w={k ? 700 : 400} c={k ? col[1] : C.soft}>
                {d2}
              </Txt>
              <Txt x={160} y={y} m w={800} c={col[k]}>
                C{k + 1}
              </Txt>
            </g>
          )
        })}
        {[0, 1, 2, 3].map((i) => (
          <line key={i} x1={196} x2={308} y1={40 + i * 36} y2={40 + i * 36} stroke="#f1f5f9" />
        ))}
        <line x1={218} y1={148} x2={296} y2={40} stroke="#fda4af" />
        <line x1={256} y1={148} x2={272} y2={86} stroke="#e11d48" strokeDasharray="3 3" strokeWidth={1.5} />
        {[
          ['A', 218, 148, 0],
          ['B', 256, 148, 1],
          ['C', 272, 86, 1],
          ['D', 296, 40, 1],
        ].map(([p, x, y, k]) => (
          <g key={p as string}>
            <circle cx={x as number} cy={y as number} r={5} fill={col[k as number]} />
            <Txt x={(x as number) + 9} y={(y as number) - 1} a="start" s={9} w={700} c={C.ink}>
              {p}
            </Txt>
          </g>
        ))}
        <circle cx={218} cy={148} r={9} fill="none" stroke={col[0]} strokeWidth={3} />
        <circle cx={256} cy={148} r={9} fill="none" stroke={col[1]} strokeWidth={3} />
        <Txt x={218} y={132} s={9} w={800} c={col[0]}>
          C1
        </Txt>
        <Txt x={256} y={132} s={9} w={800} c={col[1]}>
          C2
        </Txt>
        <rect x={262} y={109} width={10} height={10} fill="#e11d48" transform="rotate(45 267 114)" />
        <Txt x={96} y={170} m s={10} c={C.muted}>
          C2 ← mean(B, C, D)
        </Txt>
      </>
    )
  },

  // the lab's plot: pink and lilac half-planes, blue dots vs red squares, the line and its update table
  perceptron: () => (
    <>
      <rect x={16} y={18} width={160} height={164} fill="#fdf2f4" stroke={C.faint} />
      <polygon points="16,18 176,18 176,58 16,30" fill="#ede9fe" />
      <line x1={16} y1={30} x2={176} y2={58} stroke={C.ink} strokeWidth={2} />
      <line x1={16} y1={70} x2={176} y2={40} stroke={C.ink} strokeWidth={1.2} strokeDasharray="4 4" opacity={0.5} />
      {[40, 80, 120, 160].map((y) => (
        <line key={y} x1={16} x2={176} y1={y + 2} y2={y + 2} stroke="#fce7f3" />
      ))}
      <circle cx={140} cy={36} r={5} fill="#2563eb" />
      <circle cx={112} cy={32} r={5} fill="#2563eb" />
      {[[54, 90], [136, 96], [58, 152], [140, 150]].map(([x, y]) => (
        <rect key={`${x}${y}`} x={x - 4.5} y={y - 4.5} width={9} height={9} fill="#e11d48" />
      ))}
      <rect x={190} y={18} width={118} height={124} rx={10} fill={C.paper} stroke={C.faint} />
      <line x1={190} x2={308} y1={42} y2={42} stroke={C.faint} />
      {['w₁', 'w₂', 'θ'].map((h, i) => (
        <Txt key={h} x={216 + i * 34} y={30} m c={C.muted}>
          {h}
        </Txt>
      ))}
      <rect x={194} y={110} width={110} height={22} rx={5} fill={C.blueL} />
      {[
        ['0.1', '0.5', '−0.8'],
        ['0.1', '1.5', '0.2'],
        ['0.1', '0.5', '1.2'],
        ['0.6', '1.5', '1.2'],
      ].map((row, r) =>
        row.map((v, i) => (
          <Txt key={`${r}${i}`} x={216 + i * 34} y={56 + r * 20 + (r === 3 ? 4 : 0)} m s={11} w={r === 3 ? 700 : 400} c={r === 3 ? C.blue : C.muted}>
            {v}
          </Txt>
        )),
      )}
      <rect x={190} y={152} width={118} height={28} rx={8} fill="#2563eb" />
      <Txt x={249} y={166} s={11} w={700} c={C.paper}>
        Next row →
      </Txt>
    </>
  ),

  // the Chain Rule Tracker's dark network: inputs i, j feed the output neuron k; δ flows back
  backprop: () => (
    <>
      <Panel />
      <Txt x={298} y={30} a="end" m s={9} c={D.dim}>
        E = 0.1128
      </Txt>
      <line x1={58} y1={58} x2={218} y2={104} stroke="#3b82f6" strokeWidth={2.5} />
      <line x1={58} y1={146} x2={218} y2={104} stroke="#f43f5e" strokeWidth={2.5} />
      <Txt x={112} y={56} m s={11} c={D.bright}>
        w = 0.40
      </Txt>
      <Txt x={112} y={152} m s={11} c={D.bright}>
        w = −0.60
      </Txt>
      {[
        ['i', 58, '1.00'],
        ['j', 146, '0.50'],
      ].map(([l, y, o]) => (
        <g key={l as string}>
          <circle cx={58} cy={y as number} r={16} fill="#3b82f6" />
          <Txt x={58} y={y as number} s={14} w={700} c={C.paper}>
            {l}
          </Txt>
          <Txt x={58} y={(y as number) + 26} m s={10} c={D.text}>
            O = {o}
          </Txt>
        </g>
      ))}
      <circle cx={226} cy={104} r={22} fill="#ef4444" stroke="#fecaca" strokeWidth={3} />
      <Txt x={226} y={104} s={15} w={700} c={C.paper}>
        k
      </Txt>
      <Txt x={226} y={68} m s={11} w={700} c={D.amber}>
        δ = 0.1185
      </Txt>
      <Txt x={226} y={140} m s={10} c={D.text}>
        O = 0.525
      </Txt>
    </>
  ),

  // the lab's network on input (0, 0): ringed nodes with values, signed weights, δ in red
  'xor-mlp': () => {
    const edges = [
      [54, 66, 170, 66, '#f9a8d4', 'w1 −0.65'],
      [54, 66, 170, 148, '#93c5fd', ''],
      [54, 148, 170, 66, '#93c5fd', ''],
      [54, 148, 170, 148, '#93c5fd', 'w4 0.84'],
      [170, 66, 278, 106, '#93c5fd', ''],
      [170, 148, 278, 106, '#f9a8d4', ''],
    ] as const
    const nodes = [
      [54, 66, 'x₁', '0', '#94a3b8'],
      [54, 148, 'x₂', '0', '#94a3b8'],
      [170, 66, 'h₁', '0.5', '#3b82f6'],
      [170, 148, 'h₂', '0.5', '#3b82f6'],
      [278, 106, 'o', '0.435', '#3b82f6'],
    ] as const
    return (
      <>
        {edges.map(([x1, y1, x2, y2, c], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={3} />
        ))}
        {edges.map(([x1, y1, x2, y2, , l]) =>
          l ? (
            <g key={l}>
              <rect x={(x1 + x2) / 2 - 27} y={(y1 + y2) / 2 - 9} width={54} height={18} rx={3} fill={C.paper} stroke={C.faint} />
              <Txt x={(x1 + x2) / 2} y={(y1 + y2) / 2} m s={10} c={C.ink}>
                {l}
              </Txt>
            </g>
          ) : null,
        )}
        {nodes.map(([x, y, l, v, c]) => (
          <g key={l}>
            <circle cx={x} cy={y} r={22} fill={C.paper} stroke={c} strokeWidth={2.5} />
            <Txt x={x} y={y - 7} m s={9} c={C.text}>
              {l}
            </Txt>
            <Txt x={x} y={y + 7} m s={11} w={800} c={C.ink}>
              {v}
            </Txt>
          </g>
        ))}
        <Txt x={170} y={30} m s={10.5} w={700} c="#e11d48">
          δ = 0.023
        </Txt>
        <Txt x={170} y={184} m s={10.5} w={700} c="#e11d48">
          δ = −0.037
        </Txt>
        <Txt x={278} y={70} m s={10.5} w={700} c="#e11d48">
          δ = 0.107
        </Txt>
      </>
    )
  },

  // input image ∗ Sobel-x kernel (the lab's blue kernel card) = the vertical edges light up
  kernel: () => {
    const img = ['........', '..####..', '.######.', '.######.', '.######.', '.######.', '..####..', '........']
    const on = (r: number, c: number) => img[r]?.[c] === '#'
    const edge = (r: number, c: number) => on(r, c) && !(on(r, c - 1) && on(r, c + 1))
    const k = ['−1', '0', '1', '−2', '0', '2', '−1', '0', '1']
    return (
      <>
        <Cells x={12} y={56} rows={8} cols={8} s={11} cell={(r, c) => ({ f: on(r, c) ? '#9ca3af' : '#0a0a0a', st: '#262626' })} />
        <rect x={116} y={50} width={84} height={100} rx={10} fill="#eff6ff" stroke="#bfdbfe" />
        <Txt x={158} y={64} s={8} w={800} c={C.muted}>
          KERNEL
        </Txt>
        <Cells x={126} y={76} rows={3} cols={3} s={18} gap={4} rx={4} ts={10} cell={(r, c) => ({ f: C.paper, st: '#93c5fd', t: k[r * 3 + c], tc: '#1d4ed8' })} />
        <Cells x={220} y={56} rows={8} cols={8} s={11} cell={(r, c) => ({ f: edge(r, c) ? '#ffffff' : '#0a0a0a', st: '#262626' })} />
      </>
    )
  },

  // the "Slide the window" panel: padded input, blue-ringed window, flipped kernel, output filling in
  convolution: () => {
    const v = [3, 1, 2, 0, 1, 4, 0, 2, 2, 0, 1, 3, 0, 2, 3, 1]
    const k = ['−1', '0', '1', '−2', '0', '2', '−1', '0', '1']
    return (
      <>
        <Cells
          x={14}
          y={40}
          rows={6}
          cols={6}
          s={16}
          gap={3}
          rx={3}
          ts={9.5}
          cell={(r, c) => {
            const pad = r === 0 || c === 0 || r === 5 || c === 5
            const win = r < 3 && c < 3
            if (win) return { f: C.paper, st: '#2563eb', t: pad ? '0' : v[(r - 1) * 4 + (c - 1)], tc: pad ? C.soft : C.ink, hi: true }
            return pad ? { f: '#f8fafc', st: C.faint, t: '0', tc: C.line } : { f: C.paper, st: C.line, t: v[(r - 1) * 4 + (c - 1)], tc: C.ink }
          }}
        />
        <Txt x={149} y={64} a="start" s={9} c={C.muted}>
          flipped
        </Txt>
        <Cells x={146} y={76} rows={3} cols={3} s={14} gap={3} rx={3} ts={9} cell={(r, c) => ({ f: C.paper, st: '#93c5fd', t: k[r * 3 + c], tc: '#1d4ed8' })} />
        <Txt x={222} y={46} a="start" s={9} c={C.muted}>
          out 4×4
        </Txt>
        <Cells x={222} y={58} rows={4} cols={4} s={18} gap={3} rx={3} ts={11} cell={(r, c) => (r + c === 0 ? { f: '#2563eb', st: '#2563eb', t: 6, tc: C.paper } : { f: C.paper, st: C.faint })} />
        <rect x={14} y={164} width={70} height={22} rx={6} fill="#2563eb" />
        <Txt x={49} y={175} s={9.5} w={700} c={C.paper}>
          ▷ Play all
        </Txt>
        <Txt x={306} y={175} a="end" m s={9.5} c={C.muted}>
          out[0][0] = 6
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
          <rect key={i} x={16 + i * 6} y={146 - v} width={5} height={v} fill={i < T ? '#475569' : '#93c5fd'} />
        ))}
        <line x1={14} x2={210} y1={146} y2={146} stroke={C.soft} strokeWidth={1.5} />
        {h.map((_, i) => (
          <rect key={`g${i}`} x={16 + i * 6} y={152} width={6} height={8} fill={g(Math.round((i / 31) * 255))} />
        ))}
        <line x1={16 + T * 6 - 0.5} x2={16 + T * 6 - 0.5} y1={34} y2={162} stroke="#ef4444" strokeWidth={2} />
        <Txt x={16 + T * 6 - 0.5} y={24} m s={11} w={800} c="#ef4444">
          T = 127.9
        </Txt>
        <Cells x={232} y={18} rows={8} cols={8} s={9} cell={(r, c) => ({ f: g(Math.round(Math.min(255, px(r, c)))), st: 'none' })} />
        <Txt x={268} y={100} s={14} c={C.muted}>
          ↓
        </Txt>
        <Cells x={232} y={110} rows={8} cols={8} s={9} cell={(r, c) => ({ f: px(r, c) > 120 ? '#ffffff' : '#000000', st: 'none' })} />
        <rect x={232} y={110} width={72} height={72} fill="none" stroke={C.line} />
      </>
    )
  },

  // "Where does each pixel go?": the black pixel grids, one input pixel (amber) and where it lands (blue)
  affine: () => {
    const F = ['.......', '.#####.', '.#.....', '.####..', '.#.....', '.#.....', '.......']
    const on = (r: number, c: number) => F[r]?.[c] === '#'
    const nums = [
      ['0', '−1', '6'],
      ['1', '0', '0'],
    ]
    return (
      <>
        <Cells x={14} y={44} rows={7} cols={7} s={13} cell={(r, c) => (r === 1 && c === 5 ? { f: '#d1d5db', st: '#f59e0b', hi: true } : { f: on(r, c) ? '#d1d5db' : '#111111', st: '#3f3f46' })} />
        <Cells x={215} y={44} rows={7} cols={7} s={13} cell={(r, c) => (r === 5 && c === 5 ? { f: '#d1d5db', st: '#3b82f6', hi: true } : { f: on(6 - c, r) ? '#d1d5db' : '#111111', st: '#3f3f46' })} />
        <Txt x={60} y={30} s={9} w={800} c={C.muted}>
          INPUT
        </Txt>
        <Txt x={261} y={30} s={9} w={800} c={C.muted}>
          OUTPUT
        </Txt>
        <path d="M130 70H124V112H130M192 70H198V112H192" fill="none" stroke={C.ink} strokeWidth={2} />
        {nums.map((row, r) =>
          row.map((n, i) => (
            <Txt key={`${r}${i}`} x={[140, 161, 182][i]} y={82 + r * 20} m s={12.5} w={700} c={C.ink}>
              {n}
            </Txt>
          )),
        )}
        <Arrow x1={124} y1={132} x2={200} y2={132} c="#2563eb" w={2} />
        <rect x={14} y={150} width={292} height={36} rx={8} fill="#eff6ff" stroke="#bfdbfe" />
        <Txt x={160} y={168} m s={10} c={C.text}>
          out <tspan fill="#2563eb">(5, 5)</tspan> ← in <tspan fill="#d97706">(1, 5)</tspan>
        </Txt>
      </>
    )
  },

  // "The volume after each layer": light-blue boxes shrinking, then the flat dense layers
  // input, a grid of activation maps, the class bars
  'cnn-live': () => {
    const shade = (r: number, k: number) => ['#0f172a', '#334155', '#64748b', '#cbd5e1', '#1e293b', '#94a3b8', '#475569'][(r * 5 + k * 3) % 7]
    return (
      <>
        <rect x={14} y={78} width={36} height={36} rx={3} fill="#93c5fd" />
        <path d="M14 104 L 26 92 L 36 102 L 50 90 V 114 H 14 Z" fill="#2563eb" fillOpacity={0.6} />
        {Array.from({ length: 6 }, (_, r) => Array.from({ length: 7 }, (_, k) => (
          <rect key={`${r}-${k}`} x={60 + k * 25} y={32 + r * 23} width={22} height={20} rx={2} fill={shade(r, k)} />
        )))}
        {[56, 30, 18, 11, 7].map((w, i) => (
          <g key={i}>
            <rect x={240} y={46 + i * 22} width={w} height={17} rx={3} fill={i ? C.blueM : '#34d399'} />
          </g>
        ))}
      </>
    )
  },
  // the data and the boundary, then the same data in the hidden layer's bent space
  'nn-classify': () => {
    const dots = (pts: number[][], col: string) => pts.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={3.6} fill={col} stroke="#fff" strokeWidth={0.8} />)
    return (
      <>
        <rect x={14} y={40} width={136} height={120} rx={8} fill={C.blueL} />
        <circle cx={82} cy={100} r={38} fill={C.roseL} />
        {dots([[76, 92], [90, 108], [70, 112], [96, 88], [84, 100]], C.rose)}
        {dots([[30, 60], [132, 62], [34, 142], [128, 146], [82, 50], [82, 150], [24, 100], [140, 104]], C.blue)}
        <rect x={170} y={40} width={136} height={120} rx={8} fill="#f8fafc" stroke="#e2e8f0" />
        <path d="M180 150 L 296 50" stroke="#0f172a" strokeWidth={2} strokeDasharray="6 4" />
        {dots([[192, 70], [206, 84], [216, 62], [226, 92], [200, 104]], C.rose)}
        {dots([[252, 136], [270, 118], [282, 138], [288, 104], [246, 118], [266, 146]], C.blue)}
      </>
    )
  },
  // the picture, and the network's painting of it
  'nn-paint': () => {
    const scene = (x: number, blur: number) => (
      <g>
        <rect x={x} y={40} width={136} height={120} rx={8} fill="#fb923c" />
        <rect x={x} y={40} width={136} height={55} rx={8} fill="#6366f1" fillOpacity={0.7} />
        <circle cx={x + 92} cy={84} r={18} fill="#fef08a" />
        <path d={blur ? `M${x} 134 C ${x + 40} 108, ${x + 80} 140, ${x + 136} 116 V 160 H ${x} Z` : `M${x} 136 L ${x + 34} 108 L ${x + 62} 130 L ${x + 100} 102 L ${x + 136} 124 V 160 H ${x} Z`} fill="#312e81" />
      </g>
    )
    return (
      <>
        {scene(14, 0)}
        <Txt x={160} y={100} s={14} c={C.line}>→</Txt>
        {scene(170, 1)}
      </>
    )
  },
  'cnn-shapes': () => {
    const box: [string, string, string] = ['#bfdbfe', '#dbeafe', '#93c5fd']
    const flat: [string, string, string] = ['#c7d2fe', '#e0e7ff', '#a5b4fc']
    const layers = [
      { x: 12, w: 8, h: 70, d: 70, col: box, dim: '32²×3', kind: 'input', cx: 38 },
      { x: 74, w: 14, h: 60, d: 60, col: box, dim: '28²×6', kind: 'Conv2D', cx: 102 },
      { x: 138, w: 14, h: 40, d: 40, col: box, dim: '14²×6', kind: 'Pool', cx: 157 },
      { x: 186, w: 20, h: 30, d: 30, col: box, dim: '10²×16', kind: 'Conv2D', cx: 204 },
      { x: 240, w: 9, h: 76, d: 10, col: flat, dim: '120', kind: 'Dense', cx: 247 },
      { x: 282, w: 9, h: 30, d: 10, col: flat, dim: '10', kind: 'Dense', cx: 289 },
    ]
    return (
      <>
        {layers.map((l, i) => (
          <g key={i}>
            <Cuboid x={l.x} yc={92} w={l.w} h={l.h} d={l.d} col={l.col} />
            <Txt x={l.cx} y={162} m s={10} w={600} c={C.text}>
              {l.dim}
            </Txt>
            <Txt x={l.cx} y={177} s={9} c={C.soft}>
              {l.kind}
            </Txt>
            {i > 0 && (
              <Txt x={[0, 64, 128, 177, 226, 270][i]} y={92} s={11} c={C.line}>
                →
              </Txt>
            )}
          </g>
        ))}
      </>
    )
  },

  // the lab's dark editor holding the five lines of every training step
  pytorch: () => {
    const P = D.bright
    const lines: [string, string][][] = [
      [['for', D.violet], [' x, y ', P], ['in', D.violet], [' loader:', P]],
      [['    opt.', P], ['zero_grad', D.sky], ['()', P]],
      [['    pred = ', P], ['model', D.sky], ['(x)', P]],
      [['    loss = ', P], ['loss_fn', D.sky], ['(pred, y)', P]],
      [['    loss.', P], ['backward', D.sky], ['()', P]],
      [['    opt.', P], ['step', D.sky], ['()', P]],
    ]
    const notes = ['', '# clear', '# forward', '', '# grads', '# update']
    return (
      <>
        <Panel />
        <circle cx={26} cy={26} r={4} fill="#f87171" />
        <circle cx={40} cy={26} r={4} fill="#fbbf24" />
        <circle cx={54} cy={26} r={4} fill="#34d399" />
        {lines.map((parts, i) => (
          <g key={i}>
            <Code x={44} y={52 + i * 22} n={i + 1} parts={parts} s={11.5} />
            {notes[i] && (
              <Txt x={298} y={52 + i * 22} a="end" m s={10} i c={D.dim}>
                {notes[i]}
              </Txt>
            )}
          </g>
        ))}
      </>
    )
  },

  // the dark alpha-beta panel: MAX / MIN triangles, leaf values, a pruned branch in red
  alphabeta: () => {
    const mins = [70, 160, 250]
    const leaves = [
      [3, 12, 8],
      [2, 4, 6],
      [14, 5, 2],
    ]
    const minVal = ['3', '2', '2']
    const up = (x: number, y: number) => `${x},${y - 14} ${x + 16},${y + 12} ${x - 16},${y + 12}`
    const down = (x: number, y: number) => `${x - 16},${y - 12} ${x + 16},${y - 12} ${x},${y + 14}`
    return (
      <>
        <Panel />
        {mins.map((mx, i) => (
          <line key={mx} x1={160} y1={52} x2={mx} y2={84} stroke={i === 0 ? '#60a5fa' : '#475569'} strokeWidth={i === 0 ? 2.5 : 1.5} />
        ))}
        {mins.map((mx, i) =>
          leaves[i].map((v, j) => {
            const lx = mx - 28 + j * 28
            const cut = i === 1 && j > 0
            const best = i === 0 && j === 0
            return (
              <g key={`${i}${j}`}>
                <line x1={mx} y1={106} x2={lx} y2={138} stroke={best ? '#60a5fa' : cut ? '#f87171' : '#475569'} strokeWidth={best ? 2.5 : 1.5} strokeDasharray={cut ? '3 3' : undefined} />
                <circle cx={lx} cy={150} r={11} fill={best ? '#1e3a8a' : D.panel} stroke={best ? '#60a5fa' : cut ? '#7f1d1d' : '#64748b'} strokeWidth={1.5} strokeDasharray={cut ? '3 2' : undefined} />
                <Txt x={lx} y={150.5} m s={10} w={700} c={cut ? '#7f1d1d' : D.bright}>
                  {v}
                </Txt>
              </g>
            )
          }),
        )}
        <polygon points={up(160, 40)} fill="#1e3a8a" stroke="#60a5fa" strokeWidth={2} strokeLinejoin="round" />
        <Txt x={160} y={45} m s={10} w={800} c={D.bright}>
          3
        </Txt>
        {mins.map((mx, i) => (
          <g key={mx}>
            <polygon points={down(mx, 96)} fill={i === 0 ? '#1e3a8a' : D.panel} stroke={i === 0 ? '#60a5fa' : '#94a3b8'} strokeWidth={2} strokeLinejoin="round" />
            <Txt x={mx} y={91} m s={10} w={800} c={D.bright}>
              {minVal[i]}
            </Txt>
          </g>
        ))}
        <Txt x={22} y={42} a="start" m s={8.5} c={D.dim}>
          MAX
        </Txt>
        <Txt x={22} y={96} a="start" m s={8.5} c={D.dim}>
          MIN
        </Txt>
        <Txt x={194} y={36} a="start" m s={10} w={700} c={D.blue}>
          α = 3
        </Txt>
        <Txt x={160} y={176} m s={10} w={700} c={D.rose}>
          β = 2 ≤ α → pruned
        </Txt>
      </>
    )
  },
}

export default function LabArt({ id }: { id: ModuleId }) {
  return (
    <svg viewBox="0 0 320 200" className="h-full w-full" aria-hidden="true" focusable="false">
      <rect width={320} height={200} fill="#ffffff" />
      {ART[id]()}
    </svg>
  )
}
