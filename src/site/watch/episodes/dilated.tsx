/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, seg } from '../../../lib/explainer'
import { dilatedConv } from '../../../lib/conv2d'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Dilated convolution, the way Final 2024 Q6 codes it" — a 3×3 kernel that sees 5×5, built by shift-and-add.

const IMG10 = Array.from({ length: 10 }, (_, r) => Array.from({ length: 10 }, (_, c) => r * 10 + c))
const KER = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
const RES = dilatedConv(IMG10, KER, 2, 2, 'same')

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.dilated.kicker')} title={t('watch.dilated.title')} sub={t('watch.dilated.sub')} />
}

/** 7×7 board with a 3×3 kernel's taps at dilation d, centred at (cr, cc). */
function Taps({ d, cr, cc, o = 1 }: { d: number; cr: number; cc: number; o?: number }) {
  const cs = 64
  const x0 = 170
  const y0 = 170
  const taps = new Set<string>()
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) taps.add(`${cr + i * d},${cc + j * d}`)
  const span = d * 2 + 1
  return (
    <g opacity={o}>
      {Array.from({ length: 49 }, (_, k) => {
        const r = Math.floor(k / 7)
        const c = k % 7
        const on = taps.has(`${r},${c}`)
        return <rect key={k} x={x0 + c * cs} y={y0 + r * cs} width={cs - 5} height={cs - 5} rx={8} fill={on ? C.yellow : '#1a2130'} fillOpacity={on ? 0.85 : 1} stroke={on ? C.yellow : '#2a3446'} strokeWidth={2} />
      })}
      <rect x={x0 + (cc - (span - 1) / 2) * cs - 6} y={y0 + (cr - (span - 1) / 2) * cs - 6} width={span * cs + 7} height={span * cs + 7} rx={12} fill="none" stroke={C.teal} strokeWidth={3} strokeDasharray="8 6" />
    </g>
  )
}

function DilationScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const d = p < 0.4 ? 1 : 2
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Taps d={d} cr={3} cc={3} />
      </Svg>
      <At x={1200} y={220} size={56} className="font-black font-mono" color={C.yellow}>d = {d}</At>
      <At x={1200} y={330} size={30} w={620} style={{ textAlign: 'center' }}>{t(d === 1 ? 'watch.dilated.label.d1' : 'watch.dilated.label.d2')}</At>
      <At x={1200} y={480} size={28} color={C.teal} w={620} o={eseg(p, 0.5, 0.6)} style={{ textAlign: 'center' }}>{t('watch.dilated.label.same_params')}</At>
    </>
  )
}

function FormulaScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="k_{\text{eff}} = d\,(k - 1) + 1 = 2\cdot 2 + 1 = 5" x={800} y={170} size={44} color={C.teal} o={eseg(p, 0, 0.12)} />
      <Tex f="\text{pad}_{\text{same}} = \dfrac{d\,(k - 1)}{2} = 2" x={800} y={320} size={44} color={C.yellow} o={eseg(p, 0.2, 0.32)} />
      <Tex f="\text{out} = \left\lfloor \dfrac{N + 2\,\text{pad} - k_{\text{eff}}}{S} \right\rfloor + 1 = \left\lfloor \dfrac{10 + 4 - 5}{2} \right\rfloor + 1 = 5" x={800} y={500} size={40} o={eseg(p, 0.4, 0.55)} />
      <At x={800} y={680} size={28} color={C.muted} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.dilated.label.paper_formula')}</At>
    </>
  )
}

const CS = 44
function Padded({ x, y, hl }: { x: number; y: number; hl?: { rows: number[]; cols: number[]; color: string } }) {
  const n = 14
  const pad = RES.pad
  const hr = new Set(hl?.rows ?? [])
  const hc = new Set(hl?.cols ?? [])
  return (
    <g>
      {Array.from({ length: n * n }, (_, k) => {
        const r = Math.floor(k / n)
        const c = k % n
        const inside = r >= pad && c >= pad && r < n - pad && c < n - pad
        const on = hr.has(r) && hc.has(c)
        return (
          <g key={k}>
            <rect x={x + c * CS} y={y + r * CS} width={CS - 3} height={CS - 3} rx={5} fill={on ? hl!.color : inside ? '#1a2130' : C.bg} fillOpacity={on ? 0.85 : 1} stroke={inside ? '#2a3446' : C.axis} strokeDasharray={inside ? undefined : '3 3'} strokeWidth={1.2} />
            <text x={x + c * CS + (CS - 3) / 2} y={y + r * CS + CS / 2 + 4} textAnchor="middle" fill={on ? C.bg : inside ? C.muted : '#3b4556'} fontSize={15} fontFamily="ui-monospace, monospace" fontWeight={on ? 800 : 400}>
              {inside ? IMG10[r - pad][c - pad] : 0}
            </text>
          </g>
        )
      })}
    </g>
  )
}

function OutGrid({ x, y, v, o = 1 }: { x: number; y: number; v: number[][]; o?: number }) {
  return (
    <g opacity={o}>
      {v.flatMap((row, r) =>
        row.map((val, c) => (
          <g key={`${r}-${c}`}>
            <rect x={x + c * 74} y={y + r * 60} width={70} height={56} rx={8} fill="#173a35" stroke={C.teal} strokeWidth={1.5} />
            <text x={x + c * 74 + 35} y={y + r * 60 + 35} textAnchor="middle" fill={C.text} fontSize={21} fontFamily="ui-monospace, monospace" fontWeight={700}>{val}</text>
          </g>
        )),
      )}
    </g>
  )
}

function ShiftAddScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const diag = [0, 1, 2] // the kernel's non-zero cells are (0,0), (1,1), (2,2)
  const step = Math.min(2, Math.floor(seg(p, 0.08, 0.8) * 3))
  const i = diag[step]
  const cols = [C.yellow, C.blue, C.red]
  // partial sums after the first `step + 1` kernel cells
  const partial = RES.out.map((row, r) =>
    row.map((_, c) => diag.slice(0, step + 1).reduce((s, k) => s + (() => {
      const pr = RES.rows[k][r] - RES.pad
      const pc = RES.cols[k][c] - RES.pad
      return pr < 0 || pc < 0 || pr >= 10 || pc >= 10 ? 0 : IMG10[pr][pc]
    })(), 0)),
  )
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Padded x={70} y={150} hl={{ rows: RES.rows[i], cols: RES.cols[i], color: cols[step] }} />
        <OutGrid x={1000} y={330} v={partial} />
      </Svg>
      <At x={378} y={115} size={22} className="font-mono" color={C.muted}>{t('watch.dilated.label.padded')}</At>
      <At x={1180} y={150} size={26} className="font-mono" color={cols[step]}>
        kernel[{i}][{i}] = 1
      </At>
      <At x={1180} y={200} size={22} className="font-mono" color={C.muted}>
        rows = {i}·2 + 2·r = {RES.rows[i].join(', ')}
      </At>
      <At x={1180} y={240} size={22} className="font-mono" color={C.muted}>
        cols = {i}·2 + 2·c = {RES.cols[i].join(', ')}
      </At>
      <At x={1180} y={290} size={22} color={C.muted}>{t('watch.dilated.label.sum_after', { n: step + 1 })}</At>
      <At x={1180} y={680} size={26} color={C.teal} w={560} o={eseg(p, 0.82, 0.9)} style={{ textAlign: 'center' }}>{t('watch.dilated.label.matches')}</At>
    </>
  )
}

const TODOS = [
  'pad = dilation_rate * (kernel.shape[0] - 1) // 2',
  'output_rows = (padded_input.shape[0] - dilation_rate * (kernel.shape[0] - 1) - 1) // stride + 1',
  'output_cols = (padded_input.shape[1] - dilation_rate * (kernel.shape[1] - 1) - 1) // stride + 1',
  'input_row_indices = i * dilation_rate + np.arange(output_rows) * stride',
  'input_col_indices = j * dilation_rate + np.arange(output_cols) * stride',
  'kernel[i, j] * padded_input[input_row_indices][:, input_col_indices]',
]
function TodoScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
      </Svg>
      <At x={800} y={110} size={30} className="font-black" o={eseg(p, 0, 0.08)}>{t('watch.dilated.label.todos')}</At>
      {TODOS.map((code, k) => (
        <Fragment key={k}>
          <At x={120} y={210 + k * 90} size={24} anchor="l" color={C.yellow} className="font-mono font-black" o={eseg(p, 0.08 + k * 0.1, 0.14 + k * 0.1)}>TODO {k + 1}</At>
          <At x={270} y={210 + k * 90} size={20} anchor="l" className="font-mono" o={eseg(p, 0.08 + k * 0.1, 0.14 + k * 0.1)}>{code}</At>
        </Fragment>
      ))}
    </>
  )
}

function DropoutScene({ p }: { p: number }) {
  const { t } = useTranslation()
  // a fixed "random" mask so the frame is reproducible
  const mask = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
  const shown = eseg(p, 0.25, 0.45)
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {mask.map((m, k) => {
          const x = 180 + (k % 5) * 90
          const y = 250 + Math.floor(k / 5) * 90
          const dropped = m === 0 && shown > 0.5
          return <rect key={k} x={x} y={y} width={80} height={80} rx={10} fill={dropped ? C.bg : C.blue} fillOpacity={dropped ? 1 : 0.35} stroke={dropped ? C.red : C.blue} strokeDasharray={dropped ? '6 5' : undefined} strokeWidth={2.5} />
        })}
      </Svg>
      <At x={1150} y={220} size={25} className="font-mono" o={eseg(p, 0.05, 0.15)}>mask = np.random.rand(*input_array.shape) &gt; p</At>
      <At x={1150} y={320} size={26} color={C.muted} w={640} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.dilated.label.mask')}</At>
      <At x={1150} y={470} size={26} color={C.yellow} w={640} o={eseg(p, 0.55, 0.65)} style={{ textAlign: 'center' }}>{t('watch.dilated.label.why_dropout')}</At>
      <At x={1150} y={640} size={24} color={C.muted} w={640} o={eseg(p, 0.75, 0.85)} style={{ textAlign: 'center' }}>{t('watch.dilated.label.test_time')}</At>
    </>
  )
}

function Recap({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.3} />
      </Svg>
      <At x={800} y={150} size={30} color={C.blue} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>{t('watch.ui.recap')}</At>
      {[1, 2, 3].map((k) => (
        <At key={k} x={220} y={240 + k * 120} size={36} anchor="l" w={1180} o={eseg(p, 0.08 + k * 0.12, 0.16 + k * 0.12)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.dilated.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const dilated: Episode = {
  id: 'dilated',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'dilation', dur: 13000, cues: [0, 0.4, 0.6], render: (p) => <DilationScene p={p} /> },
    { id: 'formula', dur: 15000, cues: [0, 0.2, 0.4, 0.65], ponder: true, render: (p) => <FormulaScene p={p} /> },
    { id: 'shift', dur: 24000, cues: [0, 0.15, 0.45, 0.82], render: (p) => <ShiftAddScene p={p} /> },
    { id: 'todos', dur: 16000, cues: [0, 0.38, 0.68], render: (p) => <TodoScene p={p} /> },
    { id: 'dropout', dur: 15000, cues: [0, 0.25, 0.55, 0.75], render: (p) => <DropoutScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
