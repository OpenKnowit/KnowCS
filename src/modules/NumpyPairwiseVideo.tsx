import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { AnimatePresence, m } from 'framer-motion'
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { EXAM_TEST, EXAM_TRAIN, kNearest, pairwiseDiff, pairwiseDist, pairwiseSqDist, shapeTrace, sumShape } from '../lib/pairwise'
import { runPython } from '../lib/minipy'

// --- 讲解视频：用 2022S 期中 Q2(c) 原题数据，逐场景演示一行 NumPy 成对距离为什么成立 ---
// 视频由时间轴驱动：每个场景按本地进度 p ∈ [0, 1] 决定显示什么，字幕按 cue 切换。

const P = 'numpy_module.pairwise'

interface Scene {
  id: 'question' | 'data' | 'line' | 'fail' | 'axes' | 'broadcast' | 'reduce' | 'result' | 'recap'
  /** 时长（毫秒，1× 速度） */
  dur: number
  /** 字幕切换点（场景内进度），第 k 条字幕键为 c{k+1} */
  cues: number[]
}

const SCENES: Scene[] = [
  { id: 'question', dur: 13000, cues: [0, 0.36, 0.72] },
  { id: 'data', dur: 13000, cues: [0, 0.3, 0.66] },
  { id: 'line', dur: 12000, cues: [0, 0.5] },
  { id: 'fail', dur: 14000, cues: [0, 0.3, 0.64] },
  { id: 'axes', dur: 15000, cues: [0, 0.3, 0.64] },
  { id: 'broadcast', dur: 16000, cues: [0, 0.32, 0.64] },
  { id: 'reduce', dur: 17000, cues: [0, 0.3, 0.6] },
  { id: 'result', dur: 15000, cues: [0, 0.32, 0.7] },
  { id: 'recap', dur: 15000, cues: [0, 0.36, 0.72] },
]
const STARTS = SCENES.reduce<number[]>((acc, _s, i) => [...acc, i === 0 ? 0 : acc[i - 1] + SCENES[i - 1].dur], [])
const TOTAL = STARTS[STARTS.length - 1] + SCENES[SCENES.length - 1].dur
const SPEEDS = [0.75, 1, 1.25, 1.5]

const N = EXAM_TEST.length
const M = EXAM_TRAIN.length
const D_DIM = EXAM_TEST[0].length
const DIFF = pairwiseDiff(EXAM_TEST, EXAM_TRAIN)
const SQ = pairwiseSqDist(EXAM_TEST, EXAM_TRAIN)
const DIST = pairwiseDist(EXAM_TEST, EXAM_TRAIN)
const NEAREST = kNearest(DIST, 2)
const TRACE = shapeTrace(N, M, D_DIM)

const ONE_LINER = 'np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=-1))'
const TRY_CODE = `import numpy as np
X_train = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
X_test = np.array([[5, 6], [7, 8]])

diff = X_test[:, None, :] - X_train[None, :, :]
print(diff.shape)          # (2, 4, 2)
dists = np.sqrt((diff ** 2).sum(axis=-1))
print(dists.shape)         # (2, 4)
print(np.argsort(dists, axis=1)[:, :2])
dists`

/** 场景内进度 p 在 [a, b] 区间里走到了哪里 */
const seg = (p: number, a: number, b: number) => Math.min(1, Math.max(0, (p - a) / (b - a)))
const fmtTime = (ms: number) => {
  const s = Math.floor(ms / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}
/** 最后一个满足 xs[k] <= v 的下标（xs 升序） */
const lastAtOrBefore = (xs: readonly number[], v: number) => xs.reduce((best, x, k) => (x <= v ? k : best), 0)
const shp = (s: readonly number[]) => `(${s.join(', ')}${s.length === 1 ? ',' : ''})`
const vec = (v: readonly number[]) => `(${v.join(', ')})`

const TONE = {
  test: 'text-sky-300',
  train: 'text-orange-300',
  res: 'text-emerald-300',
  testPill: 'bg-sky-500/20 border-sky-400/60 text-sky-200',
  trainPill: 'bg-orange-500/20 border-orange-400/60 text-orange-200',
  resPill: 'bg-emerald-500/25 border-emerald-400/70 text-emerald-100',
}

// ---------- 小部件 ----------

const Pill = ({ children, cls, ghost }: { children: ReactNode; cls: string; ghost?: boolean }) => (
  <span className={`inline-flex items-center justify-center rounded-md border px-1.5 py-0.5 font-mono text-[11px] sm:text-xs whitespace-nowrap transition-all duration-300 ${cls} ${ghost ? 'border-dashed opacity-50' : ''}`}>
    {children}
  </span>
)

/** 形状标签，hl 指定要高亮（新插入的 1）的位置 */
const Shape = ({ shape, hl = [], cls = 'text-slate-200' }: { shape: readonly number[]; hl?: number[]; cls?: string }) => (
  <span className={`font-mono ${cls}`}>
    (
    {shape.map((d, i) => (
      <span key={i}>
        {i > 0 && ', '}
        <span className={hl.includes(i) ? 'rounded bg-yellow-400/90 px-1 text-slate-900 font-bold' : ''}>{d}</span>
      </span>
    ))}
    )
  </span>
)

const Fade = ({ show, children, className = '' }: { show: boolean; children: ReactNode; className?: string }) => (
  <div className={`transition-all duration-500 ${show ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2 pointer-events-none'} ${className}`}>{children}</div>
)

const Tag = ({ children, tone }: { children: ReactNode; tone: 'amber' | 'rose' | 'emerald' | 'slate' }) => {
  const cls = {
    amber: 'bg-amber-400/15 border-amber-400/50 text-amber-200',
    rose: 'bg-rose-500/15 border-rose-400/60 text-rose-200',
    emerald: 'bg-emerald-500/15 border-emerald-400/60 text-emerald-200',
    slate: 'bg-slate-700/60 border-slate-500/60 text-slate-200',
  }[tone]
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] sm:text-xs font-semibold ${cls}`}>{children}</span>
}

const CodeBox = ({ children, className = '' }: { children: ReactNode; className?: string }) => (
  <pre className={`rounded-lg bg-black/40 border border-slate-700 px-3 py-2 font-mono text-[10.5px] sm:text-xs leading-5 text-slate-200 whitespace-pre overflow-x-auto ${className}`}>{children}</pre>
)

// 散点图：原题 6 个点恰好都在 y = x + 1 上，连线会重叠，所以画成弧线
const Scatter = ({ pairs, nearest }: { pairs: number; nearest?: boolean }) => {
  const S = 22
  const X = (v: number) => 16 + v * S
  const Y = (v: number) => 196 - v * S
  return (
    <svg viewBox="0 0 210 210" className="w-full max-w-[210px]" role="img" aria-label="scatter">
      {Array.from({ length: 9 }, (_, v) => (
        <g key={v} className="text-slate-700">
          <line x1={X(v)} x2={X(v)} y1={Y(0)} y2={Y(8)} stroke="currentColor" strokeWidth={0.5} />
          <line x1={X(0)} x2={X(8)} y1={Y(v)} y2={Y(v)} stroke="currentColor" strokeWidth={0.5} />
        </g>
      ))}
      {EXAM_TEST.flatMap((a, i) =>
        EXAM_TRAIN.map((b, j) => {
          const k = i * M + j
          if (k >= pairs) return null
          const [x1, y1, x2, y2] = [X(a[0]), Y(a[1]), X(b[0]), Y(b[1])]
          const len = Math.hypot(x2 - x1, y2 - y1)
          const bow = (8 + j * 7) * (i === 0 ? 1 : -1)
          const cx = (x1 + x2) / 2 + (-(y2 - y1) / len) * bow
          const cy = (y1 + y2) / 2 + ((x2 - x1) / len) * bow
          const hot = nearest && NEAREST[i][0] === j
          return (
            <path
              key={k}
              d={`M${x1},${y1} Q${cx},${cy} ${x2},${y2}`}
              fill="none"
              stroke={hot ? '#34d399' : i === 0 ? '#7dd3fc' : '#c4b5fd'}
              strokeWidth={hot ? 2.2 : 1}
              strokeOpacity={nearest && !hot ? 0.2 : 0.8}
              strokeDasharray={hot ? undefined : '3 2'}
            />
          )
        }),
      )}
      {EXAM_TRAIN.map((b, j) => (
        <g key={`r${j}`}>
          <circle cx={X(b[0])} cy={Y(b[1])} r={5} fill="#fb923c" />
          <text x={X(b[0]) + 7} y={Y(b[1]) + 10} fontSize={9} fill="#fdba74" fontFamily="monospace">{`r${j}`}</text>
        </g>
      ))}
      {EXAM_TEST.map((a, i) => (
        <g key={`t${i}`}>
          <rect x={X(a[0]) - 5} y={Y(a[1]) - 5} width={10} height={10} rx={2} fill="#38bdf8" />
          <text x={X(a[0]) - 8} y={Y(a[1]) - 8} fontSize={9} fill="#7dd3fc" fontFamily="monospace">{`t${i}`}</text>
        </g>
      ))}
    </svg>
  )
}

/** 2 × 4 网格：行 = 测试点 i，列 = 训练点 j */
const PairGrid = ({ cell, corner }: { cell: (i: number, j: number) => ReactNode; corner?: ReactNode }) => (
  <div className="grid gap-1 sm:gap-1.5 items-center" style={{ gridTemplateColumns: `auto repeat(${M}, minmax(0, 1fr))` }}>
    <div className="text-[10px] text-slate-500 font-mono text-center">{corner}</div>
    {EXAM_TRAIN.map((b, j) => (
      <div key={j} className="flex flex-col items-center gap-0.5">
        <span className="text-[10px] font-mono text-orange-300/80">{`j=${j}`}</span>
        <Pill cls={TONE.trainPill}>{vec(b)}</Pill>
      </div>
    ))}
    {EXAM_TEST.map((a, i) => [
      <div key={`h${i}`} className="flex items-center gap-1 pr-1">
        <span className="text-[10px] font-mono text-sky-300/80">{`i=${i}`}</span>
        <Pill cls={TONE.testPill}>{vec(a)}</Pill>
      </div>,
      ...EXAM_TRAIN.map((_, j) => (
        <div key={`${i}-${j}`} className="min-h-[3.25rem] rounded-lg border border-slate-700 bg-slate-800/60 flex flex-col items-center justify-center gap-0.5 p-1">
          {cell(i, j)}
        </div>
      )),
    ])}
  </div>
)

// ---------- 场景 ----------

const SceneQuestion = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const struck = p > 0.4
  return (
    <div className="flex flex-col items-center gap-3 w-full max-w-2xl">
      <div className="flex flex-wrap justify-center gap-2">
        <Tag tone="amber">{t(`${P}.v.question.badge`)}</Tag>
        <Fade show={p > 0.36}><Tag tone="rose">{t(`${P}.v.question.zero`)}</Tag></Fade>
      </div>
      <div className="relative w-full">
        <CodeBox className={struck ? 'opacity-60' : ''}>
          <span className="text-slate-400">{'def compute_distances_nested_loops(X_train, X_test):\n'}</span>
          {'    num_test = X_test.shape[0]\n    num_train = X_train.shape[0]\n    distances = np.zeros((num_test, num_train))\n'}
          <span className="text-amber-300/80">{'    # --- BLOCK TO REWRITE ---\n'}</span>
          <span className={struck ? 'line-through decoration-rose-400 decoration-2' : ''}>
            {'    for i in range(num_test):\n        for j in range(num_train):\n            distances[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))\n'}
          </span>
          <span className="text-amber-300/80">{'    # --- BLOCK TO REWRITE ---\n'}</span>
          {'    return distances'}
        </CodeBox>
      </div>
      <Fade show={p > 0.66} className="w-full">
        <div className="rounded-lg border border-emerald-400/60 bg-emerald-500/10 px-3 py-2">
          <div className="text-[11px] text-emerald-300 font-semibold mb-1">{t(`${P}.v.question.one_line`)}</div>
          <code className="block font-mono text-[11px] sm:text-[13px] text-emerald-100 break-words">distances = {ONE_LINER}</code>
        </div>
      </Fade>
    </div>
  )
}

const SceneData = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const shown = Math.floor(seg(p, 0.28, 0.78) * (N * M + 0.999))
  return (
    <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-8 w-full">
      <div className="flex flex-col items-center gap-2">
        <Scatter pairs={shown} />
        <div className="flex gap-3 text-[11px] font-mono">
          <span className={TONE.test}>■ X_test {shp([N, D_DIM])}</span>
          <span className={TONE.train}>● X_train {shp([M, D_DIM])}</span>
        </div>
      </div>
      <div className="flex flex-col items-center gap-2 w-full max-w-sm">
        <div className="text-xs text-slate-300">{t(`${P}.v.data.cell`)}</div>
        <PairGrid
          corner="D"
          cell={(i, j) => (i * M + j < shown ? <span className="font-mono text-xs sm:text-sm text-emerald-200">{DIST[i][j].toFixed(2)}</span> : <span className="text-slate-600">?</span>)}
        />
        <Fade show={p > 0.66}>
          <Tag tone="slate">{t(`${P}.v.data.shape`)} <Shape shape={[N, M]} cls="text-emerald-300" /></Tag>
        </Fade>
      </div>
    </div>
  )
}

const LINE_PARTS: { code: string; step: number | null }[] = [
  { code: 'np.sqrt(', step: 5 },
  { code: '((', step: null },
  { code: 'X_test[:, None, :]', step: 1 },
  { code: ' - ', step: 2 },
  { code: 'X_train[None, :, :]', step: 1 },
  { code: ') ** 2)', step: 3 },
  { code: '.sum(axis=-1)', step: 4 },
  { code: ')', step: 5 },
]
const STEP_SHAPES: { step: number; shape: string }[] = [
  { step: 1, shape: `${shp(TRACE[0].shape)} & ${shp(TRACE[1].shape)}` },
  { step: 2, shape: shp(TRACE[2].shape) },
  { step: 3, shape: shp(TRACE[3].shape) },
  { step: 4, shape: shp(TRACE[4].shape) },
  { step: 5, shape: shp(TRACE[5].shape) },
]
const STEP_CLS = ['', 'text-yellow-300', 'text-sky-300', 'text-fuchsia-300', 'text-emerald-300', 'text-orange-300']

const SceneLine = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const active = Math.min(5, Math.floor(seg(p, 0.08, 0.85) * 5) + 1)
  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-2xl">
      <div className="rounded-xl bg-black/40 border border-slate-700 px-3 py-3 font-mono text-[12px] sm:text-[15px] leading-7 text-center">
        <span className="text-slate-400 whitespace-pre">D = </span>
        {LINE_PARTS.map((part, k) => (
          <span
            key={k}
            className={`whitespace-pre transition-all duration-300 ${part.step === null ? 'text-slate-400' : part.step <= active ? `${STEP_CLS[part.step]} ${part.step === active ? 'underline decoration-2 underline-offset-4' : ''}` : 'text-slate-500'}`}
          >
            {part.code}
          </span>
        ))}
      </div>
      <ol className="w-full grid grid-cols-1 sm:grid-cols-[1.7fr_1fr_1fr_1fr_1fr] gap-1.5">
        {STEP_SHAPES.map(({ step, shape }) => (
          <li
            key={step}
            className={`rounded-lg border px-2 py-1.5 text-center transition-all duration-500 ${step <= active ? 'border-slate-600 bg-slate-800/80 opacity-100' : 'border-slate-800 opacity-25'} ${step === active ? 'ring-1 ring-yellow-300/70' : ''}`}
          >
            <div className={`text-xs font-semibold ${STEP_CLS[step]}`}>{`${'①②③④⑤'[step - 1]} ${t(`${P}.v.line.s${step}`)}`}</div>
            <div className="font-mono text-[11px] text-slate-300">{shape}</div>
          </li>
        ))}
      </ol>
    </div>
  )
}

const SceneFail = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const rows: { axis: number; a: number; b: number; ok: boolean; at: number }[] = [
    { axis: -1, a: D_DIM, b: D_DIM, ok: true, at: 0.12 },
    { axis: -2, a: N, b: M, ok: false, at: 0.3 },
  ]
  return (
    <div className="flex flex-col items-center gap-3 w-full max-w-xl">
      <code className="font-mono text-base sm:text-lg"><span className={TONE.test}>X_test</span> - <span className={TONE.train}>X_train</span></code>
      <div className="grid grid-cols-[auto_auto] gap-x-4 gap-y-1 font-mono text-sm items-center">
        <span className={`text-xs ${TONE.test}`}>X_test</span>
        <Shape shape={[N, D_DIM]} cls={TONE.test} />
        <span className={`text-xs ${TONE.train}`}>X_train</span>
        <Shape shape={[M, D_DIM]} cls={TONE.train} />
      </div>
      <div className="space-y-1.5 w-full max-w-sm">
        {rows.map((r) => (
          <Fade key={r.axis} show={p > r.at}>
            <div className={`flex items-center justify-between rounded-md px-3 py-1 text-sm font-mono ${r.ok ? 'bg-slate-800 text-slate-200' : 'bg-rose-500/20 text-rose-200'}`}>
              <span>axis {r.axis}</span>
              <span>{r.a} vs {r.b}</span>
              <span>{r.ok ? '✓' : '✗'}</span>
            </div>
          </Fade>
        ))}
      </div>
      <Fade show={p > 0.38} className="w-full">
        <CodeBox className="!text-rose-300 !border-rose-500/50">{`ValueError: operands could not be broadcast\ntogether with shapes (${N},${D_DIM}) (${M},${D_DIM})`}</CodeBox>
      </Fade>
      <Fade show={p > 0.64}>
        <div className="rounded-lg border border-amber-400/50 bg-amber-400/10 px-3 py-2 text-xs sm:text-sm text-amber-100 max-w-xl">
          {t(`${P}.v.fail.silent`)}
        </div>
      </Fade>
    </div>
  )
}

const SceneAxes = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const lifted = p > 0.18
  return (
    <div className="flex flex-col items-center gap-3 w-full">
      <div className="grid grid-cols-2 gap-3 sm:gap-8 w-full max-w-2xl">
        {/* 测试点：竖成一列 */}
        <div className="flex flex-col items-center gap-2">
          <code className="font-mono text-[11px] sm:text-sm">
            <span className={TONE.test}>X_test</span>
            <span className={`transition-colors duration-500 ${lifted ? 'text-yellow-300' : 'text-transparent'}`}>[:, None, :]</span>
          </code>
          <div className="text-xs">{lifted ? <Shape shape={TRACE[0].shape} hl={[1]} /> : <Shape shape={[N, D_DIM]} />}</div>
          <div className="flex flex-col gap-1.5 p-2 rounded-lg border border-dashed border-sky-400/40">
            {EXAM_TEST.map((a, i) => <Pill key={i} cls={TONE.testPill}>{vec(a)}</Pill>)}
          </div>
          <Fade show={lifted}><span className="text-[11px] text-sky-300">↓ {t(`${P}.v.axes.axis0`)}</span></Fade>
        </div>
        {/* 训练点：横成一行 */}
        <div className="flex flex-col items-center gap-2">
          <code className="font-mono text-[11px] sm:text-sm">
            <span className={TONE.train}>X_train</span>
            <span className={`transition-colors duration-500 ${p > 0.34 ? 'text-yellow-300' : 'text-transparent'}`}>[None, :, :]</span>
          </code>
          <div className="text-xs">{p > 0.34 ? <Shape shape={TRACE[1].shape} hl={[0]} /> : <Shape shape={[M, D_DIM]} />}</div>
          <div className={`flex gap-1.5 p-2 rounded-lg border border-dashed border-orange-400/40 transition-all duration-700 ${p > 0.34 ? 'flex-row flex-wrap justify-center' : 'flex-col'}`}>
            {EXAM_TRAIN.map((b, j) => <Pill key={j} cls={TONE.trainPill}>{vec(b)}</Pill>)}
          </div>
          <Fade show={p > 0.34}><span className="text-[11px] text-orange-300">→ {t(`${P}.v.axes.axis1`)}</span></Fade>
        </div>
      </div>
      <Fade show={p > 0.5}><span className="text-[11px] text-slate-300">( · , · ) = {t(`${P}.v.axes.axis2`)}</span></Fade>
      <Fade show={p > 0.64} className="w-full max-w-2xl">
        <div className="rounded-lg border border-slate-600 bg-slate-800/70 px-3 py-2 text-[11px] sm:text-xs font-mono text-slate-200 grid gap-1">
          <div><span className="text-yellow-300">X_test[:, None, :]</span> ≡ X_test[:, np.newaxis] ≡ np.expand_dims(X_test, 1)</div>
          <div><span className="text-yellow-300">X_train[None, :, :]</span> ≡ X_train[None] ≡ X_train <span className="font-sans text-slate-400">({t(`${P}.v.axes.pad`)})</span></div>
        </div>
      </Fade>
    </div>
  )
}

const SceneBroadcast = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const rules: { axis: number; a: number; b: number; verdict: 'equal' | 'stretch_test' | 'stretch_train'; at: number }[] = [
    { axis: -1, a: D_DIM, b: D_DIM, verdict: 'equal', at: 0.04 },
    { axis: -2, a: 1, b: M, verdict: 'stretch_test', at: 0.13 },
    { axis: -3, a: N, b: 1, verdict: 'stretch_train', at: 0.22 },
  ]
  const copies = p > 0.34
  const filled = Math.floor(seg(p, 0.6, 0.9) * (N * M + 0.999))
  return (
    <div className="flex flex-col lg:flex-row items-center justify-center gap-3 lg:gap-6 w-full">
      <div className="flex flex-col items-center gap-1.5 shrink-0">
        <table className="font-mono text-xs border-separate border-spacing-x-2 border-spacing-y-0.5">
          <tbody>
            <tr><td className={TONE.test}>test</td>{[N, 1, D_DIM].map((d, k) => <td key={k} className="text-center">{d}</td>)}</tr>
            <tr><td className={TONE.train}>train</td>{[1, M, D_DIM].map((d, k) => <td key={k} className="text-center">{d}</td>)}</tr>
            <tr className="text-emerald-300">
              <td>out</td>
              {[...rules].reverse().map((r) => <td key={r.axis} className="text-center">{p > r.at ? Math.max(r.a, r.b) : '?'}</td>)}
            </tr>
          </tbody>
        </table>
        <div className="flex flex-col gap-0.5 w-full">
          {rules.map((r) => (
            <Fade key={r.axis} show={p > r.at}>
              <div className="text-[10.5px] text-slate-300 font-mono">axis {r.axis}: {r.a} vs {r.b} → <span className={r.verdict === 'equal' ? 'text-slate-200' : r.verdict === 'stretch_test' ? TONE.test : TONE.train}>{t(`${P}.v.broadcast.${r.verdict}`)}</span></div>
            </Fade>
          ))}
        </div>
        <Fade show={p > 0.3}><Tag tone="emerald">→ <Shape shape={TRACE[2].shape} cls="text-emerald-200" /></Tag></Fade>
      </div>
      <div className="w-full max-w-md">
        <PairGrid
          corner="−"
          cell={(i, j) => {
            const k = i * M + j
            if (k < filled) {
              return (
                <m.div initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="flex flex-col items-center">
                  <Pill cls={TONE.resPill}>{vec(DIFF[i][j])}</Pill>
                  <span className="text-[9px] text-slate-500 font-mono mt-0.5">{`t${i}−r${j}`}</span>
                </m.div>
              )
            }
            if (!copies) return <span className="text-slate-700">·</span>
            return (
              <>
                <Pill cls={TONE.testPill} ghost={j > 0}>{vec(EXAM_TEST[i])}</Pill>
                <Pill cls={TONE.trainPill} ghost={i > 0}>{vec(EXAM_TRAIN[j])}</Pill>
              </>
            )
          }}
        />
        <Fade show={copies && filled === 0} className="mt-1.5 text-center text-[10.5px] text-slate-400">{t(`${P}.v.broadcast.ghost`)}</Fade>
      </div>
    </div>
  )
}

const SceneReduce = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const squared = p > 0.08
  const summed = p > 0.34
  const axes: { axis: number; key: string; ok: boolean }[] = [
    { axis: -1, key: 'ax_last', ok: true },
    { axis: 1, key: 'ax_1', ok: false },
    { axis: 0, key: 'ax_0', ok: false },
  ]
  return (
    <div className="flex flex-col lg:flex-row items-center justify-center gap-3 lg:gap-6 w-full">
      <div className="w-full max-w-md">
        <PairGrid
          corner={summed ? 'Σ' : '²'}
          cell={(i, j) => {
            const dv = DIFF[i][j]
            if (summed) {
              return (
                <m.div key="sum" initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="flex flex-col items-center">
                  <span className="font-mono text-sm text-emerald-200 font-semibold">{SQ[i][j]}</span>
                  <span className="text-[9px] text-slate-500 font-mono">{dv.map((x) => x * x).join('+')}</span>
                </m.div>
              )
            }
            return <Pill cls={squared ? 'bg-fuchsia-500/20 border-fuchsia-400/60 text-fuchsia-100' : TONE.resPill}>{vec(squared ? dv.map((x) => x * x) : dv)}</Pill>
          }}
        />
        <div className="mt-1.5 flex justify-center">
          <Tag tone={summed ? 'emerald' : 'slate'}>
            {summed ? <>.sum(axis=-1): <Shape shape={TRACE[3].shape} /> → <Shape shape={TRACE[4].shape} cls="text-emerald-200" /></> : <>** 2: <Shape shape={TRACE[3].shape} /></>}
          </Tag>
        </div>
      </div>
      <Fade show={p > 0.6} className="w-full max-w-xs">
        <div className="space-y-1.5">
          {axes.map((a) => (
            <div key={a.axis} className={`flex items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-xs ${a.ok ? 'bg-emerald-500/15 text-emerald-200' : 'bg-rose-500/15 text-rose-200'}`}>
              <code className="font-mono">{a.axis === -1 ? 'axis=-1 / 2' : `axis=${a.axis}`}</code>
              <Shape shape={sumShape(TRACE[3].shape, a.axis)} cls="" />
              <span className="text-[10.5px] text-right">{t(`${P}.v.reduce.${a.key}`)}</span>
            </div>
          ))}
          <div className="rounded-md border border-amber-400/50 bg-amber-400/10 px-2.5 py-1.5 text-[11px] text-amber-100">{t(`${P}.v.reduce.trap`)}</div>
        </div>
      </Fade>
    </div>
  )
}

const SceneResult = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const nearest = p > 0.32
  return (
    <div className="flex flex-col md:flex-row items-center justify-center gap-3 md:gap-8 w-full">
      <div className="flex flex-col items-center gap-1">
        <Scatter pairs={N * M} nearest={nearest} />
      </div>
      <div className="flex flex-col items-center gap-2 w-full max-w-md">
        <PairGrid
          corner="√"
          cell={(i, j) => {
            const hot = nearest && NEAREST[i][0] === j
            return <span className={`font-mono text-xs sm:text-sm whitespace-nowrap ${hot ? 'text-emerald-300 font-bold' : 'text-slate-200'}`}>{DIST[i][j].toFixed(2)}{hot && <span className="text-[10px]">★</span>}</span>
          }}
        />
        <Fade show={nearest} className="w-full">
          <CodeBox>
            <span className="text-slate-400">{'>>> '}</span>{'np.argsort(D, axis=1)[:, :2]\n'}
            <span className="text-emerald-300">{`[${NEAREST.map((r) => `[${r.join(' ')}]`).join('\n ')}]`}</span>
          </CodeBox>
        </Fade>
        <Fade show={p > 0.7}><Tag tone="emerald">✓ {t(`${P}.v.result.match`)}</Tag></Fade>
      </div>
    </div>
  )
}

const CHECKS = ['no_loop', 'axis', 'square', 'sum', 'order', 'general'] as const

const SceneRecap = ({ p }: { p: number }) => {
  const { t } = useTranslation()
  const rows = Math.floor(seg(p, 0.03, 0.33) * (TRACE.length + 0.999))
  const checks = Math.floor(seg(p, 0.36, 0.7) * (CHECKS.length + 0.999))
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-6 w-full max-w-3xl">
      <div className="rounded-lg bg-black/40 border border-slate-700 p-2.5 font-mono text-[11px] sm:text-xs space-y-1">
        {TRACE.map((s, k) => (
          <div key={s.id} className={`flex justify-between gap-3 transition-opacity duration-500 ${k < rows ? 'opacity-100' : 'opacity-0'}`}>
            <span className="text-slate-300 truncate">{s.code}</span>
            <span className="text-emerald-300 shrink-0"># {shp(s.shape)}</span>
          </div>
        ))}
      </div>
      <ul className="space-y-1">
        {CHECKS.map((c, k) => (
          <li key={c} className={`flex gap-2 text-[11.5px] sm:text-xs text-slate-200 transition-all duration-500 ${k < checks ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-2'}`}>
            <span className="text-emerald-400">✓</span>
            <span>{t(`${P}.v.recap.${c}`)}</span>
          </li>
        ))}
      </ul>
      <Fade show={p > 0.72} className="md:col-span-2">
        <div className="rounded-lg border border-sky-400/40 bg-sky-500/10 px-3 py-2 text-[11px] sm:text-xs text-sky-100">{t(`${P}.v.recap.memory`)}</div>
      </Fade>
    </div>
  )
}

const RENDER: Record<Scene['id'], (props: { p: number }) => ReactNode> = {
  question: SceneQuestion,
  data: SceneData,
  line: SceneLine,
  fail: SceneFail,
  axes: SceneAxes,
  broadcast: SceneBroadcast,
  reduce: SceneReduce,
  result: SceneResult,
  recap: SceneRecap,
}

// ---------- 播放器 ----------

export const NumpyPairwiseVideo = () => {
  const { t } = useTranslation()
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const timeRef = useRef(0)

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      timeRef.current = Math.min(TOTAL, timeRef.current + (now - last) * speed)
      last = now
      setTime(timeRef.current)
      if (timeRef.current >= TOTAL) {
        setPlaying(false)
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed])

  const seek = (ms: number) => {
    timeRef.current = Math.max(0, Math.min(TOTAL, ms))
    setTime(timeRef.current)
  }
  const toggle = () => {
    if (!playing && timeRef.current >= TOTAL) seek(0)
    setPlaying(!playing)
  }

  const idx = Math.min(SCENES.length - 1, lastAtOrBefore(STARTS, time))
  const scene = SCENES[idx]
  const p = Math.min(1, (time - STARTS[idx]) / scene.dur)
  const cue = lastAtOrBefore(scene.cues, p)
  const Render = RENDER[scene.id]
  const ended = time >= TOTAL

  const goScene = (k: number) => seek(STARTS[Math.max(0, Math.min(SCENES.length - 1, k))])
  // 「上一段」：已播放超过 1.5 秒则回到本段开头，否则跳到前一段（与常见播放器一致）
  const prev = () => goScene(time - STARTS[idx] > 1500 ? idx : idx - 1)
  const onKey = (e: KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'k') toggle()
    else if (e.key === 'ArrowRight') goScene(idx + 1)
    else if (e.key === 'ArrowLeft') prev()
    else return
    e.preventDefault()
  }

  const run = useMemo(() => runPython(TRY_CODE), [])

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 leading-relaxed">{t(`${P}.intro`)}</p>

      <div
        className="rounded-2xl overflow-hidden shadow-lg bg-slate-900 outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
        tabIndex={0}
        onKeyDown={onKey}
        role="region"
        aria-label={t(`${P}.video_label`)}
      >
        {/* 画面 */}
        <div className="relative text-slate-100 min-h-[45rem] sm:min-h-[38rem] md:min-h-[33rem] flex flex-col">
          <div className="flex items-center justify-between gap-2 px-3 sm:px-4 pt-3">
            <div className="flex items-center gap-2 min-w-0">
              <span className="shrink-0 rounded bg-slate-700 px-1.5 py-0.5 text-[10px] font-mono text-slate-300">{idx + 1}/{SCENES.length}</span>
              <span className="truncate text-sm font-semibold">{t(`${P}.scenes.${scene.id}.title`)}</span>
            </div>
            <span className="hidden sm:block shrink-0 font-mono text-[10px] text-slate-500">COMP2211 · NumPy</span>
          </div>

          <div className="flex-1 flex items-center justify-center px-3 sm:px-6 py-3 cursor-pointer" onClick={toggle}>
            <AnimatePresence mode="wait">
              <m.div
                key={scene.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.35 }}
                className="w-full flex justify-center"
              >
                <Render p={p} />
              </m.div>
            </AnimatePresence>
          </div>

          {/* 字幕 */}
          <div className="px-3 sm:px-6 pb-3">
            <m.p
              key={`${scene.id}-${cue}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              aria-live="polite"
              className="mx-auto max-w-3xl rounded-lg bg-black/70 px-3 py-2 text-center text-[13px] sm:text-sm leading-relaxed text-white"
            >
              {t(`${P}.scenes.${scene.id}.c${cue + 1}`)}
            </m.p>
          </div>

          {/* 开场 / 结束遮罩 */}
          {!playing && (time === 0 || ended) && (
            <button
              onClick={toggle}
              className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/75 backdrop-blur-[2px] text-white"
            >
              <span className="flex h-16 w-16 items-center justify-center rounded-full bg-blue-600 shadow-xl hover:bg-blue-500 transition">
                {ended ? <RotateCcw size={28} /> : <Play size={30} className="ml-1" />}
              </span>
              <span className="text-base font-semibold">{ended ? t(`${P}.replay`) : t(`${P}.play_video`)}</span>
              <span className="text-xs text-slate-300 font-mono">{fmtTime(TOTAL / speed)} · {t(`${P}.scene_count`, { count: SCENES.length })}</span>
            </button>
          )}
        </div>

        {/* 时间轴：按场景分段，点击跳转 */}
        <div className="flex gap-0.5 px-3 sm:px-4 pt-2 bg-slate-950">
          {SCENES.map((s, k) => {
            const fill = k < idx ? 1 : k > idx ? 0 : p
            return (
              <button
                key={s.id}
                onClick={() => goScene(k)}
                title={t(`${P}.scenes.${s.id}.title`)}
                aria-label={t(`${P}.scenes.${s.id}.title`)}
                className="group relative h-3 flex items-center"
                style={{ flexGrow: s.dur, flexBasis: 0 }}
              >
                <span className="relative block h-1.5 w-full overflow-hidden rounded-full bg-slate-700 group-hover:h-2 transition-all">
                  <span className="absolute inset-y-0 left-0 bg-blue-500" style={{ width: `${fill * 100}%` }} />
                </span>
              </button>
            )
          })}
        </div>

        {/* 控制条 */}
        <div className="flex flex-wrap items-center gap-1 sm:gap-2 px-2 sm:px-3 py-2 bg-slate-950 text-slate-200">
          <button onClick={toggle} aria-label={playing ? t(`${P}.pause`) : t(`${P}.play`)} className="p-2 rounded-lg hover:bg-slate-800">
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button onClick={prev} aria-label={t(`${P}.prev`)} className="p-2 rounded-lg hover:bg-slate-800"><SkipBack size={16} /></button>
          <button onClick={() => goScene(idx + 1)} disabled={idx === SCENES.length - 1} aria-label={t(`${P}.next`)} className="p-2 rounded-lg hover:bg-slate-800 disabled:opacity-30"><SkipForward size={16} /></button>
          <span className="font-mono text-xs text-slate-400 tabular-nums">{fmtTime(time)} / {fmtTime(TOTAL)}</span>
          <div className="ml-auto flex items-center gap-1" role="group" aria-label={t(`${P}.speed`)}>
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                aria-pressed={speed === s}
                className={`px-1.5 py-1 rounded text-[11px] font-mono ${speed === s ? 'bg-slate-200 text-slate-900' : 'text-slate-400 hover:text-white'}`}
              >
                {s}×
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 章节 */}
      <ol className="flex flex-wrap gap-1.5">
        {SCENES.map((s, k) => (
          <li key={s.id}>
            <button
              onClick={() => goScene(k)}
              aria-current={k === idx ? 'step' : undefined}
              className={`px-2.5 py-1 rounded-lg text-xs transition ${k === idx ? 'bg-slate-800 text-white' : 'bg-white border border-gray-200 text-gray-600 hover:border-gray-400'}`}
            >
              <span className="font-mono text-[10px] opacity-60 mr-1">{fmtTime(STARTS[k])}</span>
              {t(`${P}.scenes.${s.id}.title`)}
            </button>
          </li>
        ))}
      </ol>
      <p className="text-[11px] text-gray-400">{t(`${P}.keys`)}</p>

      {/* 亲手跑一遍（站内解释器真实运行） */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="rounded-xl bg-slate-900 p-3 min-w-0">
          <div className="text-[11px] text-slate-400 mb-1">{t(`${P}.code_title`)}</div>
          <pre className="font-mono text-[12px] leading-5 text-slate-100 whitespace-pre overflow-x-auto">{TRY_CODE}</pre>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-3 min-w-0">
          <div className="text-[11px] text-gray-400 mb-1">{t(`${P}.output_title`)}</div>
          <pre className="font-mono text-[12px] leading-5 text-sky-700 whitespace-pre overflow-x-auto">{run.error ? `${run.error.type}: ${run.error.message}` : `${run.stdout}${run.out ?? ''}`}</pre>
        </div>
      </div>
    </div>
  )
}
