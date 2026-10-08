import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  FileText,
  GitBranch,
  Layers,
  ListOrdered,
  Pause,
  Play,
  RotateCcw,
  Shuffle,
  SlidersHorizontal,
} from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { AB_LEAF_IDS, AB_NODES, AB_PRESETS } from '../data/constants'
import { generateAbSteps } from '../lib/alphabeta'
import type { AbNode, AbStep } from '../types'

const fmt = (v: number): string => (v === Infinity ? '∞' : v === -Infinity ? '-∞' : String(v))

const EDGES: { from: string; to: string }[] = Object.values(AB_NODES).flatMap((n) =>
  (n.children ?? []).map((c) => ({ from: n.id, to: c }))
)

export const AlphaBetaModule = () => {
  const { t } = useTranslation()
  const [leaves, setLeaves] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {}
    AB_LEAF_IDS.forEach((id, i) => (init[id] = AB_PRESETS.user[i]))
    return init
  })
  const [preset, setPreset] = useState<string>('user')
  const [stepIdx, setStepIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)

  // 把当前叶子值注入节点表
  const nodes = useMemo<Record<string, AbNode>>(() => {
    const clone: Record<string, AbNode> = {}
    for (const id in AB_NODES) clone[id] = { ...AB_NODES[id], value: AB_NODES[id].value }
    AB_LEAF_IDS.forEach((id) => (clone[id].value = leaves[id]))
    return clone
  }, [leaves])

  const steps = useMemo<AbStep[]>(() => generateAbSteps(nodes), [nodes])
  const safeIdx = Math.min(stepIdx, steps.length - 1)
  const step = steps[safeIdx]

  // 自动播放：在异步定时器回调里推进 / 停止，避免在 effect 同步体内 setState
  useEffect(() => {
    if (!playing || safeIdx >= steps.length - 1) return
    const timer = setTimeout(() => {
      const next = Math.min(safeIdx + 1, steps.length - 1)
      setStepIdx(next)
      if (next >= steps.length - 1) setPlaying(false)
    }, speed)
    return () => clearTimeout(timer)
  }, [playing, safeIdx, steps.length, speed])

  const applyPreset = (key: string) => {
    const vals = AB_PRESETS[key]
    const next: Record<string, number> = {}
    AB_LEAF_IDS.forEach((id, i) => (next[id] = vals[i]))
    setLeaves(next)
    setPreset(key)
    setStepIdx(0)
    setPlaying(false)
  }

  const randomize = () => {
    const next: Record<string, number> = {}
    AB_LEAF_IDS.forEach((id) => (next[id] = Math.floor(Math.random() * 21)))
    setLeaves(next)
    setPreset('custom')
    setStepIdx(0)
    setPlaying(false)
  }

  const setLeaf = (id: string, v: number) => {
    setLeaves((prev) => ({ ...prev, [id]: v }))
    setPreset('custom')
  }

  // 结构化步骤 → 本地化解释文案（当前步骤与追踪链共用）
  const explain = (s: AbStep): string => {
    const node = nodes[s.nodeId]
    const label = node.type === 'max' ? 'MAX' : node.type === 'min' ? 'MIN' : 'LEAF'
    switch (s.type) {
      case 'enter':
        return t('alphabeta.explain.enter', { node: s.nodeId, label, alpha: fmt(s.alpha), beta: fmt(s.beta) })
      case 'leaf_eval':
        return t('alphabeta.explain.leaf', { node: s.nodeId, value: s.leafValue, parent: node.parent })
      case 'traverse':
        return t('alphabeta.explain.traverse', { node: s.nodeId, child: s.childId, alpha: fmt(s.alpha), beta: fmt(s.beta) })
      case 'update':
        return t(node.type === 'max' ? 'alphabeta.explain.update_max' : 'alphabeta.explain.update_min', {
          node: s.nodeId,
          child: s.childId,
          result: s.childResult,
          prevBest: fmt(s.prevBest!),
          newBest: s.newBest,
          prevBound: fmt(s.prevBound!),
          newBound: s.newBound,
        })
      case 'prune':
        return t('alphabeta.explain.prune', { node: s.nodeId, alpha: fmt(s.alpha), beta: fmt(s.beta), children: s.prunedChildren!.join(', ') })
      case 'exit':
        return t('alphabeta.explain.exit', { node: s.nodeId, value: s.returnValue })
    }
  }
  const explanation = explain(step)
  // the live condition under α / β: why the search continues or cuts here
  const formula =
    step.type === 'prune'
      ? t('alphabeta.formula.prune', { alpha: fmt(step.alpha), beta: fmt(step.beta) })
      : safeIdx === 0
        ? t('alphabeta.formula.waiting')
        : safeIdx === steps.length - 1
          ? t('alphabeta.formula.done', { value: steps.at(-1)!.returnValue })
          : t('alphabeta.formula.continue', { alpha: fmt(step.alpha), beta: fmt(step.beta) })

  // keep the current step of the trace in view without scrolling the page
  const listRef = useRef<HTMLOListElement>(null)
  useEffect(() => {
    const list = listRef.current
    const item = list?.querySelector<HTMLElement>(`[data-step="${safeIdx}"]`)
    if (list && item) list.scrollTop = item.offsetTop - list.offsetTop - list.clientHeight / 2 + item.clientHeight / 2
  }, [safeIdx])

  const counts = useMemo(() => {
    const pruned = new Set(steps.at(-1)!.prunedNodes)
    const prunedLeaves = AB_LEAF_IDS.filter((id) => pruned.has(id)).length
    return { evaluated: AB_LEAF_IDS.length - prunedLeaves, prunedLeaves, total: AB_LEAF_IDS.length }
  }, [steps])

  const mathBoxClass = step.type === 'prune' ? 'border-rose-500/60 bg-rose-950/40' : step.type === 'leaf_eval' || step.type === 'update' ? 'border-emerald-500/40 bg-emerald-950/30' : 'border-slate-800 bg-slate-950'
  const panel = 'rounded-2xl border border-slate-800 bg-slate-900 shadow-xl'
  const heading = 'flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-300'
  const STEP_TONE: Record<AbStep['type'], string> = {
    enter: 'bg-indigo-500/20 text-indigo-300',
    traverse: 'bg-slate-700 text-slate-200',
    leaf_eval: 'bg-emerald-500/20 text-emerald-300',
    update: 'bg-emerald-500/20 text-emerald-300',
    prune: 'bg-rose-500/25 text-rose-300',
    exit: 'bg-amber-500/20 text-amber-300',
  }

  return (
    <div className="space-y-8">
      <div className="space-y-5 rounded-[1.75rem] bg-slate-950 p-4 text-slate-100 sm:p-6">
        {/* header: title + state legend */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="rounded-xl border border-indigo-500/30 bg-indigo-600/20 p-2.5 text-indigo-300"><GitBranch size={22} aria-hidden /></div>
            <div>
              <p className="bg-gradient-to-r from-indigo-300 via-purple-300 to-pink-300 bg-clip-text text-lg font-extrabold tracking-tight text-transparent">{t('alphabeta.header_title')}</p>
              <p className="text-xs text-slate-400">{t('alphabeta.header_sub')}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {[['bg-emerald-500', 'visited'], ['bg-indigo-500', 'active'], ['bg-rose-500', 'pruned']].map(([c, k]) => (
              <span key={k} className="flex items-center gap-2 rounded-lg border border-slate-700/60 bg-slate-800/80 px-3 py-1.5"><span className={`h-2.5 w-2.5 rounded-full ${c}`} />{t(`alphabeta.legend.${k}`)}</span>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* left: presets + tree canvas */}
          <section className="flex min-w-0 flex-col gap-5 lg:col-span-8">
            <div className={`${panel} flex flex-wrap items-center justify-between gap-4 p-4`}>
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-400"><Layers size={14} aria-hidden /> {t('alphabeta.presets_label')}</span>
                <div className="flex flex-wrap gap-2">
                  {(['user', 'beta', 'alpha'] as const).map((key) => (
                    <button
                      key={key}
                      onClick={() => applyPreset(key)}
                      aria-pressed={preset === key}
                      className={`rounded-lg border px-3.5 py-1.5 text-xs font-semibold transition ${preset === key ? 'border-indigo-500 bg-indigo-600/25 text-indigo-200 shadow-md' : 'border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700/60'}`}
                    >
                      {t(`alphabeta.preset.${key}`)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex gap-2">
                <button onClick={randomize} className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700">
                  <Shuffle size={14} className="text-indigo-400" aria-hidden /> {t('alphabeta.randomize')}
                </button>
                <button onClick={() => applyPreset(preset === 'custom' ? 'user' : preset)} className="flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-medium text-slate-200 hover:bg-slate-700">
                  <RotateCcw size={14} className="text-slate-400" aria-hidden /> {t('alphabeta.reset')}
                </button>
              </div>
            </div>

            <div className={`${panel} relative min-h-[360px] overflow-hidden rounded-3xl p-4 shadow-2xl`}>
              <div className="absolute left-4 top-4 z-10 rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-1 text-[11px] font-semibold text-slate-400 backdrop-blur">{t('alphabeta.canvas_label')}</div>
              <div className="absolute right-4 top-4 z-10 flex flex-col gap-2 rounded-xl border border-slate-800/80 bg-slate-950/70 p-3 text-[11px] text-slate-300 backdrop-blur">
                <span className="flex items-center gap-2"><span className="h-1.5 w-3.5 rounded bg-indigo-500" />{t('alphabeta.layers.max')}</span>
                <span className="flex items-center gap-2"><span className="h-1.5 w-3.5 rounded bg-amber-500" />{t('alphabeta.layers.min')}</span>
                <span className="flex items-center gap-2"><span className="h-1.5 w-3.5 rounded bg-emerald-500" />{t('alphabeta.layers.leaf')}</span>
              </div>
              <svg viewBox="0 0 1000 470" className="mt-10 h-auto max-h-[460px] w-full" role="img" aria-label={t('alphabeta.tree_label')}>
                <defs>
                  <marker id="ab-arrow" viewBox="0 0 10 10" refX="20" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" />
                  </marker>
                </defs>
                {/* 边 */}
                {EDGES.map((e) => {
                  const p = nodes[e.from]
                  const c = nodes[e.to]
                  const key = `${e.from}->${e.to}`
                  const isPruned = step.prunedEdges.includes(key)
                  const isActive = step.activeEdge === key
                  const isVisited = step.visited.includes(e.to) && !isPruned
                  let stroke = '#334155'
                  let width = 3
                  let dash = '0'
                  if (isPruned) { stroke = '#f43f5e'; width = 2; dash = '6,5' }
                  else if (isActive) { stroke = '#818cf8'; width = 6 }
                  else if (isVisited) { stroke = '#10b981'; width = 4 }
                  return (
                    <g key={key}>
                      <line x1={p.x} y1={p.y} x2={c.x} y2={c.y} stroke={stroke} strokeWidth={width} strokeDasharray={dash} markerEnd="url(#ab-arrow)" />
                      {isPruned && (
                        <g transform={`translate(${(p.x + c.x) / 2}, ${(p.y + c.y) / 2})`}>
                          <circle r={10} fill="#f43f5e" />
                          <text textAnchor="middle" dy={3.5} fill="white" fontSize={10} fontWeight={900}>×</text>
                        </g>
                      )}
                    </g>
                  )
                })}
                {/* 节点 */}
                {Object.values(nodes).map((node) => {
                  const value = step.nodeValues[node.id]
                  const a = step.nodeAlphas[node.id]
                  const b = step.nodeBetas[node.id]
                  const isPruned = step.prunedNodes.includes(node.id)
                  const isActive = step.nodeId === node.id
                  const isVisited = step.visited.includes(node.id)
                  let fill = '#0f172a'
                  let stroke = '#334155'
                  let text = '#94a3b8'
                  if (isPruned) { fill = '#1f0a12'; stroke = '#7f1d1d'; text = '#9f1239' }
                  else if (isActive) { fill = '#1e1b4b'; stroke = '#6366f1'; text = '#c7d2fe' }
                  else if (isVisited) { fill = '#052e2b'; stroke = '#10b981'; text = '#a7f3d0' }
  
                  if (node.type === 'leaf') {
                    return (
                      <g key={node.id} className="cursor-pointer" onClick={() => {
                        const nv = window.prompt(t('alphabeta.edit_leaf', { id: node.id }), String(node.value))
                        const n = Number(nv)
                        if (nv !== null && Number.isFinite(n) && n >= 0 && n <= 20) setLeaf(node.id, Math.round(n))
                      }}>
                        <circle cx={node.x} cy={node.y} r={22} fill={fill} stroke={stroke} strokeWidth={3} />
                        <text x={node.x} y={node.y + 5} textAnchor="middle" fontSize={14} fontWeight={800} fill={isPruned ? '#9f1239' : '#f1f5f9'}>{node.value}</text>
                        <text x={node.x + 28} y={node.y + 4} fontSize={10} fontFamily="monospace" fill="#64748b" fontWeight={700}>{node.id}</text>
                      </g>
                    )
                  }
                  const isMax = node.type === 'max'
                  const points = isMax
                    ? `${node.x},${node.y - 26} ${node.x - 26},${node.y + 18} ${node.x + 26},${node.y + 18}`
                    : `${node.x - 26},${node.y - 18} ${node.x + 26},${node.y - 18} ${node.x},${node.y + 26}`
                  return (
                    <g key={node.id}>
                      <polygon points={points} fill={fill} stroke={stroke} strokeWidth={3} />
                      <text x={node.x} y={node.y + (isMax ? 10 : -2)} textAnchor="middle" fontSize={9} fontWeight={800} fill={text}>{isMax ? '▲MAX' : '▼MIN'}</text>
                      <g transform={`translate(${node.x}, ${node.y + (isMax ? -40 : -34)})`}>
                        <rect x={-52} y={-11} width={104} height={19} rx={6} fill="#020617" stroke="#1e293b" />
                        <text textAnchor="middle" dy={2} fontSize={9} fontFamily="monospace">
                          <tspan fill="#34d399">α:{fmt(a)}</tspan> <tspan fill="#fbbf24">β:{fmt(b)}</tspan>
                        </text>
                      </g>
                      <text x={node.x + 32} y={node.y + 2} fontSize={10} fontFamily="monospace" fill="#64748b" fontWeight={700}>{node.id}</text>
                      {value !== null && (
                        <g transform={`translate(${node.x}, ${node.y + (isMax ? 34 : 40)})`}>
                          <rect x={-15} y={-10} width={30} height={18} rx={5} fill="#10b981" />
                          <text textAnchor="middle" dy={3} fontSize={11} fontWeight="bold" fill="white">{value}</text>
                        </g>
                      )}
                    </g>
                  )
                })}
              </svg>
            </div>
          </section>

          {/* right: control deck + live explanation */}
          <section className="flex min-w-0 flex-col gap-5 lg:col-span-4">
            <div className="flex flex-col gap-5 rounded-3xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-xl">
              <h3 className={heading}><Play size={15} className="text-indigo-400" aria-hidden /> {t('alphabeta.control')}</h3>
              <div className="grid grid-cols-4 gap-2">
                <button onClick={() => { setStepIdx(0); setPlaying(false) }} className="flex items-center justify-center rounded-xl bg-slate-800 p-3 text-slate-200 transition hover:scale-105 hover:bg-slate-700" title={t('alphabeta.first')} aria-label={t('alphabeta.first')}><ChevronsLeft size={18} /></button>
                <button onClick={() => { setStepIdx((i) => Math.max(0, i - 1)); setPlaying(false) }} className="flex items-center justify-center rounded-xl bg-slate-800 p-3 text-slate-200 transition hover:scale-105 hover:bg-slate-700" title={t('alphabeta.prev')} aria-label={t('alphabeta.prev')}><ChevronLeft size={18} /></button>
                <button onClick={() => { if (safeIdx >= steps.length - 1) setStepIdx(0); setPlaying((p) => !p) }} className="flex items-center justify-center rounded-xl bg-indigo-600 p-3 text-white shadow-lg shadow-indigo-600/25 transition hover:scale-105 hover:bg-indigo-500" title={t('alphabeta.play')} aria-label={t('alphabeta.play')}>{playing ? <Pause size={18} /> : <Play size={18} className="fill-white" />}</button>
                <button onClick={() => { setStepIdx((i) => Math.min(steps.length - 1, i + 1)); setPlaying(false) }} className="flex items-center justify-center rounded-xl bg-slate-800 p-3 text-slate-200 transition hover:scale-105 hover:bg-slate-700" title={t('alphabeta.next')} aria-label={t('alphabeta.next')}><ChevronRight size={18} /></button>
              </div>
              <div className="flex flex-col gap-4 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs text-slate-400"><span>{t('alphabeta.progress')}</span><span className="font-mono font-bold text-indigo-300">{safeIdx + 1} / {steps.length}</span></div>
                  <input type="range" min={0} max={steps.length - 1} value={safeIdx} aria-label={t('alphabeta.progress')} onChange={(e) => { setStepIdx(parseInt(e.target.value)); setPlaying(false) }} className="w-full cursor-pointer accent-indigo-500" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs text-slate-400"><span>{t('alphabeta.speed')}</span><span className="font-mono text-slate-300">{(speed / 1000).toFixed(1)}s</span></div>
                  <input type="range" min={200} max={2400} step={200} value={speed} aria-label={t('alphabeta.speed')} onChange={(e) => setSpeed(parseInt(e.target.value))} className="w-full cursor-pointer accent-indigo-500" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 border-t border-slate-800 pt-4 text-center">
                <div><p className="text-[10px] uppercase text-slate-400">{t('alphabeta.evaluated')}</p><p className="text-xl font-black text-emerald-400">{counts.evaluated}/{counts.total}</p></div>
                <div><p className="text-[10px] uppercase text-slate-400">{t('alphabeta.pruned')}</p><p className="text-xl font-black text-rose-400">{counts.prunedLeaves}</p></div>
              </div>
            </div>

            <div className="flex min-h-[260px] flex-1 flex-col rounded-3xl border border-slate-800 bg-slate-900 p-5 shadow-xl">
              <h3 className={`${heading} border-b border-slate-800 pb-3`}><FileText size={15} className="text-emerald-400" aria-hidden /> {t('alphabeta.detail')}</h3>
              <div className="flex flex-1 flex-col justify-between gap-4 pt-4">
                <p className="text-sm leading-relaxed text-slate-200" aria-live="polite">{explanation}</p>
                <div className={`flex flex-col gap-2 rounded-xl border p-3.5 transition-colors ${mathBoxClass}`}>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">{t('alphabeta.range_label')}</div>
                  <div className="flex items-center justify-around font-mono text-xs">
                    <div className="flex flex-col items-center"><span className="mb-0.5 text-[10px] text-slate-400">α ({t('alphabeta.lower')})</span><span className="text-base font-bold text-emerald-400">{fmt(step.alpha)}</span></div>
                    <div className="text-lg text-slate-500" aria-hidden>|</div>
                    <div className="flex flex-col items-center"><span className="mb-0.5 text-[10px] text-slate-400">β ({t('alphabeta.upper')})</span><span className="text-base font-bold text-amber-400">{fmt(step.beta)}</span></div>
                  </div>
                  <div className="mt-1 border-t border-slate-800 pt-2 text-center text-xs font-semibold italic text-slate-300">{formula}</div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* leaf value sandbox, full width */}
        <section className={`${panel} flex flex-col gap-4 rounded-3xl p-5`}>
          <h3 className={`${heading} flex-wrap justify-between border-b border-slate-800 pb-3`}>
            <span className="flex items-center gap-2"><SlidersHorizontal size={15} className="text-purple-400" aria-hidden /> {t('alphabeta.sandbox')}</span>
            <span className="text-xs font-normal normal-case tracking-normal text-slate-400">{t('alphabeta.sandbox_hint')}</span>
          </h3>
          <div className="grid grid-cols-4 gap-4 md:grid-cols-8">
            {AB_LEAF_IDS.map((id) => (
              <div key={id} className="flex flex-col items-center gap-1.5">
                <span className="font-mono text-[11px] font-bold text-slate-400">{id}</span>
                <input type="range" min={0} max={20} value={leaves[id]} onChange={(e) => setLeaf(id, parseInt(e.target.value))} aria-label={t('alphabeta.leaf_label', { id })} className="w-full cursor-pointer accent-indigo-500" />
                <span className="text-sm font-black text-indigo-300">{leaves[id]}</span>
              </div>
            ))}
          </div>
        </section>

        {/* step trace: click any step to jump there */}
        <section className={`${panel} flex h-[260px] flex-col gap-3 rounded-3xl p-5`}>
          <h3 className={`${heading} border-b border-slate-800 pb-2`}><ListOrdered size={15} className="text-indigo-400" aria-hidden /> {t('alphabeta.timeline')}</h3>
          <ol ref={listRef} className="relative flex flex-1 flex-col gap-1 overflow-y-auto pr-2">
            {steps.map((s, k) => (
              <li key={k} data-step={k}>
                <button
                  type="button"
                  onClick={() => { setStepIdx(k); setPlaying(false) }}
                  aria-current={k === safeIdx ? 'step' : undefined}
                  className={`flex w-full items-start gap-3 rounded-lg px-2.5 py-1.5 text-left text-xs transition ${k === safeIdx ? 'bg-indigo-600/25 ring-1 ring-indigo-500/60' : k < safeIdx ? 'text-slate-400 hover:bg-slate-800' : 'text-slate-300 hover:bg-slate-800'}`}
                >
                  <span className="w-7 shrink-0 pt-0.5 text-right font-mono text-[10px] text-slate-500">{k + 1}</span>
                  <span className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${STEP_TONE[s.type]}`}>{t(`alphabeta.step_type.${s.type}`)}</span>
                  <span className="leading-relaxed">{explain(s)}</span>
                </button>
              </li>
            ))}
          </ol>
        </section>
      </div>
      <SeniorAdvice content={<Trans i18nKey="alphabeta.advice" components={{
        1: <Latex formula="\beta \leq \alpha" />,
        3: <strong className="font-bold text-amber-700" />,
      }} />} />
    </div>
  )
}
