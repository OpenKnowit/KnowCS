import { useEffect, useMemo, useState } from 'react'
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  GitBranch,
  Pause,
  Play,
  RotateCcw,
  Scissors,
  Shuffle,
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

  // 结构化步骤 → 本地化解释文案
  const explanation = useMemo(() => {
    const s = step
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
        return t('alphabeta.explain.prune', {
          node: s.nodeId,
          alpha: fmt(s.alpha),
          beta: fmt(s.beta),
          children: s.prunedChildren!.join(', '),
        })
      case 'exit':
        return t('alphabeta.explain.exit', { node: s.nodeId, value: s.returnValue })
    }
  }, [step, nodes, t])

  const counts = useMemo(() => {
    const pruned = new Set(steps.at(-1)!.prunedNodes)
    const prunedLeaves = AB_LEAF_IDS.filter((id) => pruned.has(id)).length
    return { evaluated: AB_LEAF_IDS.length - prunedLeaves, prunedLeaves, total: AB_LEAF_IDS.length }
  }, [steps])

  const mathBoxClass =
    step.type === 'prune'
      ? 'bg-rose-50 border-rose-300'
      : step.type === 'leaf_eval' || step.type === 'update'
        ? 'bg-emerald-50 border-emerald-200'
        : 'bg-slate-50 border-slate-200'

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* 树画布 */}
        <div className="lg:col-span-8 space-y-4">
          {/* 预设 + 随机 */}
          <div className="flex flex-wrap gap-3 items-center justify-between bg-slate-50 p-4 rounded-2xl border border-slate-200">
            <div className="flex flex-wrap gap-2">
              {(['user', 'beta', 'alpha'] as const).map((key) => (
                <button
                  key={key}
                  onClick={() => applyPreset(key)}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-bold transition ${preset === key ? 'bg-indigo-600 text-white shadow' : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-100'}`}
                >
                  {t(`alphabeta.preset.${key}`)}
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={randomize} className="px-3 py-1.5 rounded-lg text-xs font-bold bg-white border border-slate-200 text-slate-600 hover:bg-slate-100 flex items-center gap-1.5">
                <Shuffle size={13} className="text-indigo-500" /> {t('alphabeta.randomize')}
              </button>
              <button onClick={() => applyPreset(preset === 'custom' ? 'user' : preset)} className="px-3 py-1.5 rounded-lg text-xs font-bold bg-white border border-slate-200 text-slate-600 hover:bg-slate-100 flex items-center gap-1.5">
                <RotateCcw size={13} className="text-slate-400" /> {t('alphabeta.reset')}
              </button>
            </div>
          </div>

          {/* SVG 树 */}
          <div className="bg-slate-900 rounded-[2rem] p-4 shadow-2xl relative overflow-hidden">
            <div className="absolute top-4 right-4 z-10 flex flex-col gap-1.5 text-[9px] bg-slate-950/60 border border-slate-800 rounded-xl p-2.5 backdrop-blur">
              <div className="flex items-center gap-2"><span className="w-3 h-1.5 rounded bg-indigo-500" />{t('alphabeta.legend.active')}</div>
              <div className="flex items-center gap-2"><span className="w-3 h-1.5 rounded bg-emerald-500" />{t('alphabeta.legend.visited')}</div>
              <div className="flex items-center gap-2"><span className="w-3 h-1.5 rounded bg-rose-500" />{t('alphabeta.legend.pruned')}</div>
            </div>
            <svg viewBox="0 0 1000 470" className="w-full h-auto max-h-[460px]">
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

          {/* 叶子滑块沙盒 */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
            <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Shuffle size={13} className="text-purple-500" /> {t('alphabeta.sandbox')}
            </h4>
            <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
              {AB_LEAF_IDS.map((id) => (
                <div key={id} className="flex flex-col items-center gap-1.5">
                  <span className="text-[10px] font-mono font-bold text-slate-500">{id}</span>
                  <input type="range" min={0} max={20} value={leaves[id]} onChange={(e) => setLeaf(id, parseInt(e.target.value))} className="w-full accent-indigo-600 cursor-pointer" />
                  <span className="text-sm font-black text-indigo-600">{leaves[id]}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 控制台 + 解释 */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-slate-900 rounded-[2rem] p-6 text-white shadow-2xl space-y-5">
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
              <GitBranch size={14} className="text-indigo-400" /> {t('alphabeta.control')}
            </h4>
            <div className="grid grid-cols-4 gap-2">
              <button onClick={() => { setStepIdx(0); setPlaying(false) }} className="p-3 bg-slate-800 hover:bg-slate-700 rounded-xl flex items-center justify-center" title={t('alphabeta.first')}><ChevronsLeft size={18} /></button>
              <button onClick={() => { setStepIdx((i) => Math.max(0, i - 1)); setPlaying(false) }} className="p-3 bg-slate-800 hover:bg-slate-700 rounded-xl flex items-center justify-center" title={t('alphabeta.prev')}><ChevronLeft size={18} /></button>
              <button onClick={() => { if (safeIdx >= steps.length - 1) setStepIdx(0); setPlaying((p) => !p) }} className="p-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-600/25" title={t('alphabeta.play')}>{playing ? <Pause size={18} /> : <Play size={18} />}</button>
              <button onClick={() => { setStepIdx((i) => Math.min(steps.length - 1, i + 1)); setPlaying(false) }} className="p-3 bg-slate-800 hover:bg-slate-700 rounded-xl flex items-center justify-center" title={t('alphabeta.next')}><ChevronRight size={18} /></button>
            </div>
            <div className="bg-slate-950/50 p-4 rounded-xl border border-slate-800 space-y-4">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                  <span>{t('alphabeta.progress')}</span>
                  <span className="font-mono text-indigo-400 font-bold">{safeIdx + 1} / {steps.length}</span>
                </div>
                <input type="range" min={0} max={steps.length - 1} value={safeIdx} onChange={(e) => { setStepIdx(parseInt(e.target.value)); setPlaying(false) }} className="w-full accent-indigo-500 cursor-pointer" />
              </div>
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                  <span>{t('alphabeta.speed')}</span>
                  <span className="font-mono text-slate-300">{(speed / 1000).toFixed(1)}s</span>
                </div>
                <input type="range" min={200} max={2400} step={200} value={speed} onChange={(e) => setSpeed(parseInt(e.target.value))} className="w-full accent-indigo-500 cursor-pointer" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-center border-t border-slate-800 pt-4">
              <div><p className="text-[9px] text-slate-500 uppercase">{t('alphabeta.evaluated')}</p><p className="text-xl font-black text-emerald-400">{counts.evaluated}/{counts.total}</p></div>
              <div><p className="text-[9px] text-slate-500 uppercase">{t('alphabeta.pruned')}</p><p className="text-xl font-black text-rose-400">{counts.prunedLeaves}</p></div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-[2rem] p-6 shadow-sm space-y-4">
            <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
              <Scissors size={14} className="text-emerald-500" /> {t('alphabeta.detail')}
            </h4>
            <p className="text-sm text-slate-700 leading-relaxed min-h-[72px]">{explanation}</p>
            <div className={`rounded-xl p-3.5 border ${mathBoxClass} transition-colors`}>
              <div className="flex justify-around items-center text-xs font-mono">
                <div className="flex flex-col items-center">
                  <span className="text-slate-400 text-[10px] mb-0.5">α ({t('alphabeta.lower')})</span>
                  <span className="text-emerald-600 font-bold text-base">{fmt(step.alpha)}</span>
                </div>
                <div className="text-slate-300 text-lg">|</div>
                <div className="flex flex-col items-center">
                  <span className="text-slate-400 text-[10px] mb-0.5">β ({t('alphabeta.upper')})</span>
                  <span className="text-amber-600 font-bold text-base">{fmt(step.beta)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <SeniorAdvice content={<Trans i18nKey="alphabeta.advice" components={{
        1: <Latex formula="\beta \leq \alpha" />,
        3: <strong className="font-bold text-amber-700" />,
      }} />} />
    </div>
  )
}
