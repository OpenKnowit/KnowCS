import { useEffect, useMemo, useRef, useState } from 'react'
import type { FocusEvent as ReactFocusEvent, KeyboardEvent as ReactKeyboardEvent } from 'react'
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ArrowLeft,
  ArrowRight,
  FileText,
  GitBranch,
  Layers,
  ListOrdered,
  Pause,
  Play,
  RotateCcw,
  Maximize2,
  Minus,
  Pencil,
  Plus,
  Shuffle,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { AB_PRESETS } from '../data/constants'
import { addChild, canMove, canRemove, fullShape, generateAbSteps, layoutAbTree, mapLeaves, moveNode, removeNode, setLeafValue, shapeDepth, shapeLeaves } from '../lib/alphabeta'
import type { AbPath, AbShape } from '../lib/alphabeta'
import type { AbNode, AbStep } from '../types'

const fmt = (v: number): string => (v === Infinity ? '∞' : v === -Infinity ? '-∞' : String(v))

const MIN_DEPTH = 2
const MAX_DEPTH = 5
const randomLeaf = () => Math.floor(Math.random() * 21)
const randomLeaves = (n: number) => Array.from({ length: n }, randomLeaf)
// how big an edited tree may grow: the canvas stays readable
const MAX_LEAVES = 32
const MAX_LEVELS = 6

export const AlphaBetaModule = () => {
  const { t } = useTranslation()
  // the tree: a full binary one from the presets and the level buttons, or any shape in edit mode
  const [shape, setShape] = useState<AbShape>(() => fullShape(3, AB_PRESETS.alpha))
  const [preset, setPreset] = useState<string>('alpha')
  // edit mode: click a node to select it, then add a child, move it or remove it
  const [editMode, setEditMode] = useState(false)
  const [selected, setSelected] = useState<AbPath | null>(null)
  const [stepIdx, setStepIdx] = useState(0)
  // the leaf whose value is being typed in place (click a leaf); null = none
  const [editing, setEditing] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1000)

  const tree = useMemo(() => layoutAbTree(shape), [shape])
  const nodes: Record<string, AbNode> = tree.nodes
  const depth = shapeDepth(shape)
  const selectedId = selected && Object.keys(tree.paths).find((id) => tree.paths[id].join() === selected.join())
  const edges = useMemo(() => Object.values(nodes).flatMap((n) => (n.children ?? []).map((c) => ({ from: n.id, to: c }))), [nodes])
  const leafIds = useMemo(() => Object.values(nodes).filter((n) => n.type === 'leaf').map((n) => n.id), [nodes])

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

  // any change to the tree starts the search again from the first step
  const changeTree = (next: AbShape, presetKey = 'custom') => {
    setShape(next)
    setPreset(presetKey)
    setStepIdx(0)
    setPlaying(false)
  }
  const applyPreset = (key: string) => {
    changeTree(fullShape(3, AB_PRESETS[key]), key)
    setSelected(null)
  }
  const randomize = () => changeTree(mapLeaves(shape, randomLeaf))
  const setLeaf = (id: string, v: number) => changeTree(setLeafValue(shape, tree.paths[id], v))
  // more or fewer levels: a full binary tree again, keeping the leaves we have (left to right); new ones are random
  const changeDepth = (d: number) => {
    const next = Math.max(MIN_DEPTH, Math.min(MAX_DEPTH, d))
    const prev = shapeLeaves(shape)
    const n = 2 ** next
    changeTree(fullShape(next, prev.length >= n ? prev.slice(0, n) : [...prev, ...randomLeaves(n - prev.length)]))
    setSelected(null)
  }
  const isLeafSel = selectedId ? nodes[selectedId].type === 'leaf' : false
  const canAdd = !!selected && shapeLeaves(shape).length < MAX_LEAVES && (!isLeafSel || selected.length < MAX_LEVELS)
  const addToSelected = () => {
    if (!selected || !canAdd) return
    changeTree(addChild(shape, selected, randomLeaf()))
  }
  const removeSelected = () => {
    if (!selected || !canRemove(shape, selected)) return
    changeTree(removeNode(shape, selected, randomLeaf()))
    setSelected(null)
  }
  const moveSelected = (dir: -1 | 1) => {
    if (!selected || !canMove(shape, selected, dir)) return
    changeTree(moveNode(shape, selected, dir))
    setSelected([...selected.slice(0, -1), selected.at(-1)! + dir])
  }
  const toggleEdit = () => {
    setEditMode((on) => !on)
    setSelected(null)
    setEditing(null)
  }

  // the canvas: drag to move, Ctrl/⌘ + wheel or the buttons to zoom; it fits the tree again when its size changes
  const W = Math.max(tree.width, 1000)
  const H = tree.height
  const fit = { k: 1, tx: (W - tree.width) / 2, ty: 0 }
  const [view, setView] = useState(fit)
  const [viewFor, setViewFor] = useState(`${W}x${H}`)
  if (viewFor !== `${W}x${H}`) {
    setViewFor(`${W}x${H}`)
    setView(fit)
  }
  const svgRef = useRef<SVGSVGElement>(null)
  // a leaf is too small to type in when the tree is drawn small, as on a phone: the box then sits under the tree
  const [editInBar, setEditInBar] = useState(false)
  const drag = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)
  const unit = () => {
    const r = svgRef.current?.getBoundingClientRect()
    return r ? Math.min(r.width / W, r.height / H) : 1
  }
  const zoomAt = (factor: number, cx = W / 2, cy = H / 2) =>
    setView((v) => {
      const k = Math.max(0.3, Math.min(4, v.k * factor))
      return { k, tx: cx - ((cx - v.tx) * k) / v.k, ty: cy - ((cy - v.ty) * k) / v.k }
    })

  const leafInput = (node: { id: string; value?: number }) => ({
    type: 'number' as const,
    inputMode: 'numeric' as const,
    min: 0,
    max: 20,
    autoFocus: true,
    defaultValue: node.value,
    'aria-label': t('alphabeta.edit_leaf', { id: node.id }),
    onFocus: (e: ReactFocusEvent<HTMLInputElement>) => e.currentTarget.select(),
    onKeyDown: (e: ReactKeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') e.currentTarget.blur()
      if (e.key === 'Escape') {
        e.currentTarget.value = String(node.value)
        e.currentTarget.blur()
      }
    },
    onBlur: (e: ReactFocusEvent<HTMLInputElement>) => {
      const n = Number(e.currentTarget.value)
      if (e.currentTarget.value.trim() !== '' && Number.isFinite(n)) setLeaf(node.id, Math.max(0, Math.min(20, Math.round(n))))
      setEditing(null)
    },
  })

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
    const prunedLeaves = leafIds.filter((id) => pruned.has(id)).length
    return { evaluated: leafIds.length - prunedLeaves, prunedLeaves, total: leafIds.length }
  }, [steps, leafIds])

  const mathBoxClass = step.type === 'prune' ? 'border-rose-300 bg-rose-50' : step.type === 'leaf_eval' || step.type === 'update' ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-slate-50'
  const panel = 'rounded-[20px] border border-slate-200 bg-white'
  const heading = 'flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-700'
  const STEP_TONE: Record<AbStep['type'], string> = {
    enter: 'bg-blue-100 text-blue-700',
    traverse: 'bg-slate-100 text-slate-600',
    leaf_eval: 'bg-blue-100 text-blue-700',
    update: 'bg-blue-100 text-blue-700',
    prune: 'bg-rose-100 text-rose-700',
    exit: 'bg-amber-100 text-amber-800',
  }

  return (
    <div className="space-y-8">
      <div className="space-y-5">
        {/* header: title + state legend */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 pb-4">
          <div className="flex items-center gap-3">
            <div className="rounded-xl border border-blue-200 bg-blue-50 p-2.5 text-blue-600"><GitBranch size={22} aria-hidden /></div>
            <div>
              <p className="text-lg font-extrabold tracking-tight text-slate-900">{t('alphabeta.header_title')}</p>
              <p className="text-xs text-slate-500">{t('alphabeta.header_sub')}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {[['bg-blue-200 ring-1 ring-blue-500', 'visited'], ['bg-blue-800', 'active'], ['bg-rose-500', 'pruned']].map(([c, k]) => (
              <span key={k} className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-slate-700"><span className={`h-2.5 w-2.5 rounded-full ${c}`} />{t(`alphabeta.legend.${k}`)}</span>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
          {/* left: presets + tree canvas */}
          <section className="flex min-w-0 flex-col gap-5 lg:col-span-8">
            <div className={`${panel} flex flex-wrap items-center justify-between gap-4 p-4`}>
              <div className="flex flex-col gap-1.5">
                <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-500"><Layers size={14} aria-hidden /> {t('alphabeta.presets_label')}</span>
                <div className="flex flex-wrap gap-2">
                  {(['user', 'beta', 'alpha'] as const).map((key) => (
                    <button
                      key={key}
                      onClick={() => applyPreset(key)}
                      aria-pressed={preset === key}
                      className={`rounded-lg border px-3.5 py-1.5 text-xs font-semibold transition ${preset === key ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300'}`}
                    >
                      {t(`alphabeta.preset.${key}`)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="text-xs font-semibold text-slate-500">{t('alphabeta.levels')}</span>
                <div className="flex items-center gap-1">
                  <button type="button" onClick={() => changeDepth(depth - 1)} disabled={depth <= MIN_DEPTH && leafIds.length === 2 ** depth} aria-label={t('alphabeta.levels_less')} className="grid h-8 w-8 place-items-center rounded-lg border border-slate-300 bg-white text-slate-700 hover:border-slate-400 disabled:opacity-40"><Minus size={14} aria-hidden /></button>
                  <span className="w-16 text-center font-mono text-sm font-bold text-slate-900">{depth} · {leafIds.length}</span>
                  <button type="button" onClick={() => changeDepth(depth + 1)} disabled={depth >= MAX_DEPTH && leafIds.length === 2 ** depth} aria-label={t('alphabeta.levels_more')} className="grid h-8 w-8 place-items-center rounded-lg border border-slate-300 bg-white text-slate-700 hover:border-slate-400 disabled:opacity-40"><Plus size={14} aria-hidden /></button>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={toggleEdit}
                  aria-pressed={editMode}
                  className={`flex items-center gap-2 rounded-xl border px-4 py-2 text-xs font-semibold transition ${editMode ? 'border-blue-600 bg-blue-600 text-white hover:bg-blue-700' : 'border-slate-300 bg-white text-slate-700 hover:border-slate-400'}`}
                >
                  <Pencil size={14} className={editMode ? 'text-white' : 'text-blue-600'} aria-hidden /> {t(editMode ? 'alphabeta.edit_done' : 'alphabeta.edit_mode')}
                </button>
                <button onClick={randomize} className="flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:border-slate-400">
                  <Shuffle size={14} className="text-blue-600" aria-hidden /> {t('alphabeta.randomize')}
                </button>
                <button onClick={() => applyPreset(preset === 'custom' ? 'alpha' : preset)} className="flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:border-slate-400">
                  <RotateCcw size={14} className="text-slate-500" aria-hidden /> {t('alphabeta.reset')}
                </button>
              </div>
            </div>

            {editMode && (
              <div className="flex min-h-[3.25rem] flex-wrap items-center gap-2 rounded-[20px] border border-blue-300 bg-blue-50/60 px-4 py-2 text-xs text-slate-700">
                {selectedId ? (
                  <>
                    <span className="font-mono text-sm font-bold text-slate-900">{selectedId}</span>
                    <span className="text-slate-500">{nodes[selectedId].type === 'leaf' ? t('alphabeta.edit_leaf_type') : nodes[selectedId].type.toUpperCase()}</span>
                    {isLeafSel && (
                      <label className="flex items-center gap-1.5 font-semibold">
                        {t('alphabeta.edit_value')}
                        <input key={`${selectedId}:${nodes[selectedId].value}`} {...leafInput(nodes[selectedId])} autoFocus={false} className="h-8 w-16 rounded-lg border-2 border-blue-600 bg-white text-center font-mono text-sm font-bold outline-none" />
                      </label>
                    )}
                    <span className="ml-auto flex flex-wrap gap-1">
                      {[
                        { icon: Plus, label: t(isLeafSel ? 'alphabeta.edit_expand' : 'alphabeta.edit_add'), on: addToSelected, ok: canAdd },
                        { icon: ArrowLeft, label: t('alphabeta.edit_left'), on: () => moveSelected(-1), ok: canMove(shape, selected!, -1) },
                        { icon: ArrowRight, label: t('alphabeta.edit_right'), on: () => moveSelected(1), ok: canMove(shape, selected!, 1) },
                        { icon: Trash2, label: t('alphabeta.edit_delete'), on: removeSelected, ok: canRemove(shape, selected!), danger: true },
                      ].map(({ icon: Icon, label, on, ok, danger }) => (
                        <button key={label} type="button" onClick={on} disabled={!ok} className={`flex items-center gap-1 rounded-lg border px-2.5 py-1.5 font-semibold disabled:opacity-40 ${danger ? 'border-rose-200 text-rose-700 hover:bg-rose-50' : 'border-slate-300 text-slate-700 hover:bg-slate-50'}`}>
                          <Icon size={13} aria-hidden /> {label}
                        </button>
                      ))}
                      <button type="button" onClick={() => setSelected(null)} aria-label={t('alphabeta.edit_deselect')} title={t('alphabeta.edit_deselect')} className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100"><X size={14} aria-hidden /></button>
                    </span>
                  </>
                ) : (
                  <span>{t('alphabeta.edit_hint', { leaves: MAX_LEAVES, levels: MAX_LEVELS })}</span>
                )}
              </div>
            )}
            <div className={`${panel} relative flex flex-col overflow-hidden sm:block sm:h-[480px]`}>
              <div className="pointer-events-none px-4 pt-3 text-[11px] font-semibold text-slate-600 sm:absolute sm:left-4 sm:top-4 sm:z-10 sm:rounded-lg sm:border sm:border-slate-200 sm:bg-white/90 sm:px-3 sm:py-1 sm:text-slate-500 backdrop-blur">{t(editMode ? 'alphabeta.edit_canvas_label' : 'alphabeta.canvas_label')}</div>
              <div className="order-last flex items-center justify-end gap-1 border-t border-slate-100 p-1 sm:absolute sm:bottom-3 sm:right-3 sm:z-10 sm:rounded-xl sm:border sm:border-slate-200 sm:bg-white/90 sm:backdrop-blur">
                <span className="hidden px-2 text-[11px] text-slate-500 sm:inline coarse:hidden">{t('alphabeta.canvas_hint')}</span>
                <button type="button" onClick={() => zoomAt(1 / 1.25)} aria-label={t('alphabeta.zoom_out')} title={t('alphabeta.zoom_out')} className="grid h-7 w-7 place-items-center coarse:h-9 coarse:w-9 rounded-lg text-slate-600 hover:bg-slate-100"><ZoomOut size={15} aria-hidden /></button>
                <button type="button" onClick={() => zoomAt(1.25)} aria-label={t('alphabeta.zoom_in')} title={t('alphabeta.zoom_in')} className="grid h-7 w-7 place-items-center coarse:h-9 coarse:w-9 rounded-lg text-slate-600 hover:bg-slate-100"><ZoomIn size={15} aria-hidden /></button>
                <button type="button" onClick={() => setView(fit)} aria-label={t('alphabeta.zoom_fit')} title={t('alphabeta.zoom_fit')} className="grid h-7 w-7 place-items-center coarse:h-9 coarse:w-9 rounded-lg text-slate-600 hover:bg-slate-100"><Maximize2 size={14} aria-hidden /></button>
              </div>
              <div className="pointer-events-none z-10 flex flex-wrap gap-x-3 gap-y-1 border-b border-slate-100 px-4 py-2 text-[11px] text-slate-600 sm:absolute sm:right-4 sm:top-4 sm:flex-col sm:gap-2 sm:rounded-xl sm:border sm:border-slate-200 sm:bg-white/90 sm:p-3 sm:text-slate-600 backdrop-blur">
                <span className="flex items-center gap-2"><span className="w-3.5 text-center text-slate-600">▲</span>{t('alphabeta.layers.max')}</span>
                <span className="flex items-center gap-2"><span className="w-3.5 text-center text-slate-600">▼</span>{t('alphabeta.layers.min')}</span>
                <span className="flex items-center gap-2"><span className="w-3.5 text-center text-slate-600">●</span>{t('alphabeta.layers.leaf')}</span>
              </div>
              <svg
                ref={svgRef}
                viewBox={`0 0 ${W} ${H}`}
                className="h-auto max-h-[70dvh] w-full cursor-grab select-none active:cursor-grabbing sm:h-full sm:max-h-none"
                // a finger on the fitted tree still scrolls the page; zoomed in, it moves the tree in every direction
                style={{ aspectRatio: `${W} / ${H}`, touchAction: view.k > 1.01 ? 'none' : 'pan-y' }}
                role="img"
                aria-label={t('alphabeta.tree_label')}
                onPointerDown={(e) => {
                  if ((e.target as Element).closest('[data-leaf], [data-node], foreignObject')) return
                  e.currentTarget.setPointerCapture(e.pointerId)
                  drag.current = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty }
                }}
                onPointerMove={(e) => {
                  const d = drag.current
                  if (!d) return
                  const u = unit()
                  setView((v) => ({ ...v, tx: d.tx + (e.clientX - d.x) / u, ty: d.ty + (e.clientY - d.y) / u }))
                }}
                onPointerUp={() => (drag.current = null)}
                onPointerCancel={() => (drag.current = null)}
                onWheel={(e) => {
                  if (!e.ctrlKey && !e.metaKey) return
                  e.preventDefault()
                  const r = e.currentTarget.getBoundingClientRect()
                  const u = unit()
                  const ox = (r.width - W * u) / 2
                  const oy = (r.height - H * u) / 2
                  zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, (e.clientX - r.left - ox) / u, (e.clientY - r.top - oy) / u)
                }}
              >
                <defs>
                  <marker id="ab-arrow" viewBox="0 0 10 10" refX="20" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
                  </marker>
                </defs>
                <g transform={`translate(${view.tx} ${view.ty}) scale(${view.k})`}>
                {/* 边 */}
                {edges.map((e) => {
                  const p = nodes[e.from]
                  const c = nodes[e.to]
                  const key = `${e.from}->${e.to}`
                  const isPruned = step.prunedEdges.includes(key)
                  const isActive = step.activeEdge === key
                  const isVisited = step.visited.includes(e.to) && !isPruned
                  let stroke = '#94a3b8'
                  let width = 3
                  let dash = '0'
                  if (isPruned) { stroke = '#e11d48'; width = 2.5; dash = '6,5' }
                  else if (isActive) { stroke = '#1e3a8a'; width = 6 }
                  else if (isVisited) { stroke = '#3b82f6'; width = 4 }
                  return (
                    <g key={key}>
                      <line x1={p.x} y1={p.y} x2={c.x} y2={c.y} stroke={stroke} strokeWidth={width} strokeDasharray={dash} markerEnd="url(#ab-arrow)" />
                      {isPruned && (
                        <g transform={`translate(${(p.x + c.x) / 2}, ${(p.y + c.y) / 2})`}>
                          <circle r={10} fill="#e11d48" />
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
                  let fill = '#ffffff'
                  let stroke = '#64748b'
                  let text = '#334155'
                  if (isPruned) { fill = '#fff1f2'; stroke = '#e11d48'; text = '#be123c' }
                  else if (isActive) { fill = '#1e40af'; stroke = '#1e3a8a'; text = '#ffffff' }
                  else if (isVisited) { fill = '#dbeafe'; stroke = '#3b82f6'; text = '#1e40af' }
  
                  if (node.type === 'leaf') {
                    return (
                      <g
                        key={node.id}
                        data-leaf
                        className="cursor-pointer [&:hover>circle]:stroke-blue-500"
                        onClick={() => {
                          if (editMode) return setSelected(tree.paths[node.id])
                          setEditInBar(unit() < 0.75)
                          setEditing(node.id)
                        }}
                      >
                        {selectedId === node.id && <circle cx={node.x} cy={node.y} r={32} fill="#dbeafe" stroke="#2563eb" strokeWidth={2.5} strokeDasharray="5,4" />}
                        {/* a bigger target than the circle, for a finger on a small tree */}
                        <rect x={node.x - 40} y={node.y - 40} width={80} height={80} fill="transparent" />
                        <circle cx={node.x} cy={node.y} r={22} fill={fill} stroke={stroke} strokeWidth={3} strokeDasharray={isPruned ? '5,4' : undefined} />
                        {(editing !== node.id || editInBar) && <text x={node.x} y={node.y + 5} textAnchor="middle" fontSize={14} fontWeight={800} fill={isPruned ? '#be123c' : isActive ? '#ffffff' : '#0f172a'}>{node.value}</text>}
                        <text x={node.x + 28} y={node.y + 4} fontSize={11} fontFamily="monospace" fill="#475569" fontWeight={700}>{node.id}</text>
                        {editing === node.id && !editInBar && (
                          <foreignObject x={node.x - 22} y={node.y - 22} width={44} height={44} onClick={(e) => e.stopPropagation()}>
                            <input {...leafInput(node)} className="block h-full w-full rounded-full border-[3px] border-blue-600 bg-white p-0 text-center text-[14px] font-extrabold text-slate-900 outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none" />
                          </foreignObject>
                        )}
                      </g>
                    )
                  }
                  const isMax = node.type === 'max'
                  const points = isMax
                    ? `${node.x},${node.y - 26} ${node.x - 26},${node.y + 18} ${node.x + 26},${node.y + 18}`
                    : `${node.x - 26},${node.y - 18} ${node.x + 26},${node.y - 18} ${node.x},${node.y + 26}`
                  return (
                    <g
                      key={node.id}
                      data-node={editMode ? '' : undefined}
                      className={editMode ? 'cursor-pointer [&:hover>polygon]:stroke-blue-500' : undefined}
                      onClick={editMode ? () => setSelected(tree.paths[node.id]) : undefined}
                    >
                      {editMode && <rect x={node.x - 36} y={node.y - 36} width={72} height={72} fill="transparent" />}
                      {selectedId === node.id && <circle cx={node.x} cy={node.y} r={40} fill="#dbeafe" stroke="#2563eb" strokeWidth={2.5} strokeDasharray="5,4" />}
                      <polygon points={points} fill={fill} stroke={stroke} strokeWidth={3} strokeLinejoin="round" strokeDasharray={isPruned ? '5,4' : undefined} />
                      <text x={node.x} y={node.y + (isMax ? 10 : -2)} textAnchor="middle" fontSize={9} fontWeight={800} fill={text}>{isMax ? '▲MAX' : '▼MIN'}</text>
                      <g transform={`translate(${node.x}, ${node.y + (isMax ? -40 : -34)})`}>
                        <rect x={-56} y={-12} width={112} height={21} rx={6} fill="#ffffff" stroke="#cbd5e1" />
                        <text textAnchor="middle" dy={3} fontSize={10.5} fontWeight={700} fontFamily="monospace">
                          <tspan fill="#1d4ed8">α:{fmt(a)}</tspan> <tspan fill="#b45309">β:{fmt(b)}</tspan>
                        </text>
                      </g>
                      <text x={node.x + 32} y={node.y + 2} fontSize={11} fontFamily="monospace" fill="#475569" fontWeight={700}>{node.id}</text>
                      {value !== null && (
                        <g transform={`translate(${node.x}, ${node.y + (isMax ? 34 : 40)})`}>
                          <rect x={-15} y={-10} width={30} height={18} rx={5} fill="#60a5fa" />
                          <text textAnchor="middle" dy={3} fontSize={11} fontWeight="bold" fill="white">{value}</text>
                        </g>
                      )}
                    </g>
                  )
                })}
                </g>
              </svg>
              {editing && editInBar && nodes[editing] && (
                <label className="absolute inset-x-3 bottom-12 z-20 flex items-center justify-between gap-3 rounded-xl border border-blue-300 bg-white px-3 py-2 text-sm font-semibold text-slate-800 shadow-lg">
                  {t('alphabeta.edit_leaf', { id: editing })}
                  <input {...leafInput(nodes[editing])} className="h-9 w-20 rounded-lg border-2 border-blue-600 bg-white text-center font-mono text-base font-bold outline-none" />
                </label>
              )}
            </div>
          </section>

          {/* right: control deck + live explanation */}
          <section className="flex min-w-0 flex-col gap-5 lg:col-span-4">
            <div className="flex flex-col gap-5 rounded-[20px] border border-slate-200 bg-white p-5">
              <h3 className={heading}><Play size={15} className="text-blue-600" aria-hidden /> {t('alphabeta.control')}</h3>
              <div className="grid grid-cols-4 gap-2">
                <button onClick={() => { setStepIdx(0); setPlaying(false) }} className="flex items-center justify-center rounded-xl bg-slate-100 p-3 text-slate-700 transition hover:scale-105 hover:bg-slate-200" title={t('alphabeta.first')} aria-label={t('alphabeta.first')}><ChevronsLeft size={18} /></button>
                <button onClick={() => { setStepIdx((i) => Math.max(0, i - 1)); setPlaying(false) }} className="flex items-center justify-center rounded-xl bg-slate-100 p-3 text-slate-700 transition hover:scale-105 hover:bg-slate-200" title={t('alphabeta.prev')} aria-label={t('alphabeta.prev')}><ChevronLeft size={18} /></button>
                <button onClick={() => { if (safeIdx >= steps.length - 1) setStepIdx(0); setPlaying((p) => !p) }} className="flex items-center justify-center rounded-xl bg-blue-600 p-3 text-white shadow-lg shadow-blue-600/25 transition hover:scale-105 hover:bg-blue-700" title={t('alphabeta.play')} aria-label={t('alphabeta.play')}>{playing ? <Pause size={18} /> : <Play size={18} className="fill-white" />}</button>
                <button onClick={() => { setStepIdx((i) => Math.min(steps.length - 1, i + 1)); setPlaying(false) }} className="flex items-center justify-center rounded-xl bg-slate-100 p-3 text-slate-700 transition hover:scale-105 hover:bg-slate-200" title={t('alphabeta.next')} aria-label={t('alphabeta.next')}><ChevronRight size={18} /></button>
              </div>
              <div className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs text-slate-500"><span>{t('alphabeta.progress')}</span><span className="font-mono font-bold text-blue-700">{safeIdx + 1} / {steps.length}</span></div>
                  <input type="range" min={0} max={steps.length - 1} value={safeIdx} aria-label={t('alphabeta.progress')} onChange={(e) => { setStepIdx(parseInt(e.target.value)); setPlaying(false) }} className="w-full cursor-pointer accent-blue-600" />
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs text-slate-500"><span>{t('alphabeta.speed')}</span><span className="font-mono text-slate-600">{(speed / 1000).toFixed(1)}s</span></div>
                  <input type="range" min={200} max={2400} step={200} value={speed} aria-label={t('alphabeta.speed')} onChange={(e) => setSpeed(parseInt(e.target.value))} className="w-full cursor-pointer accent-blue-600" />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 border-t border-slate-200 pt-4 text-center">
                <div><p className="text-[10px] uppercase text-slate-500">{t('alphabeta.evaluated')}</p><p className="text-xl font-black text-blue-600">{counts.evaluated}/{counts.total}</p></div>
                <div><p className="text-[10px] uppercase text-slate-500">{t('alphabeta.pruned')}</p><p className="text-xl font-black text-rose-600">{counts.prunedLeaves}</p></div>
              </div>
            </div>

            <div className="flex min-h-[260px] flex-1 flex-col rounded-[20px] border border-slate-200 bg-white p-5">
              <h3 className={`${heading} border-b border-slate-200 pb-3`}><FileText size={15} className="text-blue-600" aria-hidden /> {t('alphabeta.detail')}</h3>
              <div className="flex flex-1 flex-col justify-between gap-4 pt-4">
                <p className="text-sm leading-relaxed text-slate-700" aria-live="polite">{explanation}</p>
                <div className={`flex flex-col gap-2 rounded-xl border p-3.5 transition-colors ${mathBoxClass}`}>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{t('alphabeta.range_label')}</div>
                  <div className="flex items-center justify-around font-mono text-xs">
                    <div className="flex flex-col items-center"><span className="mb-0.5 text-[10px] text-slate-500">α ({t('alphabeta.lower')})</span><span className="text-base font-bold text-blue-600">{fmt(step.alpha)}</span></div>
                    <div className="text-lg text-slate-500" aria-hidden>|</div>
                    <div className="flex flex-col items-center"><span className="mb-0.5 text-[10px] text-slate-500">β ({t('alphabeta.upper')})</span><span className="text-base font-bold text-amber-600">{fmt(step.beta)}</span></div>
                  </div>
                  <div className="mt-1 border-t border-slate-200 pt-2 text-center text-xs font-semibold italic text-slate-600">{formula}</div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* step trace: click any step to jump there */}
        <section className={`${panel} flex h-[22rem] flex-col gap-3 p-5 lg:h-[260px]`}>
          <h3 className={`${heading} border-b border-slate-200 pb-2`}><ListOrdered size={15} className="text-blue-600" aria-hidden /> {t('alphabeta.timeline')}</h3>
          <ol ref={listRef} className="relative flex flex-1 flex-col gap-1 overflow-y-auto pr-2">
            {steps.map((s, k) => (
              <li key={k} data-step={k}>
                <button
                  type="button"
                  onClick={() => { setStepIdx(k); setPlaying(false) }}
                  aria-current={k === safeIdx ? 'step' : undefined}
                  className={`flex w-full items-start gap-3 rounded-lg px-2.5 py-1.5 text-left text-xs transition ${k === safeIdx ? 'bg-blue-50 ring-1 ring-blue-300' : k < safeIdx ? 'text-slate-500 hover:bg-slate-50' : 'text-slate-700 hover:bg-slate-50'}`}
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
