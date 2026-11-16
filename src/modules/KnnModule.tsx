import { useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import { m } from 'framer-motion'
import { LineChart as LineIcon, MousePointer2, Pencil, RotateCcw, Scale } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { LineChart } from '../components/LineChart'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { KNN_RAW_DATA } from '../data/constants'
import { computeKnn, knnStats, knnView, looErrorCurve, nudgeTestPoint, scaleHeights } from '../lib/knn'
import type { KnnView } from '../lib/knn'
import { niceTicks } from '../lib/chart'
import type { KnnPoint, StudentClass, TestPoint } from '../types'

const WIDTH = 400
const HEIGHT = 400
const K_VALUES = Array.from({ length: 15 }, (_, i) => i + 1)
/** the plot, in cm and kg (the data stays in these; mm only changes the numbers the distance sees) */
const RANGE = { minH: 155, maxH: 175, minW: 55, maxW: 70 }
const PLOT = { ...RANGE, meanH: 0, meanW: 0, stdH: 1, stdW: 1 }
const START_TEST: TestPoint = { h: 162, w: 65 }
/** how far past the edge (in plot units) a dragged point has to go to be removed */
const OUT = 14

type Unit = 'cm' | 'mm'
type Point = KnnPoint & { id: number }
type Drag = { kind: 'test' } | { kind: 'point'; id: number } | { kind: 'new'; s: StudentClass }

/** where points may be placed, in cm and kg */
const PLACE = { minH: 145, maxH: 185, minW: 45, maxW: 80 }
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
/** a tick label: whole numbers as they are, z-scores with one decimal at most */
const tick = (v: number) => String(+v.toFixed(1))
const pct = (v: number) => `${Math.round(v * 100)}%`
const COLOR: Record<StudentClass, string> = { M: '#3b82f6', L: '#ef4444' }
const fresh = (): Point[] => KNN_RAW_DATA.map((p, id) => ({ ...p, id }))
const num = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2))

const ARROWS: Record<string, [number, number]> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, 1],
  ArrowDown: [0, -1],
}

// KNN Module - Interactive Visualization
export const KnnModule = () => {
  const [k, setK] = useState(5)
  const [isStandardized, setIsStandardized] = useState(false)
  const [unit, setUnit] = useState<Unit>('cm')
  const [points, setPoints] = useState<Point[]>(fresh)
  const [testPoint, setTestPoint] = useState<TestPoint>(START_TEST)
  const [editing, setEditing] = useState(false)
  const [drag, setDrag] = useState<Drag | null>(null)
  /** where a new point being dragged in from the palette is, and whether a dragged point is past the edge */
  const [ghost, setGhost] = useState<{ h: number; w: number; out: boolean } | null>(null)
  /** the class a press on the plot adds while editing */
  const [addClass, setAddClass] = useState<StudentClass>('M')
  const svgRef = useRef<SVGSVGElement>(null)
  const nextId = useRef(KNN_RAW_DATA.length)
  const { t } = useTranslation()

  // the distance sees heights in the chosen unit; the plot stays in cm
  const factor = unit === 'mm' ? 10 : 1
  const stats = useMemo(() => knnStats(points, factor, RANGE), [points, factor])
  const scaled = useMemo(() => points.map((p) => scaleHeights(p, factor)), [points, factor])
  const processedData = useMemo(
    () => computeKnn(scaled, scaleHeights(testPoint, factor), k, isStandardized, stats),
    [scaled, testPoint, factor, k, isStandardized, stats]
  )

  // 真实的留一法误差曲线（随标准化开关、单位和数据重新计算）
  const looCurve = useMemo(
    () => looErrorCurve(scaled, K_VALUES, isStandardized, stats).map((p) => ({ x: p.k, y: p.error })),
    [scaled, isStandardized, stats]
  )
  const best = useMemo(() => looCurve.reduce((a, b) => (b.y < a.y ? b : a)), [looCurve])

  // the plot in the distance's own units, one unit as long on both axes (cm & kg, mm & kg, or z & z). While something is
  // dragged it stays as it was, or editing a point would move the axes under the pointer.
  const liveView = useMemo(() => knnView(isStandardized, factor, stats, RANGE), [isStandardized, factor, stats])
  const [frozen, setFrozen] = useState<KnnView | null>(null)
  const view = frozen ?? liveView
  const px = (h: number) => ((view.toU(h) - view.u0) / view.span) * WIDTH
  const py = (w: number) => HEIGHT - ((view.toV(w) - view.v0) / view.span) * HEIGHT
  const motion = { duration: drag ? 0 : 0.45 }

  /** pointer → plot coordinates (the canvas is scaled on small screens) */
  const at = (e: { clientX: number; clientY: number }) => {
    const rect = svgRef.current!.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * WIDTH
    const y = ((e.clientY - rect.top) / rect.height) * HEIGHT
    const out = x < -OUT || x > WIDTH + OUT || y < -OUT || y > HEIGHT + OUT
    const u = view.u0 + (clamp(x, 0, WIDTH) / WIDTH) * view.span
    const v = view.v0 + ((HEIGHT - clamp(y, 0, HEIGHT)) / HEIGHT) * view.span
    const h = Math.round(clamp(view.fromU(u), PLACE.minH, PLACE.maxH))
    const w = Math.round(clamp(view.fromV(v), PLACE.minW, PLACE.maxW))
    return { h, w, out, inside: x >= 0 && x <= WIDTH && y >= 0 && y <= HEIGHT }
  }

  const begin = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!svgRef.current) return
    setFrozen(liveView)
    const target = (e.target as Element).closest('[data-point]')
    const p = at(e)
    if (target && editing) {
      setDrag({ kind: 'point', id: Number(target.getAttribute('data-point')) })
    } else if (editing && !(e.target as Element).closest('[data-test]')) {
      // a press on empty space adds a point there, and keeps it under the pointer until release
      const id = nextId.current++
      setPoints((ps) => [...ps, { id, h: p.h, w: p.w, s: addClass }])
      setDrag({ kind: 'point', id })
    } else {
      setTestPoint({ h: p.h, w: p.w })
      setDrag({ kind: 'test' })
    }
    svgRef.current.setPointerCapture(e.pointerId)
  }
  const move = (e: ReactPointerEvent) => {
    if (!drag || !svgRef.current) return
    const p = at(e)
    if (drag.kind === 'test') setTestPoint({ h: p.h, w: p.w })
    else if (drag.kind === 'point') {
      setPoints((ps) => ps.map((q) => (q.id === drag.id ? { ...q, h: p.h, w: p.w } : q)))
      setGhost(p.out ? { h: p.h, w: p.w, out: true } : null)
    } else setGhost(p.inside ? { h: p.h, w: p.w, out: false } : null)
  }
  const end = (e: ReactPointerEvent) => {
    if (drag && svgRef.current) {
      const p = at(e)
      if (drag.kind === 'point' && p.out) setPoints((ps) => ps.filter((q) => q.id !== drag.id))
      if (drag.kind === 'new' && p.inside) {
        const id = nextId.current++
        setPoints((ps) => [...ps, { id, h: p.h, w: p.w, s: drag.s }])
      }
    }
    setDrag(null)
    setGhost(null)
    setFrozen(null)
  }

  // the palette: press a chip and drag it onto the plot (or just press it, then press the plot)
  const fromPalette = (s: StudentClass) => (e: ReactPointerEvent<HTMLButtonElement>) => {
    setAddClass(s)
    setFrozen(liveView)
    setDrag({ kind: 'new', s })
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const handleSvgKey = (e: ReactKeyboardEvent<SVGSVGElement>) => {
    const d = ARROWS[e.key]
    if (!d) return
    e.preventDefault()
    setTestPoint((p) => nudgeTestPoint(p, d[0], d[1], PLOT))
  }

  const tx = px(testPoint.h)
  const ty = py(testPoint.w)
  const predLabel = processedData.prediction === 'M' ? t('knn.class_m') : t('knn.class_l')
  // every point as far as the K-th neighbour: a circle, as both axes are in the distance's units
  const radius = (processedData.radiusDist / view.span) * WIDTH
  const uTicks = niceTicks(view.u0, view.u0 + view.span, 5).filter((v) => v > view.u0 && v < view.u0 + view.span)
  const vTicks = niceTicks(view.v0, view.v0 + view.span, 5).filter((v) => v > view.v0 && v < view.v0 + view.span)
  const uUnit = isStandardized ? 'z' : unit
  const byId = new Map(points.map((p, i) => [p.id, i]))
  const seg = (on: boolean) => `px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${on ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-[2rem] p-5 sm:p-6 shadow-sm">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-3 gap-3">
            <h4 className="text-[11px] font-black text-slate-400 uppercase flex items-center gap-2">
              <MousePointer2 size={14} className="text-blue-500" aria-hidden /> {editing ? t('knn.edit_hint') : t('knn.click_instruction')}
            </h4>
            <button
              type="button"
              onClick={() => setEditing((v) => !v)}
              aria-pressed={editing}
              className={`flex shrink-0 items-center gap-1.5 rounded-xl px-3.5 py-1.5 text-[11px] font-black uppercase transition ${editing ? 'bg-blue-600 text-white shadow' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
            >
              <Pencil size={13} aria-hidden /> {editing ? t('knn.edit_done') : t('knn.edit')}
            </button>
          </div>
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <div className="flex flex-wrap gap-1 bg-slate-100 p-1 rounded-xl" role="group" aria-label={t('knn.distance_mode')}>
              <button onClick={() => setIsStandardized(false)} aria-pressed={!isStandardized} className={seg(!isStandardized)}>{t('knn.raw_data')}</button>
              <button onClick={() => setIsStandardized(true)} aria-pressed={isStandardized} className={seg(isStandardized)}>{t('knn.standardized')}</button>
            </div>
            <div className="flex flex-wrap gap-1 bg-slate-100 p-1 rounded-xl" role="group" aria-label={t('knn.unit')}>
              <button onClick={() => setUnit('cm')} aria-pressed={unit === 'cm'} className={seg(unit === 'cm')}>{t('knn.unit_cm')}</button>
              <button onClick={() => setUnit('mm')} aria-pressed={unit === 'mm'} className={seg(unit === 'mm')}>{t('knn.unit_mm')}</button>
            </div>
          </div>

          {editing && (
            <div className="mb-4 flex flex-wrap items-center gap-2 rounded-2xl border border-blue-100 bg-blue-50/60 p-2.5">
              {(['M', 'L'] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  onPointerDown={fromPalette(s)}
                  onPointerMove={move}
                  onPointerUp={end}
                  aria-pressed={addClass === s}
                  className={`flex touch-none select-none items-center gap-2 rounded-xl border-2 bg-white px-3 py-1.5 text-[11px] font-black cursor-grab active:cursor-grabbing ${addClass === s ? '' : 'border-transparent'}`}
                  style={addClass === s ? { borderColor: COLOR[s] } : undefined}
                >
                  <i className="h-3 w-3 rounded-full" style={{ background: COLOR[s] }} aria-hidden />
                  {t(s === 'M' ? 'knn.add_m' : 'knn.add_l')}
                </button>
              ))}
              <span className="text-[11px] text-slate-500">{t('knn.n_points', { n: points.length })}</span>
              <button type="button" onClick={() => setPoints(fresh())} className="ml-auto flex items-center gap-1.5 rounded-xl px-2.5 py-1.5 text-[11px] font-bold text-slate-600 hover:bg-white">
                <RotateCcw size={12} aria-hidden /> {t('knn.reset_data')}
              </button>
            </div>
          )}

          {/* 网格布局给两条坐标轴标签留位置，避免旋转文字压到画布 */}
          <div className="grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-1 w-full max-w-[min(470px,calc(100dvh-470px))] min-w-[240px] mx-auto">
            <div className="[writing-mode:vertical-rl] rotate-180 text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] whitespace-nowrap" aria-hidden>{t('knn.axis_weight_unit', { u: isStandardized ? 'z' : 'kg' })}</div>
            <div className="aspect-square w-full">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className={`w-full h-full overflow-visible border-l border-b border-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded-sm select-none ${editing ? 'cursor-copy' : 'cursor-crosshair'}`}
              // while editing, a finger on the plot moves points instead of the page
              style={{ touchAction: editing || drag ? 'none' : 'pan-y' }}
              onPointerDown={begin}
              onPointerMove={move}
              onPointerUp={end}
              onPointerCancel={end}
              onKeyDown={handleSvgKey}
              tabIndex={0}
              role="application"
              aria-label={t('knn.canvas_label', { h: testPoint.h, w: testPoint.w, k, cls: predLabel })}
            >
              <defs>
                <clipPath id="knn-plot"><rect width={WIDTH} height={HEIGHT} /></clipPath>
              </defs>
              {/* grid and ticks in the view's units: cm or mm and kg, or z-scores */}
              {uTicks.map((u) => {
                const x = ((u - view.u0) / view.span) * WIDTH
                return (
                  <g key={`u${u}`}>
                    <line x1={x} y1={0} x2={x} y2={HEIGHT} stroke="#f1f5f9" strokeWidth="1" />
                    <text x={x} y={HEIGHT - 5} textAnchor="middle" className="fill-slate-300 text-[10px] font-bold">{tick(u)}</text>
                  </g>
                )
              })}
              {vTicks.map((v) => {
                const y = HEIGHT - ((v - view.v0) / view.span) * HEIGHT
                return (
                  <g key={`v${v}`}>
                    <line x1={0} y1={y} x2={WIDTH} y2={y} stroke="#f1f5f9" strokeWidth="1" />
                    <text x={5} y={y - 4} className="fill-slate-300 text-[10px] font-bold">{tick(v)}</text>
                  </g>
                )
              })}

              {/* every point as close as the K-th neighbour: a circle in the distance's own units */}
              {processedData.topK.length > 0 && (
                <m.circle clipPath="url(#knn-plot)" initial={false} animate={{ cx: tx, cy: ty, r: radius }} transition={motion} fill="rgba(59, 130, 246, 0.05)" stroke="#3b82f6" strokeWidth="1" strokeDasharray="4 4" />
              )}

              {/* 邻居连线 */}
              {processedData.topK.map((n) => {
                const p = points[n.idx]
                return (
                  <m.line
                    key={p.id}
                    initial={{ pathLength: 0, x1: tx, y1: ty, x2: px(p.h), y2: py(p.w) }}
                    animate={{ pathLength: 1, x1: tx, y1: ty, x2: px(p.h), y2: py(p.w) }}
                    transition={motion}
                    stroke={COLOR[p.s]} strokeWidth="2" opacity="0.6"
                  />
                )
              })}

              {/* 数据点 */}
              {points.map((p) => {
                const rank = processedData.neighborsMap.get(byId.get(p.id)!)
                const dragged = drag?.kind === 'point' && drag.id === p.id
                const leaving = dragged && ghost?.out
                return (
                  <m.g key={p.id} data-point={p.id} className={editing ? 'cursor-grab' : undefined} initial={false} animate={{ x: px(p.h), y: py(p.w) }} transition={motion}>
                    {editing && <circle r={14} fill="transparent" />}
                    <m.circle
                      r={rank ? 8 : editing ? 6 : 4}
                      fill={COLOR[p.s]}
                      stroke={leaving ? '#0f172a' : '#fff'}
                      strokeWidth={leaving ? 2 : 1}
                      strokeDasharray={leaving ? '3 2' : undefined}
                      animate={{ opacity: leaving ? 0.35 : rank || editing ? 1 : 0.25, scale: rank ? 1.3 : 1 }}
                    />
                    {rank && !dragged && (
                      <text y={18} textAnchor="middle" className="text-[10px] font-black fill-slate-800 pointer-events-none">
                        #{rank}
                      </text>
                    )}
                    {leaving && <text y={-14} textAnchor="middle" className="text-[10px] font-black fill-slate-700 pointer-events-none">{t('knn.release_delete')}</text>}
                  </m.g>
                )
              })}
              {drag?.kind === 'new' && ghost && <circle cx={px(ghost.h)} cy={py(ghost.w)} r={7} fill={COLOR[drag.s]} opacity={0.6} stroke="#fff" strokeWidth={2} />}

              <m.g data-test animate={{ x: tx, y: ty }} initial={false} transition={drag?.kind === 'test' ? { duration: 0 } : motion} className="cursor-move">
                <circle r="10" fill="#10b981" stroke="#fff" strokeWidth="4" />
                <text y="-18" textAnchor="middle" className="text-[10px] font-black fill-emerald-600 uppercase pointer-events-none">{t('knn.test_sample')}</text>
              </m.g>
            </svg>
            </div>
            <div aria-hidden />
            <div className="text-center text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] whitespace-nowrap" aria-hidden>{t('knn.axis_height_unit', { u: uUnit })}</div>
          </div>
          <div className="mt-4 grid gap-1.5 text-center text-[11px] text-slate-500">
            <p className="font-mono">
              {t('knn.stats_line', { mh: num(+stats.meanH.toFixed(2)), sh: stats.stdH.toFixed(2), mw: num(+stats.meanW.toFixed(2)), sw: stats.stdW.toFixed(2), u: unit })}
            </p>
            <p className={unit === 'mm' && !isStandardized ? 'text-amber-700 font-semibold' : 'text-slate-400'}>{t(isStandardized ? 'knn.std_note' : unit === 'mm' ? 'knn.unit_note_mm' : 'knn.unit_note_cm')}</p>
            <p className="text-[10px] text-slate-400 coarse:hidden">{t('knn.keyboard_hint')}</p>
          </div>
        </div>

        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-900 rounded-[2rem] p-5 text-white shadow-2xl relative overflow-hidden">
             <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
               <LineIcon size={14} className="text-blue-500" aria-hidden /> {t('knn.model_complexity')}
             </h4>
             {/* capped so the lab fits one screen; the chart keeps its proportions */}
             <div className="[&_svg]:mx-auto [&_svg]:max-h-[190px]">
             <LineChart
               data={looCurve}
               referenceX={k}
               highlightX={best.x}
               formatY={pct}
               xLabel="K"
               ariaLabel={t('knn.loo_chart_label', { k: best.x, err: pct(best.y) })}
             />
             </div>
             <p className="text-[10px] text-slate-400 mt-3 leading-relaxed">
               {t('knn.loo_hint', { n: points.length, k: best.x, err: pct(best.y) })}
             </p>
             <div className="mt-3 grid grid-cols-2 gap-4 text-center border-t border-slate-800 pt-3">
                <div><p className="text-[9px] text-slate-500 uppercase">{t('knn.low_k')}</p><p className="text-[10px] text-red-400 font-bold">{t('knn.overfitting')}</p></div>
                <div><p className="text-[9px] text-slate-500 uppercase">{t('knn.high_k')}</p><p className="text-[10px] text-orange-400 font-bold">{t('knn.underfitting')}</p></div>
             </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-[2rem] p-5 shadow-sm">
            <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-widest mb-3">{t('knn.voting_panel')}</h4>
            <div className="space-y-4">
               <div className="flex items-center justify-between">
                  <label htmlFor="knn-k" className="text-xs font-bold text-slate-600">{t('knn.k_value')}</label>
                  <div className="flex items-center gap-4">
                     <input id="knn-k" type="range" min="1" max="15" value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-24 accent-blue-600 cursor-ew-resize" />
                     <span className="text-2xl font-black text-blue-600 font-mono w-8 text-right" aria-hidden>{k}</span>
                  </div>
               </div>
               <div className="grid grid-cols-2 gap-4" aria-live="polite">
                  <div className={`p-3 rounded-2xl border-2 transition ${processedData.prediction === 'M' ? 'bg-blue-50 border-blue-500' : 'bg-slate-50 border-transparent opacity-50'}`}>
                     <p className="text-[9px] font-black uppercase text-blue-600 mb-1">{t('knn.m_vote')}</p>
                     <p className="text-2xl font-black text-slate-800">{processedData.mCount}</p>
                  </div>
                  <div className={`p-3 rounded-2xl border-2 transition ${processedData.prediction === 'L' ? 'bg-red-50 border-red-500' : 'bg-slate-50 border-transparent opacity-50'}`}>
                     <p className="text-[9px] font-black uppercase text-red-600 mb-1">{t('knn.l_vote')}</p>
                     <p className="text-2xl font-black text-slate-800">{processedData.lCount}</p>
                  </div>
               </div>
               {processedData.isTie && (
                 <m.p
                   initial={{ opacity: 0, y: -4 }}
                   animate={{ opacity: 1, y: 0 }}
                   className="flex items-start gap-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-xl p-3"
                 >
                   <Scale size={14} className="mt-0.5 shrink-0" aria-hidden />
                   {t('knn.tie', { m: processedData.mCount, l: processedData.lCount, cls: predLabel })}
                 </m.p>
               )}
            </div>
          </div>
        </div>
      </div>
      <SeniorAdvice content={<Trans i18nKey="knn.advice" components={{
        1: <Latex formula="K" />,
        3: <strong className="font-bold text-amber-700" />,
        5: <Latex formula="K" />,
      }} />} />
    </div>
  )
}
