import React, { useMemo, useRef, useState } from 'react'
import { m } from 'framer-motion'
import { LineChart as LineIcon, MousePointer2, Scale } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { LineChart } from '../components/LineChart'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { KNN_RAW_DATA } from '../data/constants'
import { computeKnn, looErrorCurve, nudgeTestPoint, scaleLinear, svgToDataPoint } from '../lib/knn'
import type { KnnStats, TestPoint } from '../types'

const WIDTH = 400
const HEIGHT = 400
const K_VALUES = Array.from({ length: 15 }, (_, i) => i + 1)
const STATS: KnnStats = { meanH: 164, meanW: 62.33, stdH: 4.33, stdW: 2.63, minH: 155, maxH: 175, minW: 55, maxW: 70 }

const px = (h: number) => scaleLinear(h, STATS.minH, STATS.maxH, WIDTH)
const py = (w: number) => HEIGHT - scaleLinear(w, STATS.minW, STATS.maxW, HEIGHT)
const pct = (v: number) => `${Math.round(v * 100)}%`

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
  const [testPoint, setTestPoint] = useState<TestPoint>({ h: 161, w: 61 })
  const svgRef = useRef<SVGSVGElement>(null)
  const { t } = useTranslation()

  const processedData = useMemo(
    () => computeKnn(KNN_RAW_DATA, testPoint, k, isStandardized, STATS),
    [k, isStandardized, testPoint]
  )

  // 真实的留一法误差曲线（随标准化开关重新计算）
  const looCurve = useMemo(
    () => looErrorCurve(KNN_RAW_DATA, K_VALUES, isStandardized, STATS).map((p) => ({ x: p.k, y: p.error })),
    [isStandardized]
  )
  const best = useMemo(() => looCurve.reduce((a, b) => (b.y < a.y ? b : a)), [looCurve])

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    // 按实际渲染尺寸换算回 viewBox 坐标（画布在小屏上会被缩放）
    const x = ((e.clientX - rect.left) / rect.width) * WIDTH
    const y = ((e.clientY - rect.top) / rect.height) * HEIGHT
    setTestPoint(svgToDataPoint(x, y, WIDTH, HEIGHT, STATS))
  }

  const handleSvgKey = (e: React.KeyboardEvent<SVGSVGElement>) => {
    const d = ARROWS[e.key]
    if (!d) return
    e.preventDefault()
    setTestPoint((p) => nudgeTestPoint(p, d[0], d[1], STATS))
  }

  const tx = px(testPoint.h)
  const ty = py(testPoint.w)
  const predLabel = processedData.prediction === 'M' ? t('knn.class_m') : t('knn.class_l')

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-[2rem] p-5 sm:p-8 shadow-sm">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
            <h4 className="text-[11px] font-black text-slate-400 uppercase flex items-center gap-2">
              <MousePointer2 size={14} className="text-blue-500" aria-hidden /> {t('knn.click_instruction')}
            </h4>
            <div className="flex flex-wrap gap-2 bg-slate-100 p-1 rounded-xl shrink-0" role="group" aria-label={t('knn.distance_mode')}>
               <button onClick={() => setIsStandardized(false)} aria-pressed={!isStandardized} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${!isStandardized ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`}>{t('knn.raw_data')}</button>
               <button onClick={() => setIsStandardized(true)} aria-pressed={isStandardized} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${isStandardized ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`}>{t('knn.standardized')}</button>
            </div>
          </div>

          {/* 网格布局给两条坐标轴标签留位置，避免旋转文字压到画布 */}
          <div className="grid grid-cols-[auto_1fr] items-center gap-x-2 gap-y-1 w-full max-w-[470px] mx-auto">
            <div className="[writing-mode:vertical-rl] rotate-180 text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] whitespace-nowrap" aria-hidden>{t('knn.axis_weight')}</div>
            <div className="aspect-square w-full">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className="w-full h-full cursor-crosshair border-l border-b border-slate-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded-sm"
              onClick={handleSvgClick}
              onKeyDown={handleSvgKey}
              tabIndex={0}
              role="application"
              aria-label={t('knn.canvas_label', { h: testPoint.h, w: testPoint.w, k, cls: predLabel })}
            >
              {[...Array(5)].map((_, i) => (
                <React.Fragment key={i}>
                  <line x1={0} y1={(i * HEIGHT) / 4} x2={WIDTH} y2={(i * HEIGHT) / 4} stroke="#f1f5f9" strokeWidth="1" />
                  <line x1={(i * WIDTH) / 4} y1={0} x2={(i * WIDTH) / 4} y2={HEIGHT} stroke="#f1f5f9" strokeWidth="1" />
                </React.Fragment>
              ))}

              {/* 决策圆（仅原始模式下几何直观；标准化后等距线是椭圆） */}
              {!isStandardized && (
                <circle
                  cx={tx}
                  cy={ty}
                  r={scaleLinear(STATS.minH + processedData.radiusDist, STATS.minH, STATS.maxH, WIDTH)}
                  fill="rgba(59, 130, 246, 0.05)"
                  stroke="#3b82f6"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
              )}

              {/* 邻居连线 */}
              {processedData.topK.map((n) => (
                <m.line
                  key={n.idx}
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                  x1={tx} y1={ty} x2={px(n.h)} y2={py(n.w)}
                  stroke={n.s === 'M' ? '#3b82f6' : '#ef4444'} strokeWidth="2" opacity="0.6"
                />
              ))}

              {/* 数据点 */}
              {processedData.data.map((d) => {
                const rank = processedData.neighborsMap.get(d.idx)
                return (
                  <g key={d.idx}>
                    <m.circle
                      cx={px(d.h)} cy={py(d.w)}
                      r={rank ? 8 : 4}
                      fill={d.s === 'M' ? '#3b82f6' : '#ef4444'}
                      animate={{ opacity: rank ? 1 : 0.25, scale: rank ? 1.3 : 1 }}
                    />
                    {rank && (
                      <text x={px(d.h)} y={py(d.w) + 18} textAnchor="middle" className="text-[10px] font-black fill-slate-800">
                        #{rank}
                      </text>
                    )}
                  </g>
                )
              })}

              <m.g animate={{ x: tx, y: ty }} initial={false}>
                <circle r="10" fill="#10b981" stroke="#fff" strokeWidth="4" />
                <text y="-18" textAnchor="middle" className="text-[10px] font-black fill-emerald-600 uppercase">{t('knn.test_sample')}</text>
              </m.g>
            </svg>
            </div>
            <div aria-hidden />
            <div className="text-center text-[10px] font-black text-slate-300 uppercase tracking-[0.2em] whitespace-nowrap" aria-hidden>{t('knn.axis_height')}</div>
          </div>
          <p className="mt-4 text-[10px] text-slate-400 text-center">{t('knn.keyboard_hint')}</p>
        </div>

        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900 rounded-[2rem] p-6 text-white shadow-2xl relative overflow-hidden">
             <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
               <LineIcon size={14} className="text-blue-500" aria-hidden /> {t('knn.model_complexity')}
             </h4>
             <LineChart
               data={looCurve}
               referenceX={k}
               highlightX={best.x}
               formatY={pct}
               xLabel="K"
               ariaLabel={t('knn.loo_chart_label', { k: best.x, err: pct(best.y) })}
             />
             <p className="text-[10px] text-slate-400 mt-3 leading-relaxed">
               {t('knn.loo_hint', { n: KNN_RAW_DATA.length, k: best.x, err: pct(best.y) })}
             </p>
             <div className="mt-4 grid grid-cols-2 gap-4 text-center border-t border-slate-800 pt-4">
                <div><p className="text-[9px] text-slate-500 uppercase">{t('knn.low_k')}</p><p className="text-[10px] text-red-400 font-bold">{t('knn.overfitting')}</p></div>
                <div><p className="text-[9px] text-slate-500 uppercase">{t('knn.high_k')}</p><p className="text-[10px] text-orange-400 font-bold">{t('knn.underfitting')}</p></div>
             </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-[2rem] p-6 sm:p-8 shadow-sm">
            <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-widest mb-6">{t('knn.voting_panel')}</h4>
            <div className="space-y-6">
               <div className="flex items-center justify-between">
                  <label htmlFor="knn-k" className="text-xs font-bold text-slate-600">{t('knn.k_value')}</label>
                  <div className="flex items-center gap-4">
                     <input id="knn-k" type="range" min="1" max="15" value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-24 accent-blue-600 cursor-ew-resize" />
                     <span className="text-2xl font-black text-blue-600 font-mono w-8 text-right" aria-hidden>{k}</span>
                  </div>
               </div>
               <div className="grid grid-cols-2 gap-4" aria-live="polite">
                  <div className={`p-4 rounded-2xl border-2 transition ${processedData.prediction === 'M' ? 'bg-blue-50 border-blue-500' : 'bg-slate-50 border-transparent opacity-50'}`}>
                     <p className="text-[9px] font-black uppercase text-blue-600 mb-1">{t('knn.m_vote')}</p>
                     <p className="text-3xl font-black text-slate-800">{processedData.mCount}</p>
                  </div>
                  <div className={`p-4 rounded-2xl border-2 transition ${processedData.prediction === 'L' ? 'bg-red-50 border-red-500' : 'bg-slate-50 border-transparent opacity-50'}`}>
                     <p className="text-[9px] font-black uppercase text-red-600 mb-1">{t('knn.l_vote')}</p>
                     <p className="text-3xl font-black text-slate-800">{processedData.lCount}</p>
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
