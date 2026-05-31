import React, { useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { LineChart as LineIcon, MousePointer2 } from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
} from 'recharts'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { KNN_RAW_DATA } from '../data/constants'
import type { KnnStats, StudentClass, TestPoint } from '../types'

// KNN Module - Interactive Visualization
export const KnnModule = () => {
  const [k, setK] = useState(5)
  const [isStandardized, setIsStandardized] = useState(false)
  const [testPoint, setTestPoint] = useState<TestPoint>({ h: 161, w: 61 })
  const svgRef = useRef<SVGSVGElement>(null)
  const { t } = useTranslation()

  const WIDTH = 400
  const HEIGHT = 400
  const stats = useMemo<KnnStats>(
    () => ({ meanH: 164, meanW: 62.33, stdH: 4.33, stdW: 2.63, minH: 155, maxH: 175, minW: 55, maxW: 70 }),
    []
  )
  const scale = (val: number, min: number, max: number, target: number): number =>
    ((val - min) / (max - min)) * target

  const processedData = useMemo(() => {
    const dataWithDist = KNN_RAW_DATA.map((d) => {
      let dist: number
      if (isStandardized) {
        const sh1 = (d.h - stats.meanH) / stats.stdH
        const sw1 = (d.w - stats.meanW) / stats.stdW
        const sh2 = (testPoint.h - stats.meanH) / stats.stdH
        const sw2 = (testPoint.w - stats.meanW) / stats.stdW
        dist = Math.sqrt(Math.pow(sh1 - sh2, 2) + Math.pow(sw1 - sw2, 2))
      } else {
        dist = Math.sqrt(Math.pow(d.h - testPoint.h, 2) + Math.pow(d.w - testPoint.w, 2))
      }
      return { ...d, dist }
    })

    const sorted = [...dataWithDist].sort((a, b) => a.dist - b.dist)
    const topK = sorted.slice(0, k)
    const neighborsMap = new Map(topK.map((n, idx) => [`${n.h}-${n.w}`, idx + 1]))

    // Radius for decision circle (distance to K-th neighbor)
    const radiusDist = topK.length > 0 ? topK[topK.length - 1].dist : 0

    const mCount = topK.filter((n) => n.s === 'M').length
    const lCount = topK.filter((n) => n.s === 'L').length
    const prediction: StudentClass = mCount >= lCount ? 'M' : 'L'
    return { data: dataWithDist, neighborsMap, radiusDist, prediction, mCount, lCount, topK }
  }, [k, isStandardized, testPoint, stats])

  const performanceData = useMemo(
    () =>
      Array.from({ length: 15 }, (_, i) => ({
        k: i + 1,
        error: 0.1 + Math.pow(i + 1 - 4, 2) * 0.01 + ((Math.sin((i + 1) * 99.71) + 1) / 2) * 0.02,
      })),
    []
  )

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const h = (x / WIDTH) * (stats.maxH - stats.minH) + stats.minH
    const w = stats.maxW - (y / HEIGHT) * (stats.maxW - stats.minW)
    setTestPoint({ h: Math.round(h), w: Math.round(w) })
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-[2rem] p-8 shadow-sm">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
            <h4 className="text-[11px] font-black text-slate-400 uppercase flex items-center gap-2">
              <MousePointer2 size={14} className="text-blue-500" /> {t('knn.click_instruction')}
            </h4>
            <div className="flex flex-wrap gap-2 bg-slate-100 p-1 rounded-xl shrink-0">
               <button onClick={() => setIsStandardized(false)} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${!isStandardized ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`}>{t('knn.raw_data')}</button>
               <button onClick={() => setIsStandardized(true)} className={`px-4 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${isStandardized ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`}>{t('knn.standardized')}</button>
            </div>
          </div>

          <div className="relative aspect-square w-full max-w-[450px] mx-auto">
            <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full h-full cursor-crosshair border-l border-b border-slate-200" onClick={handleSvgClick}>
              {[...Array(5)].map((_, i) => (
                <React.Fragment key={i}>
                  <line x1={0} y1={(i * HEIGHT) / 4} x2={WIDTH} y2={(i * HEIGHT) / 4} stroke="#f1f5f9" strokeWidth="1" />
                  <line x1={(i * WIDTH) / 4} y1={0} x2={(i * WIDTH) / 4} y2={HEIGHT} stroke="#f1f5f9" strokeWidth="1" />
                </React.Fragment>
              ))}

              {/* Decision Circle (Only intuitive in non-standardized mode) */}
              {!isStandardized && (
                <circle
                  cx={scale(testPoint.h, stats.minH, stats.maxH, WIDTH)}
                  cy={HEIGHT - scale(testPoint.w, stats.minW, stats.maxW, HEIGHT)}
                  r={scale(stats.minH + processedData.radiusDist, stats.minH, stats.maxH, WIDTH)}
                  fill="rgba(59, 130, 246, 0.05)"
                  stroke="#3b82f6"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
              )}

              {/* Neighbor Connections */}
              {processedData.topK.map((n, i) => (
                <motion.line
                  key={i}
                  initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                  x1={scale(testPoint.h, stats.minH, stats.maxH, WIDTH)} y1={HEIGHT - scale(testPoint.w, stats.minW, stats.maxW, HEIGHT)}
                  x2={scale(n.h, stats.minH, stats.maxH, WIDTH)} y2={HEIGHT - scale(n.w, stats.minW, stats.maxW, HEIGHT)}
                  stroke={n.s === 'M' ? '#3b82f6' : '#ef4444'} strokeWidth="2" opacity="0.6"
                />
              ))}

              {/* Data Points */}
              {processedData.data.map((d, i) => {
                const rank = processedData.neighborsMap.get(`${d.h}-${d.w}`)
                return (
                  <g key={i}>
                    <motion.circle
                      cx={scale(d.h, stats.minH, stats.maxH, WIDTH)} cy={HEIGHT - scale(d.w, stats.minW, stats.maxW, HEIGHT)}
                      r={rank ? 8 : 4}
                      fill={d.s === 'M' ? '#3b82f6' : '#ef4444'}
                      animate={{
                        opacity: rank ? 1 : 0.25,
                        scale: rank ? 1.3 : 1,
                      }}
                    />
                    {rank && (
                      <text
                        x={scale(d.h, stats.minH, stats.maxH, WIDTH)}
                        y={HEIGHT - scale(d.w, stats.minW, stats.maxW, HEIGHT) + 18}
                        textAnchor="middle" className="text-[10px] font-black fill-slate-800"
                      >
                        #{rank}
                      </text>
                    )}
                  </g>
                )
              })}

              <motion.g animate={{ x: scale(testPoint.h, stats.minH, stats.maxH, WIDTH), y: HEIGHT - scale(testPoint.w, stats.minW, stats.maxW, HEIGHT) }}>
                <circle r="10" fill="#10b981" stroke="#fff" strokeWidth="4" className="shadow-2xl" />
                <text y="-18" textAnchor="middle" className="text-[10px] font-black fill-emerald-600 uppercase">{t('knn.test_sample')}</text>
              </motion.g>
            </svg>
            <div className="absolute -left-12 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] font-black text-slate-300 uppercase tracking-[0.2em]">Weight (kg)</div>
            <div className="absolute bottom-[-28px] left-1/2 -translate-x-1/2 text-[10px] font-black text-slate-300 uppercase tracking-[0.2em]">Height (cm)</div>
          </div>
        </div>

        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900 rounded-[2rem] p-6 text-white shadow-2xl relative overflow-hidden">
             <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-6 flex items-center gap-2">
               <LineIcon size={14} className="text-blue-500" /> {t('knn.model_complexity')}
             </h4>
             <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="k" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '10px' }} />
                    <ReferenceLine x={k} stroke="#fbbf24" strokeDasharray="5 5" />
                    <Line type="monotone" dataKey="error" stroke="#3b82f6" strokeWidth={3} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
             </div>
             <div className="mt-4 grid grid-cols-2 gap-4 text-center border-t border-slate-800 pt-4">
                <div><p className="text-[9px] text-slate-500 uppercase">{t('knn.low_k')}</p><p className="text-[10px] text-red-400 font-bold">{t('knn.overfitting')}</p></div>
                <div><p className="text-[9px] text-slate-500 uppercase">{t('knn.high_k')}</p><p className="text-[10px] text-orange-400 font-bold">{t('knn.underfitting')}</p></div>
             </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-[2rem] p-8 shadow-sm">
            <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-widest mb-6">{t('knn.voting_panel')}</h4>
            <div className="space-y-6">
               <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-600">{t('knn.k_value')}</span>
                  <div className="flex items-center gap-4">
                     <input type="range" min="1" max="15" value={k} onChange={(e) => setK(parseInt(e.target.value))} className="w-24 accent-blue-600 cursor-ew-resize" />
                     <span className="text-2xl font-black text-blue-600 font-mono">{k}</span>
                  </div>
               </div>
               <div className="grid grid-cols-2 gap-4">
                  <div className={`p-4 rounded-2xl border-2 transition ${processedData.prediction === 'M' ? 'bg-blue-50 border-blue-500' : 'bg-slate-50 border-transparent opacity-50'}`}>
                     <p className="text-[9px] font-black uppercase text-blue-600 mb-1">{t('knn.m_vote')}</p>
                     <p className="text-3xl font-black text-slate-800">{processedData.mCount}</p>
                  </div>
                  <div className={`p-4 rounded-2xl border-2 transition ${processedData.prediction === 'L' ? 'bg-red-50 border-red-500' : 'bg-slate-50 border-transparent opacity-50'}`}>
                     <p className="text-[9px] font-black uppercase text-red-600 mb-1">{t('knn.l_vote')}</p>
                     <p className="text-3xl font-black text-slate-800">{processedData.lCount}</p>
                  </div>
               </div>
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
