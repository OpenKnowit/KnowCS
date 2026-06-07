import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { LineChart as LineIcon, Pause, Play, RotateCcw, SkipForward } from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { KMEANS_DATA, KMEANS_INIT_CENTROIDS } from '../data/constants'
import { elbowCurve, farthestFirstInit, runKMeans, standardize } from '../lib/kmeans'
import type { Vec2 } from '../types'

const CLUSTER_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
const VIEW = 440
const PAD = 30

export const KMeansModule = () => {
  const { t } = useTranslation()
  const [k, setK] = useState(3)
  const [iter, setIter] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [zscore, setZscore] = useState(false)

  // 原始或标准化后的数据
  const data = useMemo<Vec2[]>(() => (zscore ? standardize(KMEANS_DATA).data : KMEANS_DATA), [zscore])

  // 初始质心：K=3 用教学固定值，其它用最远优先
  const initCentroids = useMemo<Vec2[]>(() => {
    if (!zscore && k === 3) return KMEANS_INIT_CENTROIDS
    return farthestFirstInit(data, k)
  }, [data, k, zscore])

  const history = useMemo(() => runKMeans(data, initCentroids), [data, initCentroids])
  const safeIter = Math.min(iter, history.length - 1)
  const frame = history[safeIter]

  useEffect(() => {
    if (!playing || safeIter >= history.length - 1) return
    const timer = setTimeout(() => {
      const next = Math.min(safeIter + 1, history.length - 1)
      setIter(next)
      if (next >= history.length - 1) setPlaying(false)
    }, 900)
    return () => clearTimeout(timer)
  }, [playing, safeIter, history.length])

  // 坐标范围 → 像素
  const bounds = useMemo(() => {
    const xs = data.map((p) => p.x)
    const ys = data.map((p) => p.y)
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) }
  }, [data])
  const sx = (x: number) => PAD + ((x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (VIEW - 2 * PAD)
  const sy = (y: number) => VIEW - PAD - ((y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (VIEW - 2 * PAD)

  const elbow = useMemo(() => elbowCurve(data, [1, 2, 3, 4, 5, 6]), [data])

  const reset = () => { setIter(0); setPlaying(false) }
  const changeK = (nk: number) => { setK(nk); reset() }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* 散点画布 */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-[2rem] p-8 shadow-sm">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-3">
            <h4 className="text-[11px] font-black text-slate-400 uppercase">{t('kmeans.canvas_title')}</h4>
            <div className="flex gap-2 bg-slate-100 p-1 rounded-xl">
              <button onClick={() => { setZscore(false); reset() }} className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${!zscore ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`}>{t('kmeans.raw')}</button>
              <button onClick={() => { setZscore(true); reset() }} className={`px-3 py-1.5 rounded-lg text-[10px] font-black uppercase transition ${zscore ? 'bg-white shadow text-slate-800' : 'text-slate-400'}`}>{t('kmeans.zscore')}</button>
            </div>
          </div>
          <div className="relative aspect-square w-full max-w-[480px] mx-auto">
            <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="w-full h-full border-l border-b border-slate-200">
              {[...Array(5)].map((_, i) => (
                <line key={`h${i}`} x1={PAD} y1={PAD + (i * (VIEW - 2 * PAD)) / 4} x2={VIEW - PAD} y2={PAD + (i * (VIEW - 2 * PAD)) / 4} stroke="#f1f5f9" />
              ))}
              {/* 数据点 */}
              {data.map((p, i) => (
                <motion.circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={6} fill={CLUSTER_COLORS[frame.assignments[i]]} animate={{ fill: CLUSTER_COLORS[frame.assignments[i]] }} opacity={0.75} />
              ))}
              {/* 质心 */}
              {frame.centroids.map((c, i) => (
                <motion.g key={i} animate={{ x: sx(c.x), y: sy(c.y) }} initial={false}>
                  <path d="M -9 -9 L 9 9 M -9 9 L 9 -9" stroke={CLUSTER_COLORS[i]} strokeWidth={4} strokeLinecap="round" />
                  <circle r={13} fill="none" stroke={CLUSTER_COLORS[i]} strokeWidth={2.5} />
                </motion.g>
              ))}
            </svg>
          </div>
          <div className="flex items-center justify-center gap-4 mt-6">
            <button onClick={reset} className="p-2.5 bg-slate-100 hover:bg-slate-200 rounded-xl" title={t('kmeans.reset')}><RotateCcw size={16} /></button>
            <button onClick={() => { if (safeIter >= history.length - 1) setIter(0); setPlaying((p) => !p) }} className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl flex items-center gap-2 font-bold text-sm shadow-lg shadow-blue-200">
              {playing ? <Pause size={16} /> : <Play size={16} />} {playing ? t('kmeans.pause') : t('kmeans.play')}
            </button>
            <button onClick={() => { setIter((i) => Math.min(i + 1, history.length - 1)); setPlaying(false) }} className="p-2.5 bg-slate-100 hover:bg-slate-200 rounded-xl" title={t('kmeans.step')}><SkipForward size={16} /></button>
          </div>
        </div>

        {/* 控制 + Elbow */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white border border-slate-200 rounded-[2rem] p-6 shadow-sm space-y-5">
            <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-widest">{t('kmeans.control')}</h4>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-600">{t('kmeans.k_clusters')}</span>
              <div className="flex gap-1.5">
                {[2, 3, 4, 5].map((nk) => (
                  <button key={nk} onClick={() => changeK(nk)} className={`w-9 h-9 rounded-lg text-sm font-black transition ${k === nk ? 'bg-blue-600 text-white shadow' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'}`}>{nk}</button>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 border-t border-slate-100 pt-4">
              <div className="text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">{t('kmeans.iteration')}</p>
                <p className="text-2xl font-black text-blue-600 font-mono">{safeIter} <span className="text-sm text-slate-300">/ {history.length - 1}</span></p>
              </div>
              <div className="text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">{t('kmeans.inertia')}</p>
                <p className="text-2xl font-black text-slate-700 font-mono">{frame.inertia.toFixed(1)}</p>
              </div>
            </div>
            <div className="text-[10px] text-slate-400 leading-relaxed">
              {frame.moved === Infinity ? t('kmeans.init_state') : frame.moved < 1e-6 ? t('kmeans.converged') : t('kmeans.moved', { d: frame.moved.toFixed(3) })}
            </div>
          </div>

          <div className="bg-slate-900 rounded-[2rem] p-6 text-white shadow-2xl">
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
              <LineIcon size={14} className="text-blue-500" /> {t('kmeans.elbow_title')}
            </h4>
            <div className="h-[180px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={elbow}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="k" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={9} tickLine={false} axisLine={false} width={28} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '10px' }} formatter={(value) => (typeof value === 'number' ? value.toFixed(1) : String(value))} />
                  <ReferenceLine x={k} stroke="#fbbf24" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="inertia" stroke="#3b82f6" strokeWidth={3} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-slate-400 mt-3 leading-relaxed">{t('kmeans.elbow_hint')}</p>
          </div>
        </div>
      </div>
      <SeniorAdvice content={<Trans i18nKey="kmeans.advice" components={{
        1: <Latex formula="K" />,
        3: <strong className="font-bold text-amber-700" />,
      }} />} />
    </div>
  )
}
