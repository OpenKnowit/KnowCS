import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Pause, Play, RotateCcw, SkipForward, Target } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { PERCEPTRON_DATA } from '../data/constants'
import { countErrors, trainPerceptron } from '../lib/perceptron'

const VIEW = 440
const PAD = 30
const POS = '#3b82f6'
const NEG = '#ef4444'

export const PerceptronModule = () => {
  const { t } = useTranslation()
  const [lr, setLr] = useState(0.1)
  const [stepIdx, setStepIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const steps = useMemo(() => trainPerceptron(PERCEPTRON_DATA, lr, 30), [lr])
  const safeIdx = Math.min(stepIdx, steps.length - 1)
  const step = steps[safeIdx]

  useEffect(() => {
    if (!playing || safeIdx >= steps.length - 1) return
    const timer = setTimeout(() => {
      const next = Math.min(safeIdx + 1, steps.length - 1)
      setStepIdx(next)
      if (next >= steps.length - 1) setPlaying(false)
    }, 700)
    return () => clearTimeout(timer)
  }, [playing, safeIdx, steps.length])

  const bounds = useMemo(() => {
    const xs = PERCEPTRON_DATA.map((p) => p.x)
    const ys = PERCEPTRON_DATA.map((p) => p.y)
    const pad = 1
    return { minX: Math.min(...xs) - pad, maxX: Math.max(...xs) + pad, minY: Math.min(...ys) - pad, maxY: Math.max(...ys) + pad }
  }, [])
  const sx = (x: number) => PAD + ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * (VIEW - 2 * PAD)
  const sy = (y: number) => VIEW - PAD - ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * (VIEW - 2 * PAD)

  // 决策边界 w0*x + w1*y + b = 0
  const boundary = useMemo(() => {
    const [w0, w1] = step.w
    const b = step.b
    if (Math.abs(w1) < 1e-9 && Math.abs(w0) < 1e-9) return null
    if (Math.abs(w1) >= 1e-9) {
      const yAt = (x: number) => -(w0 * x + b) / w1
      return { x1: bounds.minX, y1: yAt(bounds.minX), x2: bounds.maxX, y2: yAt(bounds.maxX) }
    }
    const xAt = -b / w0
    return { x1: xAt, y1: bounds.minY, x2: xAt, y2: bounds.maxY }
  }, [step, bounds])

  const totalErrors = useMemo(() => countErrors(PERCEPTRON_DATA, step.w, step.b), [step])
  const current = PERCEPTRON_DATA[step.pointIndex]
  const reset = () => { setStepIdx(0); setPlaying(false) }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-[2rem] p-8 shadow-sm">
          <h4 className="text-[11px] font-black text-slate-400 uppercase mb-6 flex items-center gap-2">
            <Target size={14} className="text-blue-500" /> {t('perceptron.canvas_title')}
          </h4>
          <div className="relative aspect-square w-full max-w-[480px] mx-auto">
            <svg viewBox={`0 0 ${VIEW} ${VIEW}`} className="w-full h-full border-l border-b border-slate-200">
              {[...Array(5)].map((_, i) => (
                <line key={`g${i}`} x1={PAD} y1={PAD + (i * (VIEW - 2 * PAD)) / 4} x2={VIEW - PAD} y2={PAD + (i * (VIEW - 2 * PAD)) / 4} stroke="#f1f5f9" />
              ))}
              {/* 决策边界 */}
              {boundary && (
                <motion.line
                  x1={sx(boundary.x1)} y1={sy(boundary.y1)} x2={sx(boundary.x2)} y2={sy(boundary.y2)}
                  stroke="#6366f1" strokeWidth={3} strokeDasharray="6 4"
                  animate={{ x1: sx(boundary.x1), y1: sy(boundary.y1), x2: sx(boundary.x2), y2: sy(boundary.y2) }}
                />
              )}
              {/* 数据点 */}
              {PERCEPTRON_DATA.map((p, i) => {
                const isCurrent = i === step.pointIndex
                return (
                  <g key={i}>
                    {isCurrent && <circle cx={sx(p.x)} cy={sy(p.y)} r={13} fill="none" stroke={step.correct ? '#10b981' : '#f59e0b'} strokeWidth={3} className={step.correct ? '' : 'animate-pulse'} />}
                    <circle cx={sx(p.x)} cy={sy(p.y)} r={7} fill={p.label === 1 ? POS : NEG} stroke="#fff" strokeWidth={2} />
                  </g>
                )
              })}
            </svg>
          </div>
          <div className="flex items-center justify-center gap-4 mt-6">
            <button onClick={reset} className="p-2.5 bg-slate-100 hover:bg-slate-200 rounded-xl" title={t('perceptron.reset')}><RotateCcw size={16} /></button>
            <button onClick={() => { if (safeIdx >= steps.length - 1) setStepIdx(0); setPlaying((p) => !p) }} className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl flex items-center gap-2 font-bold text-sm shadow-lg shadow-blue-200">
              {playing ? <Pause size={16} /> : <Play size={16} />} {playing ? t('perceptron.pause') : t('perceptron.play')}
            </button>
            <button onClick={() => { setStepIdx((i) => Math.min(i + 1, steps.length - 1)); setPlaying(false) }} className="p-2.5 bg-slate-100 hover:bg-slate-200 rounded-xl" title={t('perceptron.step')}><SkipForward size={16} /></button>
          </div>
        </div>

        <div className="lg:col-span-5 space-y-6">
          <div className="bg-slate-900 rounded-[2rem] p-6 text-white shadow-2xl space-y-4">
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{t('perceptron.weights')}</h4>
            <div className="font-mono text-sm space-y-2 bg-slate-950/50 p-4 rounded-xl border border-slate-800">
              <div className="flex justify-between"><span className="text-slate-400">w₁</span><span className="text-emerald-400 font-bold">{step.w[0].toFixed(3)}</span></div>
              <div className="flex justify-between"><span className="text-slate-400">w₂</span><span className="text-emerald-400 font-bold">{step.w[1].toFixed(3)}</span></div>
              <div className="flex justify-between"><span className="text-slate-400">b (bias)</span><span className="text-amber-400 font-bold">{step.b.toFixed(3)}</span></div>
            </div>
            <div className="grid grid-cols-2 gap-3 text-center pt-2">
              <div><p className="text-[9px] text-slate-500 uppercase">{t('perceptron.misclassified')}</p><p className={`text-2xl font-black font-mono ${totalErrors === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{totalErrors}</p></div>
              <div><p className="text-[9px] text-slate-500 uppercase">{t('perceptron.step_count')}</p><p className="text-2xl font-black text-blue-400 font-mono">{safeIdx + 1}/{steps.length}</p></div>
            </div>
            {totalErrors === 0 && <div className="text-center text-[11px] font-bold text-emerald-400 bg-emerald-950/40 rounded-lg py-2">{t('perceptron.separated')}</div>}
          </div>

          <div className="bg-white border border-slate-200 rounded-[2rem] p-6 shadow-sm space-y-4">
            <h4 className="text-[11px] font-black text-slate-400 uppercase tracking-widest">{t('perceptron.current_step')}</h4>
            <div className={`p-3 rounded-xl text-xs font-medium leading-relaxed ${step.correct ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-800'}`}>
              {step.correct
                ? t('perceptron.step_correct', { idx: step.pointIndex + 1, target: step.target })
                : t('perceptron.step_wrong', { idx: step.pointIndex + 1, pred: step.predicted, target: step.target })}
            </div>
            <div className="text-[11px] text-slate-500 font-mono bg-slate-50 p-3 rounded-xl border border-slate-100">
              <Latex formula={`\\hat{y} = \\text{sign}(${step.w[0].toFixed(2)} \\cdot ${current.x} + ${step.w[1].toFixed(2)} \\cdot ${current.y} + ${step.b.toFixed(2)})`} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-600">{t('perceptron.lr')} <Latex formula="\eta" /></span>
              <div className="flex items-center gap-3">
                <input type="range" min={0.05} max={1} step={0.05} value={lr} onChange={(e) => { setLr(parseFloat(e.target.value)); reset() }} className="w-24 accent-blue-600" />
                <span className="text-base font-black text-blue-600 font-mono">{lr.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <SeniorAdvice content={<Trans i18nKey="perceptron.advice" components={{
        1: <Latex formula="\delta = y - \hat{y}" />,
        3: <strong className="font-bold text-amber-700" />,
        5: <Latex formula="[w, b]" />,
      }} />} />
    </div>
  )
}
