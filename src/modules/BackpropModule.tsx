import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, m } from 'framer-motion'
import { Activity, Play, RotateCcw, Search, StepForward, TrendingDown } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { LineChart } from '../components/LineChart'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { applyUpdate, traceNeuron } from '../lib/backprop'
import type { NeuronParams } from '../lib/backprop'

const DEFAULTS: NeuronParams = { oi: 1, oj: 0.5, wik: 0.4, wjk: -0.6, target: 1, eta: 0.5 }
const HISTORY_MAX = 30

// 节点坐标（SVG viewBox 400×240）
const NODE = { i: { x: 70, y: 65 }, j: { x: 70, y: 175 }, k: { x: 320, y: 120 } }

const f = (v: number, d = 3) => v.toFixed(d)
/** 代入公式时负数加括号：0.5 × (−0.6) */
const p = (v: number, d = 3) => (v < 0 ? `(${f(v, d)})` : f(v, d))

interface SliderSpec {
  key: keyof NeuronParams
  label: string // LaTeX
  min: number
  max: number
  step: number
}

const SLIDERS: SliderSpec[] = [
  { key: 'oi', label: 'O_i', min: 0, max: 1, step: 0.05 },
  { key: 'oj', label: 'O_j', min: 0, max: 1, step: 0.05 },
  { key: 'wik', label: 'w_{ik}', min: -2, max: 2, step: 0.1 },
  { key: 'wjk', label: 'w_{jk}', min: -2, max: 2, step: 0.1 },
  { key: 'target', label: 'T_k', min: 0, max: 1, step: 0.05 },
  { key: 'eta', label: '\\eta', min: 0.1, max: 2, step: 0.1 },
]

/** 边的颜色 / 粗细编码权重的符号与大小 */
const edgeStyle = (w: number) => ({
  stroke: w >= 0 ? '#60a5fa' : '#f472b6',
  strokeWidth: 1.5 + Math.min(2, Math.abs(w)) * 2,
})

// Backprop Module：单输出神经元的前向 → 链式法则 → δ → Δw → 更新，全部数值实时可见
export const BackpropModule = () => {
  const { t } = useTranslation()
  const [params, setParams] = useState<NeuronParams>(DEFAULTS)
  const [phase, setPhase] = useState<'idle' | 'animating' | 'revealed'>('idle')
  const [history, setHistory] = useState<number[]>(() => [traceNeuron(DEFAULTS).error])

  const tr = useMemo(() => traceNeuron(params), [params])
  const revealed = phase === 'revealed'

  // 反向动画结束后揭示 Δw（定时器在 effect 中注册，卸载时清理）
  useEffect(() => {
    if (phase !== 'animating') return
    const timer = setTimeout(() => setPhase('revealed'), 1400)
    return () => clearTimeout(timer)
  }, [phase])

  const setParam = (key: keyof NeuronParams, value: number) => {
    const next = { ...params, [key]: value }
    setParams(next)
    // 手动改参数相当于开始一段新的训练，误差曲线从当前点重新记录
    setHistory([traceNeuron(next).error])
  }

  const stepUpdate = () => {
    const next = applyUpdate(params)
    setParams(next)
    setHistory((h) => [...h, traceNeuron(next).error].slice(-HISTORY_MAX))
  }

  const reset = () => {
    setParams(DEFAULTS)
    setHistory([traceNeuron(DEFAULTS).error])
    setPhase('idle')
  }

  const historyData = history.map((e, i) => ({ x: i, y: e }))

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-6 min-w-0">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity size={18} className="text-red-500" aria-hidden />
            {t('backprop_module.title')}
          </h3>

          {/* 网络图 */}
          <div className="bg-slate-900 rounded-2xl p-3 relative overflow-hidden">
            <svg
              viewBox="0 0 400 240"
              className="w-full h-auto"
              role="img"
              aria-label={t('backprop_module.diagram_label', {
                wik: f(params.wik, 2), wjk: f(params.wjk, 2), out: f(tr.out), target: f(params.target, 2),
              })}
            >
              {(['i', 'j'] as const).map((n) => {
                const w = n === 'i' ? params.wik : params.wjk
                const dw = n === 'i' ? tr.dwik : tr.dwjk
                const from = NODE[n]
                const mx = (from.x + NODE.k.x) / 2
                const my = (from.y + NODE.k.y) / 2 + (n === 'i' ? -14 : 22)
                const dwY = n === 'i' ? my - 15 : my + 15 // Δw 放在远离连线的一侧
                return (
                  <g key={n}>
                    <line x1={from.x} y1={from.y} x2={NODE.k.x} y2={NODE.k.y} {...edgeStyle(w)} strokeLinecap="round" opacity={0.7} />
                    <text x={mx} y={my} textAnchor="middle" fontSize={12} fontFamily="monospace" fill="#cbd5e1">
                      w<tspan fontSize={9} dy={3}>{n}k</tspan><tspan dy={-3}> = {f(w, 2)}</tspan>
                    </text>
                    {revealed && (
                      <m.text
                        x={mx} y={dwY} textAnchor="middle" fontSize={11} fontFamily="monospace" fontWeight={700}
                        fill="#f87171" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      >
                        Δw = {dw >= 0 ? '+' : ''}{f(dw, 3)}
                      </m.text>
                    )}
                  </g>
                )
              })}

              {/* 误差信号反向流动的粒子 */}
              <AnimatePresence>
                {phase === 'animating' && (['i', 'j'] as const).map((n) => (
                  <m.circle
                    key={n}
                    r={6}
                    fill="#ef4444"
                    initial={{ cx: NODE.k.x, cy: NODE.k.y, opacity: 1 }}
                    animate={{ cx: NODE[n].x, cy: NODE[n].y, opacity: [1, 1, 0.4] }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 1.2, ease: 'easeInOut' }}
                  />
                ))}
              </AnimatePresence>

              {/* 输入神经元 */}
              {(['i', 'j'] as const).map((n) => (
                <g key={n}>
                  <circle cx={NODE[n].x} cy={NODE[n].y} r={20} fill="#3b82f6" />
                  <text x={NODE[n].x} y={NODE[n].y + 5} textAnchor="middle" fontSize={14} fontFamily="monospace" fill="#fff">{n}</text>
                  <text x={NODE[n].x} y={NODE[n].y + 36} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#93c5fd">
                    O = {f(n === 'i' ? params.oi : params.oj, 2)}
                  </text>
                </g>
              ))}

              {/* 输出神经元 */}
              <circle cx={NODE.k.x} cy={NODE.k.y} r={26} fill="#ef4444" stroke="#fecaca" strokeWidth={3} />
              <text x={NODE.k.x} y={NODE.k.y + 5} textAnchor="middle" fontSize={15} fontFamily="monospace" fill="#fff">k</text>
              <text x={NODE.k.x} y={NODE.k.y + 44} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#fca5a5">O = {f(tr.out)}</text>
              <text x={NODE.k.x} y={NODE.k.y + 58} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#94a3b8">T = {f(params.target, 2)}</text>
              <text x={NODE.k.x} y={NODE.k.y - 36} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#fbbf24">δ = {f(tr.delta, 4)}</text>
              <text x={392} y={18} textAnchor="end" fontSize={10} fill="#64748b" fontWeight={700}>
                {t('backprop_module.output_layer').toUpperCase()} · E = {f(tr.error, 4)}
              </text>
            </svg>
          </div>

          {/* 参数 + 操作 */}
          <div className="bg-white border rounded-xl p-4 space-y-4 shadow-sm">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3">
              {SLIDERS.map((s) => (
                <div key={s.key} className="flex items-center gap-3">
                  <label htmlFor={`bp-${s.key}`} className="w-10 shrink-0 text-sm text-slate-700"><Latex formula={s.label} /></label>
                  <input
                    id={`bp-${s.key}`}
                    type="range" min={s.min} max={s.max} step={s.step} value={params[s.key]}
                    onChange={(e) => setParam(s.key, parseFloat(e.target.value))}
                    className="flex-1 min-w-0 accent-red-600 cursor-pointer"
                  />
                  <span className="w-12 text-right font-mono text-xs font-bold text-red-600">{f(params[s.key], 2)}</span>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setPhase('animating')}
                disabled={phase === 'animating'}
                className="px-2 py-2.5 bg-red-600 text-white rounded-lg flex items-center justify-center gap-1.5 font-bold text-xs whitespace-nowrap hover:bg-red-700 transition disabled:opacity-50 shadow-md"
              >
                <Play size={15} fill="currentColor" aria-hidden />
                {phase === 'animating' ? t('backprop_module.calculating') : t('backprop_module.track_grad')}
              </button>
              <button
                onClick={stepUpdate}
                className="px-2 py-2.5 bg-emerald-600 text-white rounded-lg flex items-center justify-center gap-1.5 font-bold text-xs whitespace-nowrap hover:bg-emerald-700 transition shadow-md"
              >
                <StepForward size={15} aria-hidden /> {t('backprop_module.apply_update')}
              </button>
              <button
                onClick={reset}
                className="px-2 py-2.5 bg-slate-100 text-slate-600 rounded-lg flex items-center justify-center gap-1.5 font-bold text-xs whitespace-nowrap hover:bg-slate-200 transition"
              >
                <RotateCcw size={15} aria-hidden /> {t('backprop_module.reset')}
              </button>
            </div>
          </div>

          {/* 误差下降曲线 */}
          <div className="bg-slate-900 rounded-2xl p-5 text-white">
            <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3 flex items-center gap-2">
              <TrendingDown size={14} className="text-emerald-400" aria-hidden /> {t('backprop_module.history_title')}
            </h4>
            <LineChart
              data={historyData}
              height={150}
              formatY={(v) => v.toFixed(3)}
              xLabel={t('backprop_module.update_axis')}
              ariaLabel={t('backprop_module.history_label', { n: history.length - 1, e: f(tr.error, 4) })}
            />
            <p className="text-[10px] text-slate-400 mt-2">{t('backprop_module.history_hint')}</p>
          </div>
        </div>

        {/* 推导笔记本：公式代入当前数值 */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 sm:p-6 flex flex-col shadow-sm min-w-0">
          <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2 italic border-b pb-2">
            <Search size={16} className="text-blue-500" aria-hidden />
            {t('backprop_module.notebook_title')}
          </h4>
          <div className="flex-1 space-y-4 overflow-x-auto">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 overflow-x-auto">
              <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.forward_title')}</p>
              <Latex displayMode formula={`\\begin{aligned} net_k &= w_{ik}O_i + w_{jk}O_j \\\\ &= ${p(params.wik)} \\times ${f(params.oi, 2)} + ${p(params.wjk)} \\times ${f(params.oj, 2)} \\\\ &= ${f(tr.net)} \\\\[6pt] O_k &= \\sigma(net_k) = \\frac{1}{1 + e^{-net_k}} \\\\ &= ${f(tr.out)} \\\\[6pt] E &= \\tfrac{1}{2}(T_k - O_k)^2 \\\\ &= \\tfrac{1}{2}(${f(params.target, 2)} - ${f(tr.out)})^2 = ${f(tr.error, 4)} \\end{aligned}`} />
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 overflow-x-auto">
              <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.step1_title')}</p>
              <Latex displayMode formula="\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial O_k} \cdot \frac{\partial O_k}{\partial net_k} \cdot \frac{\partial net_k}{\partial w_{jk}}" />
            </div>
            <div className={`p-4 rounded-xl border overflow-x-auto transition-colors ${revealed ? 'bg-amber-50 border-amber-200' : 'bg-slate-50 border-slate-100'}`}>
              <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.step2_title')}</p>
              <p className="text-xs text-gray-600 mb-2">
                <Trans i18nKey="backprop_module.step2_desc" components={{ 1: <Latex formula="\delta_k = - \frac{\partial E}{\partial net_k}" /> }} />
              </p>
              <Latex displayMode formula="\delta_k = (T_k - O_k) \cdot f'(net_k)" className="text-red-700" />
              <p className="text-xs text-gray-600 mt-2">{t('backprop_module.sigmoid_note')}</p>
              <Latex displayMode formula={`\\begin{aligned} f'(net_k) &= O_k(1 - O_k) = ${f(tr.fprime)} \\\\ \\delta_k &= ${p(tr.diff)} \\times ${f(tr.fprime)} = ${f(tr.delta, 4)} \\end{aligned}`} className="text-red-700" />
            </div>
            <div className={`p-4 rounded-xl border overflow-x-auto transition-colors ${revealed ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-100'}`}>
              <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.step3_title')}</p>
              <Latex displayMode formula="\Delta w_{xk} = \eta \cdot \delta_k \cdot O_x" className="text-green-700" />
              <Latex displayMode formula={`\\begin{aligned} \\Delta w_{ik} &= ${f(params.eta, 2)} \\times ${p(tr.delta, 4)} \\times ${f(params.oi, 2)} \\\\ &= ${f(tr.dwik, 4)} \\\\[4pt] \\Delta w_{jk} &= ${f(params.eta, 2)} \\times ${p(tr.delta, 4)} \\times ${f(params.oj, 2)} \\\\ &= ${f(tr.dwjk, 4)} \\end{aligned}`} className="text-green-700" />
            </div>
            <SeniorAdvice content={
              <Trans i18nKey="backprop_module.advice" components={{
                1: <Latex formula="w_{jk}" />,
                3: <Latex formula="E" />,
                5: <Latex formula="\delta_k" />,
              }} />
            } />
          </div>
        </div>
      </div>
    </div>
  )
}
