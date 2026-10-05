import { useMemo, useState } from 'react'
import { m } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { BAYES_DATA } from '../data/constants'
import { computeNaiveBayes } from '../lib/bayes'
import type { BayesClass, BayesFeature, NaiveBayesResults } from '../types'

const CLASSES: BayesClass[] = ['yes', 'no']
const ALPHA_MAX = 2

/** 分数显示：-∞ 如实显示；极小的连乘积用科学计数法 */
const fmtScore = (v: number, useLog: boolean): string => {
  if (v === -Infinity) return '−∞'
  if (useLog) return v.toFixed(3)
  return v !== 0 && v < 1e-3 ? v.toExponential(3) : v.toFixed(5)
}

// Naive Bayes Module (Disease Z Case)
export const NaiveBayesModule = () => {
  const [inputs, setInputs] = useState<Record<BayesFeature, string>>({
    BP: 'High',
    Fever: 'No',
    Diabetes: 'Yes',
    Vomit: 'Yes',
  })
  const [alpha, setAlpha] = useState(0)
  const [useLog, setUseLog] = useState(false)
  const { t } = useTranslation()

  const results = useMemo<NaiveBayesResults>(
    () => computeNaiveBayes(BAYES_DATA, inputs, alpha, useLog),
    [inputs, alpha, useLog]
  )
  const hasZero = CLASSES.some((cls) => results[cls].raw === 0)

  const onAlphaChange = (raw: string) => {
    const v = parseFloat(raw)
    setAlpha(Number.isFinite(v) ? Math.min(ALPHA_MAX, Math.max(0, v)) : 0)
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
          <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2 text-sm uppercase tracking-wider">{t('bayes.naive.input_title')}</h4>
          <div className="space-y-4">
            {(Object.keys(inputs) as BayesFeature[]).map((feat) => (
              <div key={feat}>
                <label htmlFor={`nb-${feat}`} className="text-[10px] font-black text-slate-400 uppercase block mb-1">{feat}</label>
                <select
                  id={`nb-${feat}`}
                  value={inputs[feat]}
                  onChange={(e) => setInputs({ ...inputs, [feat]: e.target.value })}
                  className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-xs font-bold focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
                >
                  {Object.keys(BAYES_DATA.counts[feat]).map((opt) => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </div>
            ))}
            <div className="pt-4 border-t space-y-4">
              <div className="flex items-center justify-between">
                <label htmlFor="nb-alpha" className="text-[10px] font-bold text-slate-500 uppercase">
                  <Trans i18nKey="bayes.naive.smoothing" components={{ 1: <Latex formula="\alpha" /> }} />
                </label>
                <input
                  id="nb-alpha"
                  type="number" step="0.1" min="0" max={ALPHA_MAX} value={alpha}
                  onChange={(e) => onAlphaChange(e.target.value)}
                  className="w-16 p-1 border rounded text-center text-xs font-mono"
                />
              </div>
              <button
                onClick={() => setUseLog(!useLog)}
                aria-pressed={useLog}
                className={`w-full py-2 rounded-lg text-[10px] font-black transition-all ${useLog ? 'bg-indigo-600 text-white shadow-lg' : 'bg-slate-100 text-slate-600'}`}
              >
                {useLog ? t('bayes.naive.log_mode') : t('bayes.naive.product_mode')}
              </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900 rounded-2xl p-6 text-white relative overflow-hidden">
            <h4 className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-6">{t('bayes.naive.result_title')}</h4>
            <div className="space-y-6 relative z-10">
              {CLASSES.map((cls) => (
                <div key={cls}>
                  <div className="flex justify-between items-end mb-2 gap-4">
                    <span className="text-xs font-bold uppercase">Class: {cls}</span>
                    <span className="text-xl font-mono text-blue-400">{(results[cls].prob * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                    <m.div initial={{ width: 0 }} animate={{ width: `${results[cls].prob * 100}%` }} className={`h-full ${cls === 'yes' ? 'bg-blue-500' : 'bg-slate-500'}`} />
                  </div>
                  <p className="mt-2 text-[10px] font-mono text-slate-400 flex flex-wrap items-center gap-x-2">
                    <span className="text-slate-500">{useLog ? t('bayes.naive.log_score') : t('bayes.naive.product_score')}</span>
                    <span className={results[cls].score === -Infinity ? 'text-rose-400 font-bold' : 'text-slate-200'}>
                      = {fmtScore(results[cls].score, useLog)}
                    </span>
                  </p>
                </div>
              ))}
            </div>
            {hasZero && (
              <p className="mt-5 flex items-start gap-2 text-[11px] text-amber-200 bg-amber-500/10 border border-amber-500/30 rounded-xl p-3" role="status">
                <AlertTriangle size={14} className="mt-0.5 shrink-0" aria-hidden />
                <span><Trans i18nKey="bayes.naive.zero_warning" components={{ 1: <Latex formula="\alpha" /> }} /></span>
              </p>
            )}
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl p-6">
            <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2 text-sm uppercase">{t('bayes.naive.likelihood_chain')}</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[10px]">
              {CLASSES.map((cls) => (
                <div key={cls} className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="font-bold text-slate-400 mb-2 uppercase tracking-tighter">{t('bayes.naive.prob_for', { cls })}</p>
                  <div className="space-y-2">
                    {results[cls].steps.map((s, i) => {
                      const shown = useLog ? Math.log(s.val) : s.val
                      return (
                        <div key={i} className="flex justify-between items-center gap-2">
                          <span className="text-slate-500 font-mono"><Latex formula={useLog ? `\\log ${s.label}` : s.label} /></span>
                          <span className={`font-mono font-bold ${s.val === 0 ? 'text-rose-600' : 'text-slate-800'}`}>
                            {shown === -Infinity ? '−∞' : shown.toFixed(3)}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <SeniorAdvice content={<Trans i18nKey="bayes.naive.advice" components={{
        1: <strong className="font-bold text-amber-700" />,
        3: <strong className="font-bold text-amber-700" />,
        5: <Latex formula="\alpha" />,
      }} />} />
    </div>
  )
}
