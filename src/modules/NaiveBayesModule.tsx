import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { BAYES_DATA } from '../data/constants'
import type { BayesClass, BayesFeature, NaiveBayesResults } from '../types'

const CLASSES: BayesClass[] = ['yes', 'no']

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

  const results = useMemo<NaiveBayesResults>(() => {
    const results: NaiveBayesResults = {
      yes: { score: 0, raw: 1, steps: [], prob: 0 },
      no: { score: 0, raw: 1, steps: [], prob: 0 },
    }
    CLASSES.forEach((cls) => {
      const prior = BAYES_DATA.priors[cls]
      results[cls].steps.push({ name: 'Prior', val: prior, label: cls === 'yes' ? 'P(Z=Yes)' : 'P(Z=No)' })
      let likelihoodProduct = 1
      let logSum = Math.log(prior)
      ;(Object.entries(inputs) as [BayesFeature, string][]).forEach(([feat, val]) => {
        const count = BAYES_DATA.counts[feat][val][cls]
        const totalCls = cls === 'yes' ? 9 : 5
        const m = BAYES_DATA.m_values[feat]
        const prob = (count + alpha) / (totalCls + m * alpha)
        results[cls].steps.push({
          name: feat, val: prob, label: `P(${feat}|${cls})`,
          formula: `\\frac{${count} + ${alpha}}{${totalCls} + ${m} \\times ${alpha}}`,
        })
        likelihoodProduct *= prob
        logSum += Math.log(prob || 1e-10)
      })
      results[cls].raw = prior * likelihoodProduct
      results[cls].score = useLog ? logSum : results[cls].raw
    })
    const totalRaw = results.yes.raw + results.no.raw
    results.yes.prob = totalRaw > 0 ? results.yes.raw / totalRaw : 0
    results.no.prob = totalRaw > 0 ? results.no.raw / totalRaw : 0
    return results
  }, [inputs, alpha, useLog])

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
          <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2 text-sm uppercase tracking-wider">{t('bayes.naive.input_title')}</h4>
          <div className="space-y-4">
            {(Object.keys(inputs) as BayesFeature[]).map((feat) => (
              <div key={feat}>
                <label className="text-[10px] font-black text-slate-400 uppercase block mb-1">{feat}</label>
                <select
                  value={inputs[feat]}
                  onChange={(e) => setInputs({ ...inputs, [feat]: e.target.value })}
                  className="w-full p-2 bg-slate-50 border border-slate-200 rounded-lg text-xs font-bold outline-none"
                >
                  {Object.keys(BAYES_DATA.counts[feat]).map((opt) => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </div>
            ))}
            <div className="pt-4 border-t space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold text-slate-500 uppercase"><Trans i18nKey="bayes.naive.smoothing">Smoothing <Latex formula="\alpha" />:</Trans></span>
                <input
                  type="number" step="0.1" min="0" max="2" value={alpha}
                  onChange={(e) => setAlpha(parseFloat(e.target.value) || 0)}
                  className="w-16 p-1 border rounded text-center text-xs font-mono"
                />
              </div>
              <button
                onClick={() => setUseLog(!useLog)}
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
                  <div className="flex justify-between items-end mb-2">
                    <span className="text-xs font-bold uppercase">Class: {cls}</span>
                    <span className="text-xl font-mono text-blue-400">{(results[cls].prob * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div initial={{ width: 0 }} animate={{ width: `${results[cls].prob * 100}%` }} className={`h-full ${cls === 'yes' ? 'bg-blue-500' : 'bg-slate-500'}`} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-2xl p-6">
            <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2 text-sm uppercase">{t('bayes.naive.likelihood_chain')}</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[10px]">
              {CLASSES.map((cls) => (
                <div key={cls} className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="font-bold text-slate-400 mb-2 uppercase tracking-tighter">{t('bayes.naive.prob_for', { cls })}</p>
                  <div className="space-y-2">
                    {results[cls].steps.map((s, i) => (
                      <div key={i} className="flex justify-between items-center">
                        <span className="text-slate-500 font-mono"><Latex formula={s.label} /></span>
                        <span className="font-mono font-bold text-slate-800">{s.val.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      <SeniorAdvice content={<Trans i18nKey="bayes.naive.advice">
        讲义第 23 页重点... <Latex formula="\alpha" /> ...
      </Trans>} />
    </div>
  )
}
