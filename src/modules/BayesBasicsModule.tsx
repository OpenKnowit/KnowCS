import { useMemo, useState } from 'react'
import { Calculator, Flame, Wind } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { bayesPosterior } from '../lib/bayes'

// Bayes Basics Module (Formula & Fire Case)
export const BayesBasicsModule = () => {
  const [pFire, setPFire] = useState(0.01)
  const [pSmokeGivenFire, setPSmokeGivenFire] = useState(0.9)
  const [pSmoke, setPSmoke] = useState(0.1)
  const { t } = useTranslation()

  const pFireGivenSmoke = useMemo(
    () => bayesPosterior(pFire, pSmokeGivenFire, pSmoke),
    [pFire, pSmokeGivenFire, pSmoke]
  )

  return (
    <div className="space-y-10">
      {/* Formula Parsing Section */}
      <section>
        <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Calculator size={18} className="text-blue-500" />
          {t('bayes.basics.title')}
        </h3>
        <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6">
          <div className="flex flex-col items-center justify-center py-4 bg-white rounded-xl border border-slate-100 shadow-inner mb-6">
            <Latex displayMode formula="P(B|E) = P(B) \times \frac{P(E|B)}{P(E)}" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 border border-blue-100 rounded-xl">
              <p className="text-xs font-bold text-blue-600 mb-1 uppercase tracking-wider">{t('bayes.basics.terms.posterior')}</p>
              <p className="text-sm text-slate-700">
                <Trans i18nKey="bayes.basics.terms.posterior_desc">
                  Update of belief <Latex formula="B" /> after evidence <Latex formula="E" />.
                </Trans>
              </p>
            </div>
            <div className="p-4 bg-green-50 border border-green-100 rounded-xl">
              <p className="text-xs font-bold text-green-600 mb-1 uppercase tracking-wider">{t('bayes.basics.terms.prior')}</p>
              <p className="text-sm text-slate-700">
                <Trans i18nKey="bayes.basics.terms.prior_desc">
                  Probability of belief <Latex formula="B" /> before any evidence.
                </Trans>
              </p>
            </div>
            <div className="p-4 bg-purple-50 border border-purple-100 rounded-xl">
              <p className="text-xs font-bold text-purple-600 mb-1 uppercase tracking-wider">{t('bayes.basics.terms.likelihood')}</p>
              <p className="text-sm text-slate-700">
                <Trans i18nKey="bayes.basics.terms.likelihood_desc">
                  Probability of evidence <Latex formula="E" /> given belief <Latex formula="B" /> is true.
                </Trans>
              </p>
            </div>
            <div className="p-4 bg-orange-50 border border-orange-100 rounded-xl">
              <p className="text-xs font-bold text-orange-600 mb-1 uppercase tracking-wider">{t('bayes.basics.terms.marginal')}</p>
              <p className="text-sm text-slate-700">
                <Trans i18nKey="bayes.basics.terms.marginal_desc">
                  Total probability of evidence <Latex formula="E" /> under all conditions.
                </Trans>
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Fire Case Section */}
      <section>
        <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
          <Flame size={18} className="text-red-500" />
          {t('bayes.fire_case.title')}
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700"><Trans i18nKey="bayes.fire_case.p_fire">Prob <Latex formula="P(\text{Fire})" /></Trans></span>
                <input
                  type="range" min="0.001" max="0.2" step="0.001" value={pFire}
                  onChange={(e) => setPFire(parseFloat(e.target.value))}
                  className="w-32 accent-red-500"
                />
                <span className="font-mono text-xs font-bold text-red-600">{(pFire * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700"><Trans i18nKey="bayes.fire_case.p_smoke">Prob <Latex formula="P(\text{Smoke})" /></Trans></span>
                <input
                  type="range" min="0.01" max="0.5" step="0.01" value={pSmoke}
                  onChange={(e) => setPSmoke(parseFloat(e.target.value))}
                  className="w-32 accent-slate-500"
                />
                <span className="font-mono text-xs font-bold text-slate-600">{(pSmoke * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-slate-700"><Trans i18nKey="bayes.fire_case.p_smoke_given_fire">Prob <Latex formula="P(\text{Smoke}|\text{Fire})" /></Trans></span>
                <input
                  type="range" min="0.5" max="1" step="0.01" value={pSmokeGivenFire}
                  onChange={(e) => setPSmokeGivenFire(parseFloat(e.target.value))}
                  className="w-32 accent-purple-500"
                />
                <span className="font-mono text-xs font-bold text-purple-600">{(pSmokeGivenFire * 100).toFixed(1)}%</span>
              </div>
            </div>

            <SeniorAdvice content={<Trans i18nKey="bayes.fire_case.advice">
              讲义中的经典案例... <Latex formula="9\%" />... <b>Evidence ≠ Conclusion...</b>
            </Trans>} />
          </div>

          <div className="flex flex-col justify-center items-center bg-slate-900 rounded-3xl p-8 text-white relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10"><Wind size={100} /></div>
            <div className="text-center z-10">
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">{t('bayes.fire_case.result_title')}</p>
              <p className="text-5xl font-black text-red-400 mb-2">{(pFireGivenSmoke * 100).toFixed(2)}%</p>
              <div className="mt-6 p-4 bg-slate-800 rounded-xl border border-slate-700 text-xs font-mono leading-relaxed">
                <p className="mb-2 text-slate-400">{t('bayes.fire_case.calc_process')}</p>
                <Latex formula={`P(F|S) = \\frac{${pFire} \\times ${pSmokeGivenFire}}{${pSmoke}}`} />
                <p className="mt-2 text-red-300">= {pFireGivenSmoke.toFixed(4)}</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
