import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Activity, Play, Search } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'

// Backprop Module
export const BackpropModule = () => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [errorVal, setErrorVal] = useState(0.8)
  const { t } = useTranslation()

  const startTracking = () => {
    setIsAnimating(true)
    setTimeout(() => setIsAnimating(false), 2000)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
      <div className="space-y-6">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Activity size={18} className="text-red-500" />
          {t('backprop_module.title')}
        </h3>
        <div className="relative h-64 bg-slate-900 rounded-2xl overflow-hidden p-8 flex items-center justify-between">
          <div className="z-10 flex flex-col gap-12">
            <div className="w-10 h-10 rounded-full bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.6)] flex items-center justify-center text-white font-mono text-sm">i</div>
            <div className="w-10 h-10 rounded-full bg-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.6)] flex items-center justify-center text-white font-mono text-sm">j</div>
          </div>

          <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-40">
            <line x1="15%" y1="35%" x2="85%" y2="50%" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4" />
            <line x1="15%" y1="65%" x2="85%" y2="50%" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4" />

            <AnimatePresence>
              {isAnimating && (
                <motion.circle
                  initial={{ cx: '85%', cy: '50%', r: 4 }}
                  animate={{ cx: '15%', cy: '35%', r: 6 }}
                  transition={{ duration: 1.5, ease: 'easeInOut' }}
                  fill="#ef4444"
                />
              )}
            </AnimatePresence>
          </svg>

          <div className="z-10 flex flex-col">
            <div className="w-14 h-14 rounded-full bg-red-500 shadow-[0_0_20px_rgba(239,68,68,0.7)] flex items-center justify-center text-white font-mono border-4 border-red-200">k</div>
          </div>

          <div className="absolute top-4 right-4 text-right">
            <p className="text-[10px] text-slate-500 font-mono uppercase tracking-widest">{t('backprop_module.output_layer')}</p>
            <p className="text-xs text-red-400 font-bold flex items-center gap-1 justify-end">
              Error <Latex formula="\delta_k" />
            </p>
          </div>
        </div>

        <div className="bg-white border rounded-xl p-4 space-y-4 shadow-sm">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700">{t('backprop_module.error_val')} <Latex formula="(Target - Out)" />:</label>
            <input
              type="range" min="0" max="2" step="0.1" value={errorVal}
              onChange={(e) => setErrorVal(parseFloat(e.target.value))}
              className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-600"
            />
            <span className="font-mono text-red-600 font-bold">{errorVal}</span>
          </div>
          <button
            onClick={startTracking}
            disabled={isAnimating}
            className="w-full py-3 bg-red-600 text-white rounded-lg flex items-center justify-center gap-2 font-bold hover:bg-red-700 transition disabled:opacity-50 shadow-md"
          >
            <Play size={16} fill="currentColor" /> {isAnimating ? t('backprop_module.calculating') : t('backprop_module.track_grad')}
          </button>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl p-6 flex flex-col shadow-sm">
        <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2 italic border-b pb-2">
          <Search size={16} className="text-blue-500" />
          {t('backprop_module.notebook_title')}
        </h4>
        <div className="flex-1 space-y-4 overflow-y-auto">
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
            <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.step1_title')}</p>
            <Latex displayMode formula="\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial O_k} \cdot \frac{\partial O_k}{\partial net_k} \cdot \frac{\partial net_k}{\partial w_{jk}}" />
          </div>
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
            <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.step2_title')}</p>
            <p className="text-xs text-gray-600 mb-2">
              <Trans i18nKey="backprop_module.step2_desc">
                Let <Latex formula="\delta_k = - \frac{\partial E}{\partial net_k}" />. Under MSE loss:
              </Trans>
            </p>
            <Latex displayMode formula="\delta_k = (T_k - O_k) \cdot f'(net_k)" className="text-red-700" />
          </div>
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
            <p className="font-bold text-xs text-slate-400 uppercase mb-2">{t('backprop_module.step3_title')}</p>
            <Latex displayMode formula="\Delta w_{jk} = \eta \cdot \delta_k \cdot O_j" className="text-green-700" />
          </div>
          <SeniorAdvice content={
            <Trans i18nKey="backprop_module.advice">
              Remember, backpropagation... <Latex formula="w_{jk}" /> ... <Latex formula="E" /> ... <Latex formula="\delta_k" /> ...
            </Trans>
          } />
        </div>
      </div>
    </div>
  )
}
