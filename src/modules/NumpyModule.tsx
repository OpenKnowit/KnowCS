import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowRightLeft, ChevronRight, Grid3X3 } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { INITIAL_MATRIX } from '../data/constants'
import type { SliceType } from '../types'

interface SliceInfo {
  label: string
  color: string
  desc: string
}

// NumPy Mechanism
export const NumpyModule = () => {
  const [sliceType, setSliceType] = useState<SliceType>('none')
  const { t } = useTranslation()

  const getHighlight = (r: number, _c: number, val: number): boolean => {
    if (sliceType === 'slice') return r === 1
    if (sliceType === 'fancy') return r === 0 || r === 2
    if (sliceType === 'mask') return val % 2 === 0
    return false
  }

  const getInfo = (): SliceInfo | null => {
    switch (sliceType) {
      case 'slice': return { label: t('numpy_module.view_label'), color: 'bg-green-500', desc: t('numpy_module.view_desc') }
      case 'fancy': return { label: t('numpy_module.fancy_label'), color: 'bg-blue-500', desc: t('numpy_module.fancy_desc') }
      case 'mask': return { label: t('numpy_module.mask_label'), color: 'bg-purple-500', desc: t('numpy_module.mask_desc') }
      default: return null
    }
  }

  const info = getInfo()

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Grid3X3 size={18} className="text-blue-500" />
          {t('numpy_module.title')}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className="grid grid-cols-4 gap-2 bg-gray-100 p-4 rounded-xl shadow-inner">
              {INITIAL_MATRIX.map((row, r) =>
                row.map((val, c) => (
                  <motion.div
                    key={`${r}-${c}`}
                    animate={{
                      scale: getHighlight(r, c, val) ? 1.05 : 1,
                      backgroundColor: getHighlight(r, c, val) ? (info ? '#dcfce7' : '#ffffff') : '#ffffff',
                    }}
                    className={`h-12 flex items-center justify-center rounded border text-sm font-mono
                      ${getHighlight(r, c, val) ? 'border-green-500 border-2 z-10' : 'border-gray-200 text-gray-400'}`}
                  >
                    {val}
                  </motion.div>
                ))
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              <button onClick={() => setSliceType('slice')} className={`px-4 py-2 rounded-lg text-xs font-mono transition shadow-sm ${sliceType === 'slice' ? 'bg-green-600 text-white' : 'bg-white border border-gray-200'}`}>{t('numpy_module.slice_btn')}</button>
              <button onClick={() => setSliceType('fancy')} className={`px-4 py-2 rounded-lg text-xs font-mono transition shadow-sm ${sliceType === 'fancy' ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200'}`}>{t('numpy_module.fancy_btn')}</button>
              <button onClick={() => setSliceType('mask')} className={`px-4 py-2 rounded-lg text-xs font-mono transition shadow-sm ${sliceType === 'mask' ? 'bg-purple-600 text-white' : 'bg-white border border-gray-200'}`}>{t('numpy_module.mask_btn')}</button>
            </div>
          </div>

          <div className="flex flex-col justify-center">
            {info ? (
              <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="p-4 rounded-xl border-2 border-dashed border-gray-200 bg-white shadow-sm">
                <div className={`inline-block px-2 py-1 rounded text-white text-[10px] font-bold mb-2 uppercase tracking-wider ${info.color}`}>
                  {info.label}
                </div>
                <p className="text-sm text-gray-600 leading-relaxed">{info.desc}</p>
              </motion.div>
            ) : (
              <p className="text-sm text-gray-400 italic">{t('numpy_module.instruction')}</p>
            )}
          </div>
        </div>
      </div>

      <div className="border-t pt-8">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <ArrowRightLeft size={18} className="text-blue-500" />
          {t('numpy_module.broadcast_title')}
        </h3>
        <div className="bg-gray-50 p-6 rounded-xl border border-gray-100">
          <div className="flex flex-col md:flex-row items-center justify-center gap-12">
            {/* Shape A */}
            <div className="text-center">
              <div className="mb-2"><Latex formula="(3, 1)" className="text-xs text-gray-500 font-bold" /></div>
              <div className="grid grid-cols-1 gap-1">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="flex gap-1">
                    <div className="w-8 h-8 bg-blue-400 rounded flex items-center justify-center text-white text-[10px] font-bold">A</div>
                    <div className="w-8 h-8 bg-blue-100 rounded border-dashed border-2 border-blue-200 flex items-center justify-center text-blue-300 text-[10px]">?</div>
                    <div className="w-8 h-8 bg-blue-100 rounded border-dashed border-2 border-blue-200 flex items-center justify-center text-blue-300 text-[10px]">?</div>
                    <div className="w-8 h-8 bg-blue-100 rounded border-dashed border-2 border-blue-200 flex items-center justify-center text-blue-300 text-[10px]">?</div>
                  </div>
                ))}
              </div>
            </div>
            <div className="text-gray-400 font-bold text-2xl">+</div>
            {/* Shape B */}
            <div className="text-center">
              <div className="mb-2"><Latex formula="(1, 4)" className="text-xs text-gray-500 font-bold" /></div>
              <div className="grid grid-cols-1 gap-1">
                <div className="flex gap-1">
                  {[1, 2, 3, 4].map((i) => <div key={i} className="w-8 h-8 bg-orange-400 rounded flex items-center justify-center text-white text-[10px] font-bold">B</div>)}
                </div>
                {[1, 2].map((i) => (
                  <div key={i} className="flex gap-1">
                    {[1, 2, 3, 4].map((j) => <div key={j} className="w-8 h-8 bg-orange-100 rounded border-dashed border-2 border-orange-200 flex items-center justify-center text-orange-300 text-[10px]">?</div>)}
                  </div>
                ))}
              </div>
            </div>
            <ChevronRight className="rotate-90 md:rotate-0 text-gray-300" />
            <div className="text-center">
               <div className="mb-2"><Latex formula="(3, 4)" className="text-xs text-green-600 font-bold" /></div>
               <div className="grid grid-cols-4 gap-1 p-1 bg-green-50 rounded border border-green-200">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div key={i} className="w-8 h-8 bg-green-400 rounded flex items-center justify-center text-white text-[10px] font-bold">A+B</div>
                ))}
               </div>
            </div>
          </div>
          <SeniorAdvice content={
            <Trans i18nKey="numpy_module.broadcast_advice">
              Broadcasting isn't actually copying data in memory... <Latex formula="1" />.
            </Trans>
          } />
        </div>
      </div>
    </div>
  )
}
