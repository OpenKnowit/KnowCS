import { ArrowRightLeft, ChevronRight, Code2 } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { NumpyPlayground } from './NumpyPlayground'

// NumPy Mechanism
export const NumpyModule = () => {
  const { t } = useTranslation()

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Code2 size={18} className="text-blue-500" />
          {t('numpy_module.playground.title')}
        </h3>
        <NumpyPlayground />
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
