import { useMemo, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { KERNEL_PRESETS } from '../data/constants'
import type { ActivePreset, KernelPresetName } from '../types'

// Convolution Module
export const KernelModule = () => {
  const [kernel, setKernel] = useState<number[][]>(KERNEL_PRESETS['Sobel-X'])
  const [activePreset, setActivePreset] = useState<ActivePreset>('Sobel-X')
  const { t } = useTranslation()

  const sourceImage = useMemo<number[][]>(() => {
    const img: number[][] = Array.from({ length: 8 }, () => Array<number>(8).fill(0))
    for (let i = 2; i < 6; i++) {
      for (let j = 2; j < 6; j++) img[i][j] = 100
    }
    return img
  }, [])

  const handleKernelChange = (r: number, c: number, val: string) => {
    const newKernel = kernel.map((row, ri) =>
      row.map((v, ci) => (ri === r && ci === c ? Number(val) : v))
    )
    setKernel(newKernel)
    setActivePreset('Custom')
  }

  const applyPreset = (name: KernelPresetName) => {
    setKernel(KERNEL_PRESETS[name])
    setActivePreset(name)
  }

  const getConvolvedVal = (ri: number, ci: number): number => {
    let sum = 0
    for (let kr = 0; kr < 3; kr++) {
      for (let kc = 0; kc < 3; kc++) {
        sum += sourceImage[ri + kr][ci + kc] * kernel[kr][kc]
      }
    }
    return Math.max(0, Math.min(255, Math.abs(sum)))
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap gap-4 items-center justify-between bg-slate-50 p-4 rounded-xl border">
        <div className="flex gap-2">
          {(Object.keys(KERNEL_PRESETS) as KernelPresetName[]).map((name) => (
            <button
              key={name}
              onClick={() => applyPreset(name)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all shadow-sm ${activePreset === name ? 'bg-indigo-600 text-white shadow-indigo-200' : 'bg-white border text-slate-600 hover:bg-slate-100'}`}
            >
              {name}
            </button>
          ))}
        </div>
        <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">
           {t('kernel_module.operation')} <Latex formula="(I * K)_{x,y}" />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="flex flex-col items-center">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">{t('kernel_module.input_image')} <Latex formula="I" /></span>
          <div className="grid grid-cols-8 gap-px bg-slate-300 p-px border-2 border-slate-200 rounded shadow-sm">
            {sourceImage.map((row, ri) =>
              row.map((val, ci) => (
                <div key={`${ri}-${ci}`} className="w-6 h-6 sm:w-8 sm:h-8" style={{ backgroundColor: `rgb(${val},${val},${val})` }} />
              ))
            )}
          </div>
        </div>

        <div className="flex flex-col items-center justify-center relative">
          <div className="text-3xl font-bold text-slate-200 absolute -left-6 top-1/2 -translate-y-1/2 hidden lg:block">×</div>
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">{t('kernel_module.kernel')} <Latex formula="K" /></span>
          <div className="grid grid-cols-3 gap-2 bg-indigo-50 p-4 rounded-xl border-2 border-indigo-100 shadow-inner">
            {kernel.map((row, ri) =>
              row.map((val, ci) => (
                <input
                  key={`${ri}-${ci}`}
                  type="number"
                  value={val}
                  onChange={(e) => handleKernelChange(ri, ci, e.target.value)}
                  className="w-10 h-10 sm:w-12 sm:h-12 bg-white border border-indigo-200 rounded-lg text-center font-mono font-bold text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-400 transition"
                />
              ))
            )}
          </div>
        </div>

        <div className="flex flex-col items-center relative">
          <div className="text-3xl font-bold text-slate-200 absolute -left-6 top-1/2 -translate-y-1/2 hidden lg:block">=</div>
          <span className="text-xs font-bold uppercase tracking-wider text-green-600 mb-4">{t('kernel_module.output_feature_map')}</span>
          <div className="grid grid-cols-6 gap-px bg-slate-300 p-px border-2 border-slate-200 rounded shadow-sm">
            {Array.from({ length: 6 }).map((_, ri) =>
              Array.from({ length: 6 }).map((_, ci) => {
                const val = getConvolvedVal(ri, ci)
                return (
                  <div key={`${ri}-${ci}`} className="w-6 h-6 sm:w-8 sm:h-8" style={{ backgroundColor: `rgb(${val},${val},${val})` }} />
                )
              })
            )}
          </div>
        </div>
      </div>
      <SeniorAdvice content={
        <Trans i18nKey="kernel_module.advice">
          A convolution kernel... <Latex formula="\text{Laplacian}" /> ...
        </Trans>
      } />
    </div>
  )
}
