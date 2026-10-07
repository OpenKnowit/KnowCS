import { ArrowRightLeft, Clapperboard, Code2 } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { NumpyBroadcast } from './NumpyBroadcast'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { NumpyPlayground } from './NumpyPlayground'
import { NumpyPairwiseVideo } from './NumpyPairwiseVideo'

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
          {t('numpy_module.broadcast.title')}
        </h3>
        <NumpyBroadcast />
        <SeniorAdvice content={<Trans i18nKey="numpy_module.broadcast.advice" components={{ 1: <Latex formula="1" /> }} />} />
      </div>

      <div className="border-t pt-8">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Clapperboard size={18} className="text-blue-500" />
          {t('numpy_module.pairwise.title')}
        </h3>
        <NumpyPairwiseVideo />
        <SeniorAdvice content={t('numpy_module.pairwise.advice')} />
      </div>
    </div>
  )
}
