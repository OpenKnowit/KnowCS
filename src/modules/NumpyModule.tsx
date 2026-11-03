import { Code2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { NumpyPlayground } from './NumpyPlayground'

// L1 · NumPy indexing: every indexing operation traced (which cells are picked, view or copy).
// Broadcasting has its own chapter: src/site/modules/NumpyBroadcasting.tsx.
export const NumpyModule = () => {
  const { t } = useTranslation()
  return (
    <div className="space-y-8">
      <div>
        <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold">
          <Code2 size={18} className="text-blue-500" />
          {t('numpy_module.playground.title')}
        </h3>
        <NumpyPlayground />
      </div>
    </div>
  )
}
