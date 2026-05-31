import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Zap } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface SeniorAdviceProps {
  content: ReactNode
}

export const SeniorAdvice = ({ content }: SeniorAdviceProps) => {
  const { t } = useTranslation()
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-4 p-4 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg"
    >
      <div className="flex gap-2">
        <Zap className="text-amber-500 shrink-0" size={18} />
        <div>
          <span className="font-bold text-amber-800 text-sm">{t('senior_advice')}</span>
          <div className="text-amber-900 text-sm leading-relaxed inline">{content}</div>
        </div>
      </div>
    </motion.div>
  )
}
