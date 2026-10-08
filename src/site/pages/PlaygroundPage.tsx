import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import HomeLab from './HomeLab'

const GRID = { backgroundImage: 'linear-gradient(#e7ebf1 1px, transparent 1px), linear-gradient(90deg, #e7ebf1 1px, transparent 1px)', backgroundSize: '32px 32px' }

/** /playground/: every interactive lab as a toolkit grid, filtered by area. */
export default function PlaygroundPage() {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${t('site.playground.title')} · KnowCS`
  }, [t])
  return (
    <main className="min-h-screen pb-10 pt-6" style={GRID}>
      <div className="mx-auto max-w-[1280px] px-4 sm:px-8">
        <h1 className="text-[clamp(26px,3.2vw,36px)] font-black tracking-tight text-slate-900">{t('site.playground.title')}</h1>
      </div>
      <div className="mt-3">
        <HomeLab />
      </div>
    </main>
  )
}
