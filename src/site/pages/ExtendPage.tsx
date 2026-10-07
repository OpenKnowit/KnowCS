import { useEffect } from 'react'
import { Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { SectionTitle } from '../../components/SectionTitle'
import { ExtendModule } from '../../modules/ExtendModule'
import { extendUrl } from '../../lib/sitemap'

/** /extend/ (gallery) and /extend/<id>/ (one self-contained page in an iframe). */
export default function ExtendPage({ id }: { id: string | null }) {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${id ? t(`extend.items.${id}.title`) : t('app.section.extend.title')} · KnowCS`
  }, [id, t])
  return (
    <main className="mx-auto max-w-[1480px] px-4 pb-10 pt-4 sm:px-6">
      <div className="rounded-[2rem] border border-slate-200 bg-white p-5 shadow-2xl shadow-slate-200/50 sm:p-8 md:p-12">
        <SectionTitle icon={Sparkles} title={t('app.section.extend.title')} subtitle={t('app.section.extend.subtitle')} />
        <ExtendModule openId={id} onOpen={(next) => window.location.assign(next ? extendUrl(next) : extendUrl())} />
      </div>
    </main>
  )
}
