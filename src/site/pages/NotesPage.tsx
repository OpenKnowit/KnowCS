import { useEffect } from 'react'
import { BookOpen } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { SectionTitle } from '../../components/SectionTitle'
import { PackageModule } from '../../modules/PackageModule'
import { noteUrl } from '../../lib/sitemap'

/** /notes/ (list) and /notes/<id>/ (one note; the NumPy note brings the API panel). */
export default function NotesPage({ id }: { id: string | null }) {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${id ? t(`package.notes.${id}`) : t('app.section.package.title')} · KnowCS`
  }, [id, t])
  return (
    <main className="mx-auto max-w-[1480px] px-4 pb-10 pt-4 sm:px-6">
      <div className="rounded-[2rem] border border-slate-200 bg-white p-5 shadow-2xl shadow-slate-200/50 sm:p-8 md:p-12">
        <SectionTitle as={id ? 'h2' : 'h1'} icon={BookOpen} title={t('app.section.package.title')} subtitle={t('app.section.package.subtitle')} />
        <PackageModule openId={id} onOpen={(next) => window.location.assign(next ? noteUrl(next) : noteUrl())} />
      </div>
    </main>
  )
}
