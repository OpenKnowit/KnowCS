import type { ComponentType, LazyExoticComponent, ReactNode } from 'react'
import { lazy } from 'react'
import { Info } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../../components/Latex'
import { Motion } from '../../components/Motion'
import type { ModuleId } from '../../lib/sitemap'
import { CLASSIC_LOADERS } from '../pageLoaders'
import { ModuleHero, PageBar } from '../ui'

// Only the original modules use KaTeX in their exam tip, so it lives here, out of the lab pages' way.
const COMPONENTS = Object.fromEntries(Object.entries(CLASSIC_LOADERS).map(([id, load]) => [id, lazy(load!)])) as Partial<Record<ModuleId, LazyExoticComponent<ComponentType>>>

const TIPS: Partial<Record<ModuleId, ReactNode>> = {
  numpy: <Trans i18nKey="app.sidebar.exam_tip.content_numpy" components={{ 1: <Latex formula="\text{Broadcasting}" />, 3: <Latex formula="\mathbf{A} \cdot \mathbf{B}" />, 5: <Latex formula="\mathbf{A} \odot \mathbf{B}" /> }} />,
  'bayes-basics': <Trans i18nKey="app.sidebar.exam_tip.content_bayesBasics" components={{ 1: <Latex formula="P(B|E) \propto P(B) \cdot P(E|B)" /> }} />,
  'naive-bayes': <Trans i18nKey="app.sidebar.exam_tip.content_naiveBayes" components={{ 1: <Latex formula="\alpha" />, 3: <Latex formula="\log" /> }} />,
  knn: <Trans i18nKey="app.sidebar.exam_tip.content_knn" components={{ 1: <Latex formula="K" /> }} />,
  kmeans: <Trans i18nKey="app.sidebar.exam_tip.content_kmeans" components={{ 1: <Latex formula="\text{WCSS}" />, 3: <Latex formula="K" /> }} />,
  backprop: <Trans i18nKey="app.sidebar.exam_tip.content_backprop" components={{ 1: <Latex formula="\delta_k" />, 3: <Latex formula="\Delta w = \eta \cdot \delta \cdot O" /> }} />,
  kernel: <Trans i18nKey="app.sidebar.exam_tip.content_kernel" components={{ 1: <Latex formula="N - K + 1" /> }} />,
  alphabeta: <Trans i18nKey="app.sidebar.exam_tip.content_alphabeta" components={{ 1: <Latex formula="\beta \leq \alpha" />, 3: <strong className="font-bold" /> }} />,
}

export default function ClassicPage({ id }: { id: ModuleId }) {
  const { t } = useTranslation()
  const Component = COMPONENTS[id]!
  const tip = TIPS[id]
  return (
    <>
      <PageBar id={id} />
      <main className="mx-auto max-w-[1480px] px-4 pb-10 pt-6 sm:px-6">
        <ModuleHero id={id} />
        <div className="min-w-0 rounded-[2rem] border border-slate-200 bg-white p-5 shadow-xl shadow-slate-200/40 sm:p-8">
          {/* the exam tip lives inside the module card, so the interactive part gets the full width */}
          <aside className="mb-6 flex flex-col gap-1.5 rounded-2xl bg-gradient-to-r from-blue-600 to-indigo-600 px-4 py-3 text-white shadow-md shadow-blue-200 sm:flex-row sm:items-center sm:gap-3">
            <span className="flex shrink-0 items-center gap-1.5 text-xs font-bold uppercase tracking-wider">
              <Info size={16} aria-hidden /> {t('app.sidebar.exam_tip.title')}
            </span>
            <p className="text-[13px] font-medium leading-relaxed opacity-95">{tip}</p>
          </aside>
          <Motion>
            <Component />
          </Motion>
        </div>
      </main>
    </>
  )
}
