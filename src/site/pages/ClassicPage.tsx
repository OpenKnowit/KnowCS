import type { ComponentType, LazyExoticComponent, ReactNode } from 'react'
import { lazy } from 'react'
import { ChevronDown, GraduationCap } from 'lucide-react'
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
      <main className="mx-auto max-w-[1480px] px-4 pb-10 pt-5 sm:px-6">
        <ModuleHero id={id} />
        <div className="min-w-0 rounded-[2rem] border border-slate-200 bg-white p-5 shadow-xl shadow-slate-200/40 sm:p-8">
          {/* the exam tip stays folded until asked for, so the interactive part comes first */}
          <details className="group mb-6 rounded-2xl border border-blue-200 bg-blue-50/60">
            <summary className="flex cursor-pointer list-none items-center gap-2.5 rounded-2xl px-4 py-3 text-sm font-bold text-blue-900 hover:bg-blue-50 [&::-webkit-details-marker]:hidden">
              <GraduationCap size={17} className="text-blue-600" aria-hidden />
              {t('app.sidebar.exam_tip.title')}
              <ChevronDown size={16} className="ml-auto text-blue-700 transition-transform group-open:rotate-180" aria-hidden />
            </summary>
            <p className="px-4 pb-3.5 text-[13.5px] leading-relaxed text-slate-700">{tip}</p>
          </details>
          <Motion>
            <Component />
          </Motion>
        </div>
      </main>
    </>
  )
}
