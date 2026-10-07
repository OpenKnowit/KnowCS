import { lazy } from 'react'
import type { ComponentType, LazyExoticComponent, ReactNode } from 'react'
import { Info } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../../components/Latex'
import type { ModuleId } from '../../lib/sitemap'
import { ModuleHero, PageBar } from '../ui'

const named = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K): LazyExoticComponent<ComponentType> =>
  lazy(() => load().then((m) => ({ default: m[name] })))

// Pages added with the lab redesign render their own LabPage (page bar, hero, quiz mode).
const LAB: Partial<Record<ModuleId, LazyExoticComponent<ComponentType>>> = {
  'bayes-virus': lazy(() => import('../modules/BayesVirus')),
  'gaussian-nb': lazy(() => import('../modules/GaussianNb')),
  evaluation: lazy(() => import('../modules/Evaluation')),
  'cross-validation': lazy(() => import('../modules/CrossValidation')),
  'kmeans-table': lazy(() => import('../modules/KMeansTable')),
  perceptron: lazy(() => import('../modules/Perceptron')),
  'xor-mlp': lazy(() => import('../modules/XorMlp')),
  convolution: lazy(() => import('../modules/Convolution')),
  otsu: lazy(() => import('../modules/Otsu')),
  affine: lazy(() => import('../modules/Affine')),
  'cnn-shapes': lazy(() => import('../modules/CnnShapes')),
  pytorch: lazy(() => import('../modules/PyTorch')),
}

// The original modules: their component plus the exam tip that used to sit in the sidebar.
const CLASSIC: Partial<Record<ModuleId, { Component: LazyExoticComponent<ComponentType>; tip: ReactNode; wide?: boolean }>> = {
  numpy: {
    Component: named(() => import('../../modules/NumpyModule'), 'NumpyModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_numpy" components={{ 1: <Latex formula="\text{Broadcasting}" />, 3: <Latex formula="\mathbf{A} \cdot \mathbf{B}" />, 5: <Latex formula="\mathbf{A} \odot \mathbf{B}" /> }} />,
  },
  'bayes-basics': {
    Component: named(() => import('../../modules/BayesBasicsModule'), 'BayesBasicsModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_bayesBasics" components={{ 1: <Latex formula="P(B|E) \propto P(B) \cdot P(E|B)" /> }} />,
  },
  'naive-bayes': {
    Component: named(() => import('../../modules/NaiveBayesModule'), 'NaiveBayesModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_naiveBayes" components={{ 1: <Latex formula="\alpha" />, 3: <Latex formula="\log" /> }} />,
  },
  knn: {
    Component: named(() => import('../../modules/KnnModule'), 'KnnModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_knn" components={{ 1: <Latex formula="K" /> }} />,
  },
  kmeans: {
    Component: named(() => import('../../modules/KMeansModule'), 'KMeansModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_kmeans" components={{ 1: <Latex formula="\text{WCSS}" />, 3: <Latex formula="K" /> }} />,
  },
  backprop: {
    Component: named(() => import('../../modules/BackpropModule'), 'BackpropModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_backprop" components={{ 1: <Latex formula="\delta_k" />, 3: <Latex formula="\Delta w = \eta \cdot \delta \cdot O" /> }} />,
  },
  kernel: {
    Component: named(() => import('../../modules/KernelModule'), 'KernelModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_kernel" components={{ 1: <Latex formula="N - K + 1" /> }} />,
  },
  alphabeta: {
    Component: named(() => import('../../modules/AlphaBetaModule'), 'AlphaBetaModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_alphabeta" components={{ 1: <Latex formula="\beta \leq \alpha" />, 3: <strong className="font-bold" /> }} />,
  },
}

function ClassicPage({ id }: { id: ModuleId }) {
  const { t } = useTranslation()
  const { Component, tip } = CLASSIC[id]!
  return (
    <>
      <PageBar id={id} />
      <main className="mx-auto max-w-[1480px] px-4 pb-10 pt-6 sm:px-6">
        <ModuleHero id={id} />
        <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
          <div className="min-w-0 rounded-[2rem] border border-slate-200 bg-white p-5 shadow-xl shadow-slate-200/40 sm:p-8">
            <Component />
          </div>
          <aside className="relative overflow-hidden rounded-[1.5rem] bg-gradient-to-br from-blue-600 to-indigo-700 p-6 text-white shadow-lg shadow-blue-200 xl:sticky xl:top-16">
            <div className="mb-3 flex items-center gap-2">
              <Info size={16} aria-hidden />
              <span className="text-xs font-bold uppercase tracking-wider">{t('app.sidebar.exam_tip.title')}</span>
            </div>
            <p className="text-[13px] font-medium leading-relaxed opacity-95">{tip}</p>
          </aside>
        </div>
      </main>
    </>
  )
}

export default function ModulePage({ id }: { id: string }) {
  const Lab = LAB[id as ModuleId]
  if (Lab) return <Lab />
  if (CLASSIC[id as ModuleId]) return <ClassicPage id={id as ModuleId} />
  return <p className="p-10 text-center text-slate-500">404</p>
}
