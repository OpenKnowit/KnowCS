import { lazy } from 'react'
import type { ComponentType, LazyExoticComponent, ReactNode } from 'react'
import { Info } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../../components/Latex'
import type { ModuleId } from '../../lib/sitemap'
import { ModuleHero, PageBar } from '../ui'
import i18n from '../../i18n'
import { withPacks } from '../../i18nPacks'

const named = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K): LazyExoticComponent<ComponentType> =>
  lazy(() => load().then((m) => ({ default: m[name] })))

// Pages added with the lab redesign render their own LabPage (page bar, hero, quiz mode).
const lab = (pack: string, load: () => Promise<{ default: ComponentType }>) => lazy(withPacks(i18n, [`lab.${pack}`], load))
const LAB: Partial<Record<ModuleId, LazyExoticComponent<ComponentType>>> = {
  'bayes-virus': lab('bayes', () => import('../modules/BayesVirus')),
  'gaussian-nb': lab('gnb', () => import('../modules/GaussianNb')),
  evaluation: lab('eval', () => import('../modules/Evaluation')),
  'cross-validation': lab('cv', () => import('../modules/CrossValidation')),
  'kmeans-table': lab('km', () => import('../modules/KMeansTable')),
  perceptron: lab('perc', () => import('../modules/Perceptron')),
  'xor-mlp': lab('xor', () => import('../modules/XorMlp')),
  convolution: lab('conv', () => import('../modules/Convolution')),
  otsu: lab('otsu', () => import('../modules/Otsu')),
  affine: lab('affine', () => import('../modules/Affine')),
  'cnn-shapes': lab('cnn', () => import('../modules/CnnShapes')),
  pytorch: lab('pytorch', () => import('../modules/PyTorch')),
}

// The original modules: their component plus the exam tip that used to sit in the sidebar.
const CLASSIC: Partial<Record<ModuleId, { Component: LazyExoticComponent<ComponentType>; tip: ReactNode; wide?: boolean }>> = {
  numpy: {
    Component: named(withPacks(i18n, ['numpy_module', 'numpy_api'], () => import('../../modules/NumpyModule')), 'NumpyModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_numpy" components={{ 1: <Latex formula="\text{Broadcasting}" />, 3: <Latex formula="\mathbf{A} \cdot \mathbf{B}" />, 5: <Latex formula="\mathbf{A} \odot \mathbf{B}" /> }} />,
  },
  'bayes-basics': {
    Component: named(withPacks(i18n, ['bayes'], () => import('../../modules/BayesBasicsModule')), 'BayesBasicsModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_bayesBasics" components={{ 1: <Latex formula="P(B|E) \propto P(B) \cdot P(E|B)" /> }} />,
  },
  'naive-bayes': {
    Component: named(withPacks(i18n, ['bayes'], () => import('../../modules/NaiveBayesModule')), 'NaiveBayesModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_naiveBayes" components={{ 1: <Latex formula="\alpha" />, 3: <Latex formula="\log" /> }} />,
  },
  knn: {
    Component: named(withPacks(i18n, ['knn'], () => import('../../modules/KnnModule')), 'KnnModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_knn" components={{ 1: <Latex formula="K" /> }} />,
  },
  kmeans: {
    Component: named(withPacks(i18n, ['kmeans'], () => import('../../modules/KMeansModule')), 'KMeansModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_kmeans" components={{ 1: <Latex formula="\text{WCSS}" />, 3: <Latex formula="K" /> }} />,
  },
  backprop: {
    Component: named(withPacks(i18n, ['backprop_module'], () => import('../../modules/BackpropModule')), 'BackpropModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_backprop" components={{ 1: <Latex formula="\delta_k" />, 3: <Latex formula="\Delta w = \eta \cdot \delta \cdot O" /> }} />,
  },
  kernel: {
    Component: named(withPacks(i18n, ['kernel_module'], () => import('../../modules/KernelModule')), 'KernelModule'),
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_kernel" components={{ 1: <Latex formula="N - K + 1" /> }} />,
  },
  alphabeta: {
    Component: named(withPacks(i18n, ['alphabeta'], () => import('../../modules/AlphaBetaModule')), 'AlphaBetaModule'),
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
