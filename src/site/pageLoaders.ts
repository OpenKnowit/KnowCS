/**
 * Every lazily loaded page and module, memoised, so main.tsx can start downloading a page's code and strings
 * right away — in parallel with the core strings — and the components reuse the same promise.
 */
import type { ComponentType } from 'react'
import i18n from '../i18n'
import { withPacks } from '../i18nPacks'
import type { ModuleId } from '../lib/sitemap'

type Loader = () => Promise<{ default: ComponentType<never> }>
const memo = <T,>(f: () => Promise<T>): (() => Promise<T>) => {
  let p: Promise<T> | null = null
  return () => (p ??= f())
}

/** Lab pages render their own LabPage; each brings its lab.<key> strings. */
export const LAB_LOADERS: Partial<Record<ModuleId, Loader>> = {
  'bayes-virus': memo(withPacks(i18n, ['lab.bayes'], () => import('./modules/BayesVirus'))),
  'gaussian-nb': memo(withPacks(i18n, ['lab.gnb'], () => import('./modules/GaussianNb'))),
  evaluation: memo(withPacks(i18n, ['lab.eval'], () => import('./modules/Evaluation'))),
  'cross-validation': memo(withPacks(i18n, ['lab.cv'], () => import('./modules/CrossValidation'))),
  'kmeans-table': memo(withPacks(i18n, ['lab.km'], () => import('./modules/KMeansTable'))),
  perceptron: memo(withPacks(i18n, ['lab.perc'], () => import('./modules/Perceptron'))),
  'xor-mlp': memo(withPacks(i18n, ['lab.xor'], () => import('./modules/XorMlp'))),
  convolution: memo(withPacks(i18n, ['lab.conv'], () => import('./modules/Convolution'))),
  otsu: memo(withPacks(i18n, ['lab.otsu'], () => import('./modules/Otsu'))),
  affine: memo(withPacks(i18n, ['lab.affine'], () => import('./modules/Affine'))),
  'cnn-shapes': memo(withPacks(i18n, ['lab.cnn'], () => import('./modules/CnnShapes'))),
  pytorch: memo(withPacks(i18n, ['lab.pytorch'], () => import('./modules/PyTorch'))),
}

/** The original modules (shown inside ClassicPage with an exam tip). */
export const CLASSIC_LOADERS: Partial<Record<ModuleId, () => Promise<{ default: ComponentType }>>> = {
  numpy: memo(withPacks(i18n, ['numpy_module', 'numpy_api'], () => import('../modules/NumpyModule').then((m) => ({ default: m.NumpyModule })))),
  'bayes-basics': memo(withPacks(i18n, ['bayes'], () => import('../modules/BayesBasicsModule').then((m) => ({ default: m.BayesBasicsModule })))),
  'naive-bayes': memo(withPacks(i18n, ['bayes'], () => import('../modules/NaiveBayesModule').then((m) => ({ default: m.NaiveBayesModule })))),
  knn: memo(withPacks(i18n, ['knn'], () => import('../modules/KnnModule').then((m) => ({ default: m.KnnModule })))),
  kmeans: memo(withPacks(i18n, ['kmeans'], () => import('../modules/KMeansModule').then((m) => ({ default: m.KMeansModule })))),
  backprop: memo(withPacks(i18n, ['backprop_module'], () => import('../modules/BackpropModule').then((m) => ({ default: m.BackpropModule })))),
  kernel: memo(withPacks(i18n, ['kernel_module'], () => import('../modules/KernelModule').then((m) => ({ default: m.KernelModule })))),
  alphabeta: memo(withPacks(i18n, ['alphabeta'], () => import('../modules/AlphaBetaModule').then((m) => ({ default: m.AlphaBetaModule })))),
}

export const PAGE_LOADERS = {
  home: memo(() => import('./pages/HomePage')),
  module: memo(() => import('./pages/ModulePage')),
  classic: memo(() => import('./pages/ClassicPage')),
  notes: memo(withPacks(i18n, ['numpy_api'], () => import('./pages/NotesPage'))),
  extend: memo(() => import('./pages/ExtendPage')),
  watch: memo(() => import('./pages/WatchPage')),
  drill: memo(withPacks(i18n, ['drill'], () => import('./pages/DrillPage'))),
  formulas: memo(withPacks(i18n, ['formulas.items'], () => import('./pages/FormulasPage'))),
  papers: memo(() => import('./pages/PapersPage')),
  notfound: memo(() => import('./pages/NotFoundPage')),
}

/** Kick off the downloads a page will need, before the core strings have arrived. */
export function preloadPage(page: string, id: string): void {
  const kind = page === 'note' ? 'notes' : page === 'extend-item' ? 'extend' : page === 'watch-item' ? 'watch' : page
  void (PAGE_LOADERS as Record<string, (() => Promise<unknown>) | undefined>)[kind]?.()
  if (page !== 'module') return
  const lab = LAB_LOADERS[id as ModuleId]
  if (lab) void lab()
  else if (CLASSIC_LOADERS[id as ModuleId]) void Promise.all([PAGE_LOADERS.classic(), CLASSIC_LOADERS[id as ModuleId]!()])
}
