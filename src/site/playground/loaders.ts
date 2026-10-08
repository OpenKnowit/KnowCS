/**
 * The library playgrounds shown beside a note (/notes/<id>/): each loader brings the playground's config
 * (its sandbox libraries and examples) and its strings. The NumPy note keeps its own API panel.
 */
import i18n from '../../i18n'
import { withPacks } from '../../i18nPacks'
import type { PlayConfig } from '../../data/playgrounds/types'

const memo = <T,>(f: () => Promise<T>): (() => Promise<T>) => {
  let p: Promise<T> | null = null
  return () => (p ??= f())
}

/** strings every playground needs; numpy_api is for the element-provenance view of array / tensor operations */
const SHARED = ['numpy_api', 'playground.ui', 'playground.view', 'playground.note']

export const PLAYGROUND_LOADERS: Partial<Record<string, () => Promise<PlayConfig>>> = {
  matplotlib: memo(withPacks(i18n, [...SHARED, 'playground.matplotlib'], () => import('../../data/playgrounds/matplotlib').then((m) => m.MATPLOTLIB_PLAYGROUND))),
  pytorch: memo(withPacks(i18n, [...SHARED, 'playground.pytorch'], () => import('../../data/playgrounds/pytorch').then((m) => m.PYTORCH_PLAYGROUND))),
  keras: memo(withPacks(i18n, [...SHARED, 'playground.keras'], () => import('../../data/playgrounds/keras').then((m) => m.KERAS_PLAYGROUND))),
  tensorflow: memo(withPacks(i18n, [...SHARED, 'playground.tensorflow'], () => import('../../data/playgrounds/tensorflow').then((m) => m.TENSORFLOW_PLAYGROUND))),
}

export const loadPlaygroundView = memo(() => import('./Playground'))
