import type { ComponentType } from 'react'
import i18n from '../../i18n'
import { ensurePacks } from '../../i18nPacks'
import type { PlayConfig } from '../../data/playgrounds/types'
import { PLAYGROUND_LOADERS, loadPlaygroundView } from './loaders'

// The libraries on /playground/, in lecture order. The chosen one sits in the URL hash (/playground/#keras).

export const LIBS = [
  { id: 'numpy', name: 'NumPy', line: 'import numpy as np', color: '#0284c7' },
  { id: 'matplotlib', name: 'matplotlib', line: 'import matplotlib.pyplot as plt', color: '#0d9488' },
  { id: 'pandas', name: 'pandas', line: 'import pandas as pd', color: '#7c3aed' },
  { id: 'pytorch', name: 'PyTorch', line: 'import torch', color: '#e11d48' },
  { id: 'keras', name: 'Keras', line: 'import keras', color: '#d97706' },
  { id: 'tensorflow', name: 'TensorFlow', line: 'import tensorflow as tf', color: '#ea580c' },
] as const
export type LibId = (typeof LIBS)[number]['id']

export const libFromHash = (): LibId => {
  const h = typeof window === 'undefined' ? '' : window.location.hash.slice(1)
  return LIBS.find((l) => l.id === h)?.id ?? 'numpy'
}

export type LoadedLib =
  | { id: LibId; kind: 'numpy'; View: ComponentType<{ wide?: boolean }> }
  | { id: LibId; kind: 'lib'; View: ComponentType<{ config: PlayConfig; wide?: boolean }>; config: PlayConfig }

/** The sandbox's code, examples and strings (NumPy uses its API panel; the rest share one playground view). */
export function loadLib(id: LibId): Promise<LoadedLib> {
  if (id === 'numpy') {
    return Promise.all([ensurePacks(i18n, ['numpy_api', 'playground.ui']), import('../../modules/NumpyApiPanel')]).then(([, m]) => ({ id, kind: 'numpy' as const, View: m.NumpyApiPanel }))
  }
  return Promise.all([loadPlaygroundView(), PLAYGROUND_LOADERS[id]!()]).then(([m, config]) => ({ id, kind: 'lib' as const, config, View: m.default }))
}
