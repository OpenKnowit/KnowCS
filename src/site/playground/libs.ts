import type { ComponentType } from 'react'
import i18n from '../../i18n'
import { ensurePacks } from '../../i18nPacks'
import type { RunResult } from '../../lib/minipy'
import { PLAYGROUND_LOADERS } from './loaders'

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

/** one example in the workbench's list */
export interface BenchEntry {
  id: string
  cat: string
  label: string
  code: string
  /** step to open first (api name or event type); default = the last step */
  focus?: string[]
  focusFirst?: boolean
}

/** What the workbench needs from a library: its examples, where their strings live, and how to run code. */
export interface LibSpec {
  entries: BenchEntry[]
  cats: string[]
  /** example opened first */
  start: string
  catKey: (cat: string) => string
  explainKey: (entry: string) => string
  /** one line on what the sandbox can and cannot do */
  noteKey: string
  run: (code: string) => RunResult
  /** traced calls listed as steps, by api prefix; null = every call */
  callApis: string[] | null
  /** ms after the last keystroke before re-running */
  debounce: number
  /** real Python (Pyodide) has a build of this library */
  realOk: boolean
}

export interface BenchProps {
  lib: LibId
  spec: LibSpec
  side: boolean
  onSide: (open: boolean) => void
  onPickLib: (id: LibId) => void
}

export type LoadedLib = { id: LibId; spec: LibSpec; Bench: ComponentType<BenchProps> }

const SHARED = ['playground.ui', 'playground.view', 'playground.note']

/** The library's examples, strings and runner, plus the workbench that shows them. */
export function loadLib(id: LibId): Promise<LoadedLib> {
  const bench = import('./Workbench')
  if (id === 'numpy') {
    return Promise.all([ensurePacks(i18n, ['numpy_api', ...SHARED]), import('../../data/numpyApis'), import('../../lib/minipy'), bench]).then(([, d, py, m]) => ({
      id,
      Bench: m.default,
      spec: {
        entries: d.NUMPY_APIS.map((e) => ({ id: e.id, cat: e.cat, label: e.label, code: e.code, focus: e.match, focusFirst: true })),
        cats: d.API_CATS,
        start: d.NUMPY_APIS.find((e) => e.cat === 'reduce')!.id,
        catKey: (c) => `numpy_api.cat.${c}`,
        explainKey: (e) => `numpy_api.api.${e}`,
        noteKey: 'numpy_api.sandbox_note',
        run: (code) => py.runPython(code),
        callApis: null,
        debounce: 300,
        realOk: true,
      },
    }))
  }
  return Promise.all([PLAYGROUND_LOADERS[id]!(), import('../../data/playgrounds/run'), import('../../lib/pyodide/config'), bench]).then(([config, r, py, m]) => ({
    id,
    Bench: m.default,
    spec: {
      entries: config.entries,
      cats: config.cats,
      start: config.entries[0].id,
      catKey: (c) => `playground.${id}.cat.${c}`,
      explainKey: (e) => `playground.${id}.e.${e}`,
      noteKey: `playground.${id}.sandbox_note`,
      run: (code) => r.runIn(config, code),
      callApis: config.callApis,
      debounce: config.debounce,
      realOk: py.REAL_PYTHON_LIBS.has(id),
    },
  }))
}
