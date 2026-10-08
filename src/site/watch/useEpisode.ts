import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { normalizeLang } from '../../lib/lang'
import { ensurePacks, hasPack } from '../../i18nPacks'
import type { Episode } from './Player'

/** One chunk per episode, so a page downloads only the explainers it shows. */
const LOADERS: Record<string, () => Promise<Episode>> = {
  affine: () => import('./episodes/affine').then((m) => m.affine),
  alphabeta: () => import('./episodes/alphabeta').then((m) => m.alphabeta),
  autograd: () => import('./episodes/autograd').then((m) => m.autograd),
  backprop: () => import('./episodes/backprop').then((m) => m.backpropEp),
  bayes: () => import('./episodes/bayes').then((m) => m.bayes),
  broadcast: () => import('./episodes/broadcast').then((m) => m.broadcast),
  cnn: () => import('./episodes/cnn').then((m) => m.cnn),
  dilated: () => import('./episodes/dilated').then((m) => m.dilated),
  ethics: () => import('./episodes/ethics').then((m) => m.ethics),
  evaluate: () => import('./episodes/evaluate').then((m) => m.evaluate),
  gaussian: () => import('./episodes/gaussian').then((m) => m.gaussian),
  imagenp: () => import('./episodes/imagenp').then((m) => m.imagenp),
  kmeans: () => import('./episodes/kmeans').then((m) => m.kmeans),
  knn: () => import('./episodes/knn').then((m) => m.knn),
  otsu: () => import('./episodes/otsu').then((m) => m.otsu),
  perceptron: () => import('./episodes/perceptron').then((m) => m.perceptron),
  xor: () => import('./episodes/xor').then((m) => m.xor),
}

export const episodeIds = (): string[] => Object.keys(LOADERS)

/** Packs an episode borrows besides its own captions (the CNN episode names layers with the CNN lab's words). */
export const EXTRA_PACKS: Record<string, string[]> = { cnn: ['lab.cnn'] }

const cache = new Map<string, Episode>()

/** The episode's code once its chunk and its strings in the current language have loaded (null until then). */
export function useEpisode(id: string): Episode | null {
  const { i18n } = useTranslation()
  const lang = normalizeLang(i18n.resolvedLanguage ?? i18n.language)
  const [, rerender] = useState(0)
  const packs = [`watch.${id}`, ...(EXTRA_PACKS[id] ?? [])]
  const ready = cache.has(id) && packs.every((p) => hasPack(p, lang))
  useEffect(() => {
    if (ready) return
    let live = true
    const code = cache.has(id) ? Promise.resolve() : LOADERS[id]?.().then((e) => void cache.set(id, e))
    void Promise.all([code, ensurePacks(i18n, packs)]).then(() => live && rerender((n) => n + 1))
    return () => {
      live = false
    }
  }, [id, lang, ready, i18n]) // eslint-disable-line react-hooks/exhaustive-deps -- packs follows id
  return ready ? cache.get(id)! : null
}
