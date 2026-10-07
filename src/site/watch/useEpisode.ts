import { useEffect, useState } from 'react'
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

const cache = new Map<string, Episode>()

/** The episode's code once its chunk has loaded (null until then). */
export function useEpisode(id: string): Episode | null {
  const [loaded, setLoaded] = useState<{ id: string; ep: Episode } | null>(null)
  useEffect(() => {
    if (cache.has(id)) return
    let live = true
    void LOADERS[id]?.().then((e) => {
      cache.set(id, e)
      if (live) setLoaded({ id, ep: e })
    })
    return () => {
      live = false
    }
  }, [id])
  return cache.get(id) ?? (loaded?.id === id ? loaded.ep : null)
}
