/* eslint-disable react-refresh/only-export-components -- explainer kit: components plus data and helpers */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { Captions, CaptionsOff, ChevronLeft, ChevronRight, Maximize, Minimize, Pause, Play, RotateCcw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cueAt, fmtClock, locate, stopBetween, timeline } from '../../lib/explainer'
import type { TimedScene } from '../../lib/explainer'
import { C } from './stage'
import { markWatched } from './watched'

export interface Scene extends TimedScene {
  id: string
  /** draws the stage at scene progress p ∈ [0, 1] */
  render: (p: number) => ReactNode
}

export interface Episode {
  id: string
  scenes: Scene[]
}

const SPEEDS = [0.75, 1, 1.25, 1.5, 2]

/** Caption key for scene `s`, line `k` of episode `ep`. */
export const capKey = (ep: string, s: string, k: number) => `watch.${ep}.scenes.${s}.c${k + 1}`

/** The 16:9 stage alone, drawn at absolute time t. Used by the player and for static thumbnails. */
export function Frame({ episode, t, className = '' }: { episode: Episode; t: number; className?: string }) {
  const tl = useMemo(() => timeline(episode.scenes), [episode])
  const { i, p } = locate(tl, episode.scenes, t)
  return (
    <div className={`relative aspect-video w-full select-none overflow-hidden [container-type:inline-size] ${className}`} style={{ background: C.bg }}>
      {episode.scenes[i].render(p)}
    </div>
  )
}

export function Player({ episode, onTime, seekRef }: { episode: Episode; onTime?: (t: number) => void; seekRef?: React.MutableRefObject<((t: number) => void) | null> }) {
  const { t: tr } = useTranslation()
  const ep = episode.id
  const scenes = episode.scenes
  const tl = useMemo(() => timeline(scenes), [scenes])
  // #t=72 in the URL starts the episode at 1:12 (paused)
  const [t, setT] = useState(() => {
    const m = /(?:^#|&)t=(\d+(?:\.\d+)?)/.exec(typeof window === 'undefined' ? '' : window.location.hash)
    return m ? Math.min(timeline(scenes).total, Number(m[1]) * 1000) : 0
  })
  const [playing, setPlaying] = useState(false)
  const [ponder, setPonder] = useState<number | null>(null)
  const [speed, setSpeed] = useState(1)
  const [captions, setCaptions] = useState(true)
  const [full, setFull] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const tRef = useRef(0)
  useEffect(() => {
    tRef.current = t
  }, [t])

  const { i, p } = locate(tl, scenes, t)
  const scene = scenes[i]
  const cue = cueAt(scene.cues, p)
  const ended = t >= tl.total

  useEffect(() => onTime?.(t), [t, onTime])
  const done = t >= tl.total * 0.9
  useEffect(() => {
    if (done) markWatched(ep)
  }, [done, ep])

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(100, now - last) * speed
      last = now
      const from = tRef.current
      const to = Math.min(tl.total, from + dt)
      const stop = stopBetween(tl.stops, from, to)
      if (stop !== null) {
        tRef.current = stop
        setT(stop)
        setPlaying(false)
        setPonder(stop)
        return
      }
      tRef.current = to
      setT(to)
      if (to >= tl.total) {
        setPlaying(false)
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, tl])

  const seek = useCallback(
    (v: number) => {
      setPonder(null)
      setT(Math.max(0, Math.min(tl.total, v)))
    },
    [tl.total],
  )
  useEffect(() => {
    if (seekRef) seekRef.current = (v: number) => {
      seek(v)
      setPlaying(true)
      box.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
    }
  }, [seek, seekRef])

  const toggle = () => {
    setPonder(null)
    if (ended) {
      setT(0)
      setPlaying(true)
    } else setPlaying(!playing)
  }
  const chapter = (d: number) => {
    const j = Math.max(0, Math.min(scenes.length - 1, (d < 0 && p > 0.08 ? i : i + d)))
    seek(tl.starts[j])
  }
  const toggleFull = () => {
    if (document.fullscreenElement) {
      void document.exitFullscreen()
      return
    }
    void box.current
      ?.requestFullscreen?.()
      // on phones, full screen is most useful sideways; browsers that cannot lock simply ignore this
      .then(() => (screen.orientation as ScreenOrientation & { lock?: (o: string) => Promise<void> }).lock?.('landscape'))
      .catch(() => {})
  }
  useEffect(() => {
    const on = () => setFull(document.fullscreenElement === box.current)
    document.addEventListener('fullscreenchange', on)
    return () => document.removeEventListener('fullscreenchange', on)
  }, [])

  const onKey = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
    const k = e.key.toLowerCase()
    if (k === ' ' || k === 'k') toggle()
    else if (k === 'arrowleft') seek(t - 5000)
    else if (k === 'arrowright') seek(t + 5000)
    else if (k === 'j') chapter(-1)
    else if (k === 'l') chapter(1)
    else if (k === 'f') toggleFull()
    else if (k === 'c') setCaptions((c) => !c)
    else return
    e.preventDefault()
  }

  const ponderScene = ponder !== null ? scenes[locate(tl, scenes, ponder - 1).i] : null

  return (
    <div
      ref={box}
      tabIndex={0}
      onKeyDown={onKey}
      role="region"
      aria-label={tr('watch.ui.player', { title: tr(`watch.${ep}.title`) })}
      aria-roledescription={tr('watch.ui.video')}
      className={`group overflow-hidden rounded-2xl bg-[#0b0d12] shadow-2xl shadow-slate-900/30 outline-none ring-blue-400/60 focus-visible:ring-4 ${full ? 'flex h-screen flex-col justify-center rounded-none' : ''}`}
    >
      <div className={`relative mx-auto w-full ${full ? 'max-w-[calc((100vh-150px)*16/9)]' : ''}`}>
        <button type="button" onClick={toggle} className="block w-full cursor-pointer text-left" aria-label={playing ? tr('watch.ui.pause') : tr('watch.ui.play')}>
          <div className="relative aspect-video w-full select-none overflow-hidden [container-type:inline-size]" style={{ background: C.bg }}>
            {scene.render(p)}
          </div>
        </button>
        {!playing && t === 0 && (
          <button type="button" onClick={toggle} className="absolute inset-0 grid place-items-center bg-black/25 transition hover:bg-black/15" aria-label={tr('watch.ui.play')}>
            <span className="grid h-20 w-20 place-items-center rounded-full bg-white/95 text-slate-900 shadow-2xl sm:h-24 sm:w-24">
              <Play className="ml-1 h-9 w-9 fill-current sm:h-11 sm:w-11" />
            </span>
          </button>
        )}
        {ponderScene && (
          <div className="absolute inset-0 grid place-items-center bg-[#0b0d12]/80 p-4 backdrop-blur-[2px]">
            <div className="max-w-xl text-center">
              <p className="text-xs font-black uppercase tracking-[0.2em] text-amber-300 sm:text-sm">{tr('watch.ui.ponder')}</p>
              <p className="mt-2 text-base font-bold leading-snug text-white sm:mt-3 sm:text-2xl">{tr(`watch.${ep}.scenes.${ponderScene.id}.ponder`)}</p>
              <button type="button" onClick={() => { setPonder(null); setPlaying(true) }} className="mt-4 rounded-full bg-amber-300 px-5 py-2 text-sm font-black text-slate-900 hover:bg-amber-200 sm:mt-6 sm:text-base">
                {tr('watch.ui.continue')} →
              </button>
            </div>
          </div>
        )}
      </div>

      {captions && (
        <div className="flex min-h-[4.5rem] items-center justify-center border-t border-white/5 px-4 py-3 text-center sm:min-h-[5.25rem] sm:px-10" aria-live="polite">
          <p className="max-w-4xl text-[15px] font-medium leading-snug text-slate-100 sm:text-lg">{tr(capKey(ep, scene.id, cue))}</p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-white/5 px-3 py-2.5 text-slate-300 sm:px-4">
        <button type="button" onClick={toggle} className="grid h-9 w-9 place-items-center rounded-full bg-white text-slate-900 hover:bg-blue-100" aria-label={ended ? tr('watch.ui.replay') : playing ? tr('watch.ui.pause') : tr('watch.ui.play')}>
          {ended ? <RotateCcw className="h-4 w-4" /> : playing ? <Pause className="h-4 w-4 fill-current" /> : <Play className="ml-0.5 h-4 w-4 fill-current" />}
        </button>
        <button type="button" onClick={() => chapter(-1)} className="grid h-8 w-8 place-items-center rounded-full hover:bg-white/10" aria-label={tr('watch.ui.prev_chapter')}>
          <ChevronLeft className="h-4 w-4" />
        </button>
        <button type="button" onClick={() => chapter(1)} className="grid h-8 w-8 place-items-center rounded-full hover:bg-white/10" aria-label={tr('watch.ui.next_chapter')}>
          <ChevronRight className="h-4 w-4" />
        </button>
        <span className="font-mono text-xs tabular-nums text-slate-400">
          {fmtClock(t)} / {fmtClock(tl.total)}
        </span>
        <span className="hidden min-w-0 flex-1 truncate text-xs font-bold text-slate-300 sm:block">{tr(`watch.${ep}.scenes.${scene.id}.title`)}</span>
        <div className="ml-auto flex items-center gap-1">
          <label className="sr-only" htmlFor={`speed-${ep}`}>{tr('watch.ui.speed')}</label>
          <select id={`speed-${ep}`} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="rounded-md bg-white/10 px-1.5 py-1 font-mono text-xs text-slate-200 outline-none">
            {SPEEDS.map((s) => <option key={s} value={s} className="text-slate-900">{s}×</option>)}
          </select>
          <button type="button" onClick={() => setCaptions(!captions)} aria-pressed={captions} className="grid h-8 w-8 place-items-center rounded-full hover:bg-white/10" aria-label={tr('watch.ui.captions')}>
            {captions ? <Captions className="h-4 w-4" /> : <CaptionsOff className="h-4 w-4" />}
          </button>
          <button type="button" onClick={toggleFull} className="grid h-8 w-8 place-items-center rounded-full hover:bg-white/10" aria-label={tr('watch.ui.fullscreen')}>
            {full ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
          </button>
        </div>
        <div className="relative order-last h-5 w-full">
          <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-white/10" />
          <div className="absolute left-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-blue-400" style={{ width: `${(t / tl.total) * 100}%` }} />
          {tl.starts.slice(1).map((s) => <span key={s} className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-[#0b0d12]" style={{ left: `${(s / tl.total) * 100}%` }} />)}
          {tl.stops.map((s) => <span key={'p' + s} className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-amber-300" style={{ left: `${(s / tl.total) * 100}%` }} />)}
          <input
            type="range"
            min={0}
            max={tl.total}
            step={100}
            value={t}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label={tr('watch.ui.seek')}
            aria-valuetext={`${fmtClock(t)} — ${tr(`watch.${ep}.scenes.${scene.id}.title`)}`}
            className="absolute inset-0 w-full cursor-pointer opacity-0"
          />
        </div>
      </div>
    </div>
  )
}
