import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, ImageUp, LoaderCircle, Pause, Play } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { PyBlock } from '../../components/PyCode'
import { RunInPlayground } from '../../components/RunInPlayground'
import { INPUT_SIZE, KERAS_CODE, channelPixels, layerRange, loadNet, ranked, runCnn, toInput } from '../../lib/cnnLive'
import type { LiveLayer } from '../../lib/cnnLive'
import { ImageInputError, readUploadSquare } from '../../lib/imageInput'
import meta from '../../data/cnnLive/meta.json'
import samplesUrl from '../../data/cnnLive/samples.png?url'
import weightsUrl from '../../data/cnnLive/weights.bin?url'
import { fmt } from '../format'
import { Card, LabPage, Note, Stat } from '../ui'

const PER_ROW = 10
const TICK_MS = 3000

/** one picture of the sprite as RGBA */
const sampleRgba = (sprite: ImageData, index: number): Uint8ClampedArray => {
  const out = new Uint8ClampedArray(INPUT_SIZE * INPUT_SIZE * 4)
  const left = (index % PER_ROW) * INPUT_SIZE
  const top = Math.floor(index / PER_ROW) * INPUT_SIZE
  for (let y = 0; y < INPUT_SIZE; y++) {
    const from = ((top + y) * sprite.width + left) * 4
    out.set(sprite.data.subarray(from, from + INPUT_SIZE * 4), y * INPUT_SIZE * 4)
  }
  return out
}

const layerNames = (layers: LiveLayer[]): string[] => {
  let conv = 0
  let pool = 0
  return layers.map((l) => {
    if (l.kind === 'conv') return `conv${++conv}`
    if (l.kind === 'pool') return `pool${++pool}`
    return l.kind
  })
}

/** pixels on a canvas, enlarged by CSS without smoothing */
function Pixels({ pixels, size, className = '', label }: { pixels: Uint8ClampedArray; size: number; className?: string; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (ctx) ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels), size, size), 0, 0)
  }, [pixels, size])
  return <canvas ref={ref} width={size} height={size} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true} className={`block w-full [image-rendering:pixelated] ${className}`} />
}

export default function CnnLive() {
  const { t } = useTranslation()
  const [net, setNet] = useState<ReturnType<typeof loadNet> | null>(null)
  const [sprite, setSprite] = useState<ImageData | null>(null)
  const [spriteUrl, setSpriteUrl] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [index, setIndex] = useState(() => Math.floor(Math.random() * meta.labels.length))
  const [own, setOwn] = useState<Uint8ClampedArray | null>(null)
  const [playing, setPlaying] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [picked, setPicked] = useState<{ layer: number; channel: number }>({ layer: 2, channel: 0 })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let live = true
    Promise.all([
      fetch(weightsUrl).then((r) => {
        if (!r.ok) throw new Error('weights')
        return r.arrayBuffer()
      }),
      new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image()
        img.onload = () => resolve(img)
        img.onerror = reject
        img.src = samplesUrl
      }),
    ]).then(([buf, img]) => {
      if (!live) return
      const c = document.createElement('canvas')
      c.width = img.width
      c.height = img.height
      const ctx = c.getContext('2d', { willReadFrequently: true })!
      ctx.drawImage(img, 0, 0)
      setSprite(ctx.getImageData(0, 0, img.width, img.height))
      setSpriteUrl(samplesUrl)
      setNet(loadNet(new Float32Array(buf)))
    }).catch(() => {
      if (live) setFailed(true)
    })
    return () => {
      live = false
    }
  }, [])

  // a new picture every few seconds, as long as the page is in view
  useEffect(() => {
    if (!playing || own) return
    const id = window.setInterval(() => {
      if (!document.hidden) setIndex((i) => (i + 1) % meta.labels.length)
    }, TICK_MS)
    return () => window.clearInterval(id)
  }, [playing, own])

  const rgba = useMemo(() => own ?? (sprite ? sampleRgba(sprite, index) : null), [own, sprite, index])
  const run = useMemo(() => {
    if (!net || !rgba) return null
    const t0 = performance.now()
    const r = runCnn(net, toInput(rgba))
    return { ...r, ms: performance.now() - t0 }
  }, [net, rgba])
  const maps = useMemo(() => run?.layers.slice(1).map((l) => Array.from({ length: l.depth }, (_, c) => channelPixels(l, c))) ?? [], [run])
  const names = useMemo(() => (run ? layerNames(run.layers) : []), [run])

  const choose = useCallback((i: number) => {
    setOwn(null)
    setError(null)
    setPlaying(false)
    setIndex((i + meta.labels.length) % meta.labels.length)
  }, [])

  const onFile = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      setOwn(await readUploadSquare(file, INPUT_SIZE))
      setPlaying(false)
    } catch (e) {
      setError(t(`lab.live.err_${e instanceof ImageInputError ? e.code : 'decode'}`))
    } finally {
      setBusy(false)
    }
  }

  const top = run ? ranked(run.probs).slice(0, 5) : []
  const truth = own ? null : meta.labels[index]
  const right = truth !== null && top[0]?.k === truth
  const className = (k: number) => t(`lab.live.class.${meta.classes[k]}`)

  // ---- the layer and channel looked at closely
  const chosenLayer = run?.layers[picked.layer]
  const closeUp = useMemo(() => {
    if (!chosenLayer) return null
    const n = chosenLayer.size * chosenLayer.size
    const from = picked.channel * n
    const vals = chosenLayer.data.subarray(from, from + n)
    let min = Infinity
    let max = -Infinity
    let zeros = 0
    for (const v of vals) {
      if (v < min) min = v
      if (v > max) max = v
      if (v === 0) zeros++
    }
    return { min, max, zeros: zeros / n, layer: layerRange(chosenLayer) }
  }, [chosenLayer, picked.channel])

  const inputBox = (
    <div className="grid gap-1.5">
      {rgba ? <Pixels pixels={rgba} size={INPUT_SIZE} className="rounded-md" label={own ? t('lab.live.own_picture') : t('lab.live.sample_aria', { n: index + 1, c: className(meta.labels[index]) })} /> : <div className="aspect-square rounded-md bg-slate-100" />}
      <p className="text-center text-[11px] leading-snug text-slate-500">
        {truth === null ? t('lab.live.own_picture') : <>{t('lab.live.truth')} <b className="text-slate-800">{className(truth)}</b></>}
      </p>
    </div>
  )

  const bars = (
    <ol className="grid gap-1.5" aria-label={t('lab.live.top5')}>
      {top.map(({ k, p }, i) => (
        <li key={k} className="relative overflow-hidden rounded-md bg-slate-100 px-2 py-1.5 text-[12.5px]">
          <span aria-hidden className={`absolute inset-y-0 left-0 rounded-md transition-[width] duration-200 ${k === truth ? 'bg-emerald-300' : i === 0 && truth !== null ? 'bg-rose-200' : 'bg-blue-200'}`} style={{ width: `${Math.max(1, p * 100)}%` }} />
          <span className="relative flex justify-between gap-2 font-semibold text-slate-800">
            <span className="truncate">{className(k)}</span>
            <span className="font-mono text-[11px] text-slate-600">{fmt(p * 100, 1)}%</span>
          </span>
        </li>
      ))}
    </ol>
  )

  return (
    <LabPage id="cnn-live">
      <div className="grid gap-5">
        <Card
          title={t('lab.live.net_title')}
          sub={run ? t('lab.live.net_sub', { ms: fmt(run.ms, 1), params: meta.params.toLocaleString('en-US') }) : undefined}
        >
          {failed && <Note tone="warn">{t('lab.live.load_failed')}</Note>}
          {!failed && !run && <p className="flex items-center gap-2 py-16 text-sm text-slate-500"><LoaderCircle size={16} className="animate-spin" aria-hidden />{t('lab.live.loading')}</p>}
          {run && (
            <div className="grid items-center gap-4 lg:grid-cols-[108px_minmax(0,1fr)_168px]">
              <div className="grid grid-cols-[96px_minmax(0,1fr)] items-center gap-3 lg:contents">
                {inputBox}
                <div className="lg:hidden">{bars}</div>
              </div>
              <div className="min-w-0 overflow-x-auto pb-1" tabIndex={0} role="group" aria-label={t('lab.live.grid_aria')}>
                <div className="grid min-w-[680px] gap-x-[3px] gap-y-[3px]" style={{ gridTemplateColumns: `repeat(${maps.length}, minmax(0, 1fr))` }}>
                  {maps.map((_, col) => {
                    const l = run.layers[col + 1]
                    return (
                      <div key={col} className={`pb-1 text-center font-mono text-[9.5px] leading-tight ${l.kind === 'pool' ? 'text-amber-700' : l.kind === 'fc' ? 'text-emerald-700' : l.kind === 'conv' ? 'text-blue-700' : 'text-slate-500'}`}>
                        <div className="font-bold">{names[col + 1]}</div>
                        <div>{l.kind === 'fc' ? '10' : `${l.size}²`}</div>
                      </div>
                    )
                  })}
                  {Array.from({ length: 10 }, (_, channel) =>
                    maps.map((layerMaps, col) => {
                      const l = run.layers[col + 1]
                      const on = picked.layer === col + 1 && picked.channel === channel
                      return (
                        <button
                          key={`${col}-${channel}`}
                          type="button"
                          onClick={() => setPicked({ layer: col + 1, channel })}
                          aria-pressed={on}
                          aria-label={t('lab.live.cell_aria', { layer: names[col + 1], channel: channel + 1 })}
                          className={`block overflow-hidden rounded-[2px] outline-offset-1 ring-1 ring-slate-200 focus-visible:outline-2 focus-visible:outline-blue-500 ${on ? 'outline-2 outline-blue-500' : ''}`}
                        >
                          <Pixels pixels={layerMaps[channel]} size={l.size} />
                        </button>
                      )
                    }),
                  )}
                </div>
              </div>
              <div className="hidden lg:block">{bars}</div>
            </div>
          )}
          {run && (
            <p aria-live="polite" className={`mt-3 text-[13px] font-semibold ${truth === null ? 'text-slate-600' : right ? 'text-emerald-700' : 'text-rose-700'}`}>
              {truth === null
                ? t('lab.live.verdict_own', { c: className(top[0].k), p: fmt(top[0].p * 100, 1) })
                : right
                  ? t('lab.live.verdict_right', { c: className(top[0].k), p: fmt(top[0].p * 100, 1) })
                  : t('lab.live.verdict_wrong', { c: className(top[0].k), p: fmt(top[0].p * 100, 1), truth: className(truth) })}
            </p>
          )}
        </Card>

        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
          <Card
            title={t('lab.live.pick_title')}
            sub={t('lab.live.pick_sub', { n: meta.labels.length })}
            right={
              <div className="flex items-center gap-1">
                <button type="button" onClick={() => choose(index - 1)} aria-label={t('lab.live.prev')} className="grid h-8 w-8 place-items-center rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-50"><ChevronLeft size={16} aria-hidden /></button>
                <button type="button" onClick={() => { setOwn(null); setPlaying((p) => !p) }} aria-pressed={playing && !own} aria-label={playing && !own ? t('lab.live.pause') : t('lab.live.play')} className="grid h-8 w-8 place-items-center rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-50">
                  {playing && !own ? <Pause size={15} aria-hidden /> : <Play size={15} aria-hidden />}
                </button>
                <button type="button" onClick={() => choose(index + 1)} aria-label={t('lab.live.next')} className="grid h-8 w-8 place-items-center rounded-lg border border-slate-300 text-slate-600 hover:bg-slate-50"><ChevronRight size={16} aria-hidden /></button>
              </div>
            }
          >
            {spriteUrl && (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(40px,1fr))] gap-1">
                {meta.labels.map((label, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => choose(i)}
                    aria-pressed={!own && index === i}
                    aria-label={t('lab.live.sample_aria', { n: i + 1, c: className(label) })}
                    title={className(label)}
                    className={`aspect-square rounded-[4px] [image-rendering:pixelated] outline-offset-1 focus-visible:outline-2 focus-visible:outline-blue-500 ${!own && index === i ? 'outline-2 outline-blue-600' : ''}`}
                    style={{
                      backgroundImage: `url(${spriteUrl})`,
                      backgroundSize: `${PER_ROW * 100}% ${(meta.labels.length / PER_ROW) * 100}%`,
                      backgroundPosition: `${((i % PER_ROW) / (PER_ROW - 1)) * 100}% ${(Math.floor(i / PER_ROW) / (meta.labels.length / PER_ROW - 1)) * 100}%`,
                    }}
                  />
                ))}
              </div>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => fileInput.current?.click()} disabled={busy} className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-[13px] font-semibold text-slate-700 hover:bg-slate-50">
                {busy ? <LoaderCircle size={14} className="animate-spin" aria-hidden /> : <ImageUp size={14} aria-hidden />}
                {t('lab.live.upload')}
              </button>
              <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp" className="hidden" onChange={(e) => { void onFile(e.target.files?.[0]); e.target.value = '' }} />
              <p className="min-w-[200px] flex-1 text-[11.5px] leading-snug text-slate-500">{t('lab.live.upload_note')}</p>
            </div>
            {error && <p role="alert" className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700">{error}</p>}
          </Card>

          <Card title={t('lab.live.close_title', { layer: names[picked.layer] ?? '', channel: picked.channel + 1 })}>
            {chosenLayer && closeUp ? (
              <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-4">
                <Pixels pixels={maps[picked.layer - 1][picked.channel]} size={chosenLayer.size} className="rounded-md" />
                <div className="grid content-start gap-1.5 text-[13px] leading-relaxed text-slate-700">
                  <p>{t(`lab.live.kind.${chosenLayer.kind}`)}</p>
                  <p className="font-mono text-[11.5px] text-slate-500">
                    {chosenLayer.kind === 'fc' ? t('lab.live.close_fc', { c: className(picked.channel), v: fmt(chosenLayer.data[picked.channel], 2) }) : t('lab.live.close_stats', { n: chosenLayer.size, min: fmt(closeUp.min, 2), max: fmt(closeUp.max, 2), zeros: fmt(closeUp.zeros * 100, 0) })}
                  </p>
                </div>
              </div>
            ) : <p className="text-sm text-slate-500">{t('lab.live.loading')}</p>}
          </Card>
        </div>

        <div className="grid items-start gap-5 lg:grid-cols-2">
          <Card title={t('lab.live.read_title')}>
            <ul className="grid gap-2 text-[13.5px] leading-relaxed text-slate-700">
              {(['conv', 'relu', 'pool', 'deep', 'fc'] as const).map((k) => <li key={k}><Trans i18nKey={`lab.live.read.${k}`} components={{ b: <b className="font-extrabold text-slate-900" /> }} /></li>)}
            </ul>
          </Card>
          <Card title={t('lab.live.score_title')} sub={t('lab.live.score_sub')}>
            <div className="grid grid-cols-2 gap-2">
              <Stat k={t('lab.live.score_acc')} v={`${fmt(meta.testAccuracy * 100, 1)}%`} d={t('lab.live.score_chance')} tone="blue" />
              <Stat k={t('lab.live.score_params')} v={meta.params.toLocaleString('en-US')} d={t('lab.live.score_params_d')} />
            </div>
            <ul className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(meta.perClass).sort((a, b) => b[1] - a[1]).map(([c, v]) => (
                <li key={c} className="grid grid-cols-[minmax(0,1fr)_60px_34px] items-center gap-1.5 text-[11.5px] text-slate-600">
                  <span className="truncate">{t(`lab.live.class.${c}`)}</span>
                  <span className="h-1.5 overflow-hidden rounded bg-slate-100"><i className="block h-full rounded bg-blue-500" style={{ width: `${v * 100}%` }} /></span>
                  <span className="text-right font-mono">{fmt(v * 100, 0)}%</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <Card title={t('lab.live.code_title')} sub={t('lab.live.code_sub')} right={<RunInPlayground lib="keras" code={KERAS_CODE} />}>
          <PyBlock tone="light" code={KERAS_CODE} />
        </Card>

        <p className="text-[11.5px] leading-relaxed text-slate-500">
          <Trans
            i18nKey="lab.live.credit"
            components={{
              a: <a className="underline" href="https://cs231n.stanford.edu/" target="_blank" rel="noopener noreferrer" />,
              b: <a className="underline" href="https://cs.stanford.edu/people/karpathy/convnetjs/" target="_blank" rel="noopener noreferrer" />,
              c: <a className="underline" href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank" rel="noopener noreferrer" />,
            }}
          />
        </p>
      </div>
    </LabPage>
  )
}
