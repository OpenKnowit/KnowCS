// /mnist/: train a small CNN on MNIST in the browser, after ConvNetJS's MNIST demo, with the same network in PyTorch.
// The training runs in a worker (mnistWorker.ts); this page loads the images, sends the settings and draws what comes back.
import { useEffect, useMemo, useRef, useState } from 'react'
import { LoaderCircle, Pause, Play, RotateCcw } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { PyBlock } from '../../components/PyCode'
import { CLASSES, DEFAULTS, IMAGE, SHOWN, makeNet, paramCount, pytorchCode, unpackSprite } from '../../lib/mnistNet'
import type { OptName, TrainConfig } from '../../lib/mnistNet'
import labelsUrl from '../../data/mnist/labels.bin?url'
import testUrl from '../../data/mnist/test.webp?url'
import { moduleUrl } from '../../lib/sitemap'
import { fmt } from '../format'
import { Btn, Card, LabPage, Note, Seg, Stat, Workspace } from '../ui'
import type { Evaluation, FromWorker, Point, Preds, Stats, ToWorker, Vis } from './mnistProtocol'

const TRAIN_URLS = Object.entries(import.meta.glob('../../data/mnist/train-*.webp', { eager: true, query: '?url', import: 'default' }) as Record<string, string>)
  .sort(([a], [b]) => Number(/(\d+)\.webp$/.exec(a)![1]) - Number(/(\d+)\.webp$/.exec(b)![1]))
  .map(([, url]) => url)
const PER_BATCH = 3000
const N_TRAIN = TRAIN_URLS.length * PER_BATCH
const N_SHOWN = 40
const PARAMS = paramCount(makeNet())

const LR: Record<OptName, number[]> = { adadelta: [0.1, 0.5, 1, 2], sgd: [0.001, 0.01, 0.05, 0.1], adam: [0.0003, 0.001, 0.003, 0.01] }
const pct = (v: number | null) => (v === null ? '—' : `${(v * 100).toFixed(1)}%`)

/** a lossless sprite of 60 digits per row → 784 grey bytes per digit */
function decode(url: string, count: number): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = img.width
      c.height = img.height
      const ctx = c.getContext('2d', { willReadFrequently: true })!
      ctx.drawImage(img, 0, 0)
      resolve(unpackSprite(ctx.getImageData(0, 0, img.width, img.height).data, img.width, count))
    }
    img.onerror = reject
    img.src = url
  })
}

// ---------------------------------------------------------------- drawing

type Ramp = 'grey' | 'act' | 'signed'
function paint(values: ArrayLike<number>, side: number, ramp: Ramp): ImageData {
  const out = new ImageData(side, side)
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < values.length; i++) {
    lo = Math.min(lo, values[i])
    hi = Math.max(hi, values[i])
  }
  const span = hi - lo || 1
  const big = Math.max(Math.abs(lo), Math.abs(hi)) || 1
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    let rgb: [number, number, number]
    if (ramp === 'signed') {
      // blue below 0, white at 0, red above
      const s = v / big
      rgb = s < 0 ? [255 + s * 218, 255 + s * 156, 255 + s * 20] : [255 - s * 30, 255 - s * 226, 255 - s * 183]
    } else {
      const s = (v - lo) / span
      // dark slate to white, or (activations) slate to blue to white
      rgb = ramp === 'grey' ? [15 + s * 240, 23 + s * 232, 42 + s * 213] : s < 0.5 ? [15 + s * 2 * 22, 23 + s * 2 * 76, 42 + s * 2 * 193] : [37 + (s - 0.5) * 2 * 218, 99 + (s - 0.5) * 2 * 156, 235 + (s - 0.5) * 2 * 20]
    }
    out.data.set([rgb[0], rgb[1], rgb[2], 255], i * 4)
  }
  return out
}

function HeatMap({ values, side, ramp, className = '', label }: { values: ArrayLike<number>; side: number; ramp: Ramp; className?: string; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    ref.current?.getContext('2d')?.putImageData(paint(values, side, ramp), 0, 0)
  }, [values, side, ramp])
  return <canvas ref={ref} width={side} height={side} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true} className={`block aspect-square w-full rounded-[3px] [image-rendering:pixelated] ${className}`} />
}

/** one 28 × 28 test digit */
const digit = (images: Uint8Array, i: number) => images.subarray(i * IMAGE * IMAGE, (i + 1) * IMAGE * IMAGE)

function Chart({ points, kind, label }: { points: Point[]; kind: 'loss' | 'acc'; label: string }) {
  const W = 420, H = 170, L = 40, R = 10, T = 10, B = 26
  const xMax = Math.max(points.at(-1)?.seen ?? 0, 1000)
  const yMax = kind === 'loss' ? Math.max(0.5, Math.ceil(Math.max(...points.map((p) => p.loss), 0) * 2) / 2) : 1
  const X = (v: number) => L + (v / xMax) * (W - L - R)
  const Y = (v: number) => T + (1 - v / yMax) * (H - T - B)
  const line = (f: (p: Point) => number) => points.map((p, i) => `${i ? 'L' : 'M'}${X(p.seen).toFixed(1)},${Y(f(p)).toFixed(1)}`).join('')
  const yTicks = kind === 'loss' ? [0, yMax / 2, yMax] : [0, 0.5, 0.9, 1]
  const xTicks = [0, xMax / 2, xMax]
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke={v === 0.9 && kind === 'acc' ? '#cbd5e1' : '#eef2f7'} strokeDasharray={v === 0.9 && kind === 'acc' ? '3 3' : undefined} />
          <text x={L - 6} y={Y(v) + 4} textAnchor="end" className="fill-slate-400 text-[10px]">{kind === 'loss' ? fmt(v, 2) : `${v * 100}%`}</text>
        </g>
      ))}
      {xTicks.map((v) => <text key={v} x={X(v)} y={H - 8} textAnchor={v ? (v === xMax ? 'end' : 'middle') : 'start'} className="fill-slate-400 text-[10px]">{Math.round(v).toLocaleString('en-US')}</text>)}
      {kind === 'loss' ? (
        <path d={line((p) => p.loss)} fill="none" stroke="#2563eb" strokeWidth={2} />
      ) : (
        <>
          <path d={line((p) => p.train)} fill="none" stroke="#2563eb" strokeWidth={2} />
          <path d={line((p) => p.val)} fill="none" stroke="#059669" strokeWidth={2} />
        </>
      )}
    </svg>
  )
}

// ---------------------------------------------------------------- the page

export default function MnistTrain() {
  const { t } = useTranslation()
  const worker = useRef<Worker | null>(null)
  const [cfg, setCfg] = useState<TrainConfig>({ optimizer: 'adadelta', lr: 1, momentum: 0, weightDecay: 0, batchSize: 20 })
  const [stats, setStats] = useState<Stats | null>(null)
  const [vis, setVis] = useState<Vis | null>(null)
  const [preds, setPreds] = useState<Preds | null>(null)
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const [loaded, setLoaded] = useState(0)
  const [failed, setFailed] = useState(false)
  const [test, setTest] = useState<{ images: Uint8Array; labels: Uint8Array; shown: number[] } | null>(null)
  // started by the reader; paused while the tab is hidden and resumed when it is back
  const [wanted, setWanted] = useState(false)
  const [hidden, setHidden] = useState(() => document.hidden)
  const send = (m: ToWorker, transfer: Transferable[] = []) => worker.current?.postMessage(m, transfer)

  useEffect(() => {
    const w = new Worker(new URL('./mnistWorker.ts', import.meta.url), { type: 'module' })
    worker.current = w
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data
      if (m.type === 'stats') setStats(m)
      else if (m.type === 'vis') setVis(m)
      else if (m.type === 'preds') setPreds(m)
      else {
        setEvaluation(m)
        setEvaluating(false)
      }
    }
    let live = true
    ;(async () => {
      const labels = new Uint8Array(await (await fetch(labelsUrl)).arrayBuffer())
      // the test images first (the page shows some), then the training batches one after another
      const images = await decode(testUrl, PER_BATCH)
      if (!live) return
      const testLabels = labels.slice(N_TRAIN)
      const shown = Array.from({ length: N_SHOWN }, (_, k) => Math.floor((k + Math.random()) * (PER_BATCH / N_SHOWN)))
      setTest({ images, labels: testLabels, shown })
      w.postMessage({ type: 'test', images: images.slice(), labels: testLabels, shown } satisfies ToWorker)
      for (let b = 0; b < TRAIN_URLS.length; b++) {
        const batch = await decode(TRAIN_URLS[b], PER_BATCH)
        if (!live) return
        w.postMessage({ type: 'train-batch', batch: b, images: batch, labels: labels.slice(b * PER_BATCH, (b + 1) * PER_BATCH) } satisfies ToWorker, [batch.buffer])
        setLoaded(b + 1)
      }
    })().catch(() => {
      if (live) setFailed(true)
    })
    const onVisibility = () => setHidden(document.hidden)
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      live = false
      document.removeEventListener('visibilitychange', onVisibility)
      w.terminate()
    }
  }, [])

  const running = wanted && !hidden && loaded > 0
  useEffect(() => {
    worker.current?.postMessage({ type: 'run', on: running } satisfies ToWorker)
  }, [running])

  const configure = (next: Partial<TrainConfig>) => {
    const c = { ...cfg, ...next }
    setCfg(c)
    send({ type: 'config', cfg: c })
  }
  const pickOptimizer = (o: OptName) => configure({ optimizer: o, lr: DEFAULTS[o].lr, momentum: DEFAULTS[o].momentum })
  const reset = () => {
    setEvaluation(null)
    setVis(null)
    send({ type: 'reset' })
  }
  const evaluate = () => {
    setEvaluating(true)
    send({ type: 'evaluate' })
  }

  const code = useMemo(() => pytorchCode(cfg), [cfg])
  const seen = stats?.seen ?? 0
  const trainSize = stats?.trainSize ?? 0
  const layerName = (key: string) => t(`lab.mnist.layer.${key}`)

  const controls = (
    <>
      <Card title={t('lab.mnist.run_title')}>
        <div className="grid gap-3">
          {failed ? (
            <Note tone="bad">{t('lab.mnist.load_failed')}</Note>
          ) : (
            <div className="grid gap-1.5">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-500">
                <span className="flex items-center gap-1.5">{loaded < TRAIN_URLS.length && <LoaderCircle size={13} className="animate-spin" aria-hidden />}{t('lab.mnist.loaded', { n: (loaded * PER_BATCH).toLocaleString('en-US'), total: N_TRAIN.toLocaleString('en-US') })}</span>
                <span className="font-mono">{Math.round((loaded / TRAIN_URLS.length) * 100)}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-blue-500 transition-[width]" style={{ width: `${(loaded / TRAIN_URLS.length) * 100}%` }} /></div>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <button type="button" disabled={!loaded} onClick={() => setWanted((w) => !w)} className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-bold text-white shadow-lg shadow-blue-600/20 transition hover:bg-blue-700 disabled:opacity-40">
              {wanted ? <Pause size={16} aria-hidden /> : <Play size={16} className="fill-white" aria-hidden />}
              {wanted ? t('lab.mnist.pause') : seen ? t('lab.mnist.resume') : t('lab.mnist.start')}
            </button>
            <button type="button" onClick={reset} className="flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-3.5 py-2.5 text-sm font-semibold text-slate-700 hover:border-slate-400">
              <RotateCcw size={15} aria-hidden /> {t('lab.mnist.reset')}
            </button>
          </div>
          {wanted && hidden && <p className="text-xs text-slate-500">{t('lab.mnist.hidden')}</p>}
          <p className="text-xs text-slate-500">{t('lab.mnist.params', { n: PARAMS.toLocaleString('en-US') })}</p>
        </div>
      </Card>
      <Card title={t('lab.mnist.hyper_title')} sub={t('lab.mnist.hyper_sub')}>
        <div className="grid gap-3">
          <Seg label={t('lab.mnist.optimizer')} value={cfg.optimizer} onChange={pickOptimizer} options={[{ v: 'adadelta', label: 'Adadelta' }, { v: 'sgd', label: 'SGD' }, { v: 'adam', label: 'Adam' }]} />
          <Seg label={t('lab.mnist.lr')} value={cfg.lr} onChange={(lr) => configure({ lr })} options={LR[cfg.optimizer].map((v) => ({ v, label: String(v) }))} />
          {cfg.optimizer === 'sgd' && <Seg label={t('lab.mnist.momentum')} value={cfg.momentum} onChange={(momentum) => configure({ momentum })} options={[0, 0.5, 0.9].map((v) => ({ v, label: String(v) }))} />}
          <Seg label={t('lab.mnist.batch')} value={cfg.batchSize} onChange={(batchSize) => configure({ batchSize })} options={[1, 10, 20, 50, 100].map((v) => ({ v, label: String(v) }))} />
          <Seg label={t('lab.mnist.wd')} value={cfg.weightDecay} onChange={(weightDecay) => configure({ weightDecay })} options={[0, 1e-4, 1e-3].map((v) => ({ v, label: v ? v.toExponential() : '0' }))} />
        </div>
      </Card>
      <Note title={t('lab.mnist.about_title')}>
        <Trans i18nKey="lab.mnist.about" components={{ a: <a className="font-bold text-blue-700 hover:underline" href="https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html" target="_blank" rel="noreferrer" /> }} />
      </Note>
    </>
  )

  return (
    <LabPage id="mnist">
      <Workspace wide controls={controls}>
        <Card step={1} title={t('lab.mnist.progress_title')} sub={t('lab.mnist.progress_sub')}>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
            <Stat k={t('lab.mnist.seen')} v={seen.toLocaleString('en-US')} d={trainSize ? t('lab.mnist.epochs', { n: fmt(seen / trainSize, 2) }) : undefined} />
            <Stat k={t('lab.mnist.loss')} v={stats?.loss == null ? '—' : fmt(stats.loss, 3)} />
            <Stat k={t('lab.mnist.train_acc')} v={pct(stats?.trainAcc ?? null)} />
            <Stat k={t('lab.mnist.val_acc')} v={pct(stats?.valAcc ?? null)} tone={stats?.valAcc != null && stats.valAcc >= 0.95 ? 'good' : 'none'} />
            <Stat k={t('lab.mnist.speed')} v={stats?.perSec ? t('lab.mnist.per_sec', { n: Math.round(stats.perSec) }) : '—'} />
            <Stat k={t('lab.mnist.time')} v={stats?.fwdMs == null ? '—' : `${fmt(stats.fwdMs + (stats.bwdMs ?? 0), 2)} ms`} d={stats?.fwdMs == null ? undefined : t('lab.mnist.time_d', { f: fmt(stats.fwdMs, 2), b: fmt(stats.bwdMs ?? 0, 2) })} />
          </div>
          {stats && stats.points.length > 1 ? (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <figure className="grid gap-1">
                <figcaption className="text-xs font-bold text-slate-500">{t('lab.mnist.chart_loss')}</figcaption>
                <Chart points={stats.points} kind="loss" label={t('lab.mnist.chart_loss')} />
              </figure>
              <figure className="grid gap-1">
                <figcaption className="flex flex-wrap items-center gap-3 text-xs font-bold text-slate-500">
                  {t('lab.mnist.chart_acc')}
                  <span className="flex items-center gap-1 font-semibold"><i className="h-0.5 w-4 bg-blue-600" />{t('lab.mnist.train_acc')}</span>
                  <span className="flex items-center gap-1 font-semibold"><i className="h-0.5 w-4 bg-emerald-600" />{t('lab.mnist.val_acc')}</span>
                </figcaption>
                <Chart points={stats.points} kind="acc" label={t('lab.mnist.chart_acc')} />
              </figure>
            </div>
          ) : (
            <p className="mt-4 rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-500">{t('lab.mnist.chart_empty')}</p>
          )}
        </Card>

        <Card step={2} title={t('lab.mnist.net_title')} sub={vis && vis.label >= 0 ? t('lab.mnist.net_sub', { y: vis.label }) : undefined}>
          {vis && vis.layers.length ? (
            <div className="grid gap-4">
              {SHOWN.map((l, k) => (
                <div key={l.key} className="grid gap-1.5 sm:grid-cols-[150px_minmax(0,1fr)] sm:items-start sm:gap-3">
                  <div className="text-[12px] leading-snug">
                    <div className="font-bold text-slate-800">{layerName(l.key)}</div>
                    <div className="font-mono text-slate-500">({l.depth}, {l.side}, {l.side})</div>
                  </div>
                  <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${k === 0 ? 1 : l.depth}, minmax(0, ${k === 0 ? '96px' : l.side >= 24 ? '72px' : l.side >= 12 ? '48px' : '32px'}))` }}>
                    {vis.layers[k].map((ch, c) => <HeatMap key={c} values={ch} side={l.side} ramp={k === 0 ? 'grey' : 'act'} />)}
                  </div>
                </div>
              ))}
              <div className="grid gap-1.5 sm:grid-cols-[150px_minmax(0,1fr)] sm:items-start sm:gap-3">
                <div className="text-[12px] font-bold leading-snug text-slate-800">{layerName('out')}</div>
                <div className="grid gap-1">
                  {vis.probs.map((p, c) => (
                    <div key={c} className="grid grid-cols-[18px_minmax(0,1fr)_48px] items-center gap-2 font-mono text-[12px]">
                      <span className={`font-bold ${c === vis.label ? 'text-emerald-700' : 'text-slate-600'}`}>{c}</span>
                      <span className="h-2.5 overflow-hidden rounded-full bg-slate-100"><span className={`block h-full rounded-full ${c === vis.label ? 'bg-emerald-500' : 'bg-blue-400'}`} style={{ width: `${p * 100}%` }} /></span>
                      <span className="text-right text-slate-500">{(p * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="grid gap-1.5 border-t border-slate-100 pt-3 sm:grid-cols-[150px_minmax(0,1fr)] sm:items-start sm:gap-3">
                <div className="text-[12px] leading-snug">
                  <div className="font-bold text-slate-800">{layerName('kernels')}</div>
                  <div className="text-slate-500">{t('lab.mnist.kernels_d')}</div>
                </div>
                <div className="grid gap-1.5" style={{ gridTemplateColumns: 'repeat(8, minmax(0, 44px))' }}>
                  {vis.kernels.map((w, c) => <HeatMap key={c} values={w} side={5} ramp="signed" />)}
                </div>
              </div>
            </div>
          ) : (
            <p className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-center text-sm text-slate-500">{t('lab.mnist.net_empty')}</p>
          )}
        </Card>

        <Card step={3} title={t('lab.mnist.preds_title')} sub={t('lab.mnist.preds_sub')}>
          {test && preds ? (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(74px,1fr))] gap-2">
              {test.shown.map((i, n) => {
                const top = preds.top[n]
                const right = top.k[0] === test.labels[i]
                return (
                  <div key={i} title={top.k.map((k, j) => `${k}: ${(top.p[j] * 100).toFixed(1)}%`).join('  ')} className={`grid gap-1 rounded-xl border p-1.5 ${right ? 'border-slate-200 bg-white' : 'border-rose-300 bg-rose-50'}`}>
                    <div className="flex items-center gap-1.5">
                      <div className="w-9 shrink-0"><HeatMap values={digit(test.images, i)} side={IMAGE} ramp="grey" label={t('lab.mnist.digit_aria', { y: test.labels[i] })} /></div>
                      <span className={`font-mono text-xl font-black ${right ? 'text-slate-900' : 'text-rose-600'}`}>{top.k[0]}</span>
                    </div>
                    {top.k.map((k, j) => (
                      <div key={k} className="grid grid-cols-[10px_minmax(0,1fr)] items-center gap-1 font-mono text-[10px] text-slate-500">
                        {k}
                        <span className="h-1.5 overflow-hidden rounded-full bg-slate-100"><span className={`block h-full rounded-full ${j === 0 ? (right ? 'bg-emerald-500' : 'bg-rose-500') : 'bg-slate-300'}`} style={{ width: `${top.p[j] * 100}%` }} /></span>
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          ) : (
            <p className="flex items-center gap-2 text-sm text-slate-500"><LoaderCircle size={15} className="animate-spin" aria-hidden />{t('lab.mnist.loading')}</p>
          )}
        </Card>

        <Card step={4} title={t('lab.mnist.eval_title')} sub={t('lab.mnist.eval_sub')}>
          <div className="flex flex-wrap items-center gap-3">
            <Btn primary onClick={evaluate} disabled={!test || evaluating}>{evaluating ? t('lab.mnist.evaluating') : t('lab.mnist.eval_btn')}</Btn>
            {evaluation && <Stat k={t('lab.mnist.test_acc')} v={pct(evaluation.accuracy)} d={t('lab.mnist.eval_at', { n: evaluation.seen.toLocaleString('en-US') })} tone={evaluation.accuracy >= 0.95 ? 'good' : 'none'} />}
          </div>
          {evaluation && test && (
            <div className="mt-4 grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <figure className="grid gap-1.5">
                <figcaption className="text-xs font-bold text-slate-500">{t('lab.mnist.confusion')}</figcaption>
                <div className="overflow-x-auto">
                  <table className="border-separate border-spacing-0.5 font-mono text-[11px]">
                    <thead>
                      <tr><th />{Array.from({ length: CLASSES }, (_, c) => <th key={c} className="w-8 text-slate-500">{c}</th>)}</tr>
                    </thead>
                    <tbody>
                      {evaluation.confusion.map((row, y) => {
                        const total = row.reduce((s, v) => s + v, 0) || 1
                        return (
                          <tr key={y}>
                            <th className="pr-1 text-right text-slate-500">{y}</th>
                            {row.map((v, k) => (
                              <td key={k} className="h-7 w-8 rounded text-center" style={{ background: v ? (k === y ? `rgba(16,185,129,${0.15 + 0.6 * (v / total)})` : `rgba(225,29,72,${0.12 + Math.min(0.7, (v / total) * 6)})`) : '#f8fafc', color: v ? '#0f172a' : '#cbd5e1' }}>{v}</td>
                            ))}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </figure>
              <figure className="grid content-start gap-1.5">
                <figcaption className="text-xs font-bold text-slate-500">{t('lab.mnist.mistakes', { n: (test.labels.length - Math.round(evaluation.accuracy * test.labels.length)).toLocaleString('en-US') })}</figcaption>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(48px,1fr))] gap-1.5">
                  {evaluation.mistakes.map((m) => (
                    <div key={m.i} className="grid gap-0.5 text-center font-mono text-[10px]" title={t('lab.mnist.mistake_title', { y: test.labels[m.i], k: m.k, p: (m.p * 100).toFixed(0) })}>
                      <HeatMap values={digit(test.images, m.i)} side={IMAGE} ramp="grey" />
                      <span><b className="text-rose-600">{m.k}</b> <span className="text-slate-400">/ {test.labels[m.i]}</span></span>
                    </div>
                  ))}
                </div>
              </figure>
            </div>
          )}
        </Card>

        <Card step={5} title={t('lab.mnist.code_title')} sub={t('lab.mnist.code_sub')}>
          <PyBlock className="overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] leading-relaxed text-slate-100" code={code} />
          <ul className="mt-3 grid gap-1.5 text-[13px] leading-relaxed text-slate-600 md:grid-cols-2">
            {(['map_loss', 'map_acc', 'map_val', 'map_step'] as const).map((k) => <li key={k} className="rounded-lg bg-slate-50 px-3 py-2"><Trans i18nKey={`lab.mnist.${k}`} values={{ b: cfg.batchSize }} components={{ c: <code className="font-mono text-[12px] text-slate-800" /> }} /></li>)}
          </ul>
        </Card>

        <div className="grid gap-5 md:grid-cols-2">
          <Note title={t('lab.mnist.watch_title')}>{t('lab.mnist.watch')}</Note>
          <Note title={t('lab.mnist.diff_title')}>
            {t('lab.mnist.diff')}{' '}
            <a className="font-bold text-blue-700 hover:underline" href={moduleUrl('pytorch')}>{t('site.modules.pytorch.title')} →</a>
          </Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
