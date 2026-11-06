import { useCallback, useEffect, useRef, useState } from 'react'
import { ImageUp, LoaderCircle } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { ImageInputError, readUploadSquare } from '../../lib/imageInput'
import { PAINT_SIZE, makeNet, makeTrainer, mulberry, outputOf, paintPicture, rgbFromRgba, vol } from '../../lib/nn'
import type { PaintKind } from '../../lib/nn'
import type { Net, SGDTrainer } from '../../lib/vendor/convnet.js'
import { fmt } from '../format'
import { Btn, Card, LabPage, Presets, Seg, Slider, Stat, Workspace } from '../ui'

const KINDS: PaintKind[] = ['sunset', 'face', 'stripes']
const BATCH = 5
/** time for training in each frame, so the page stays smooth on a phone */
const TRAIN_MS = 10
/** rows of the painting redrawn each frame */
const ROWS_PER_FRAME = 6
const SIZE = PAINT_SIZE
const coord = (i: number) => (i + 0.5) / SIZE - 0.5

export default function NnPaint() {
  const { t } = useTranslation()
  const [kind, setKind] = useState<PaintKind | null>('sunset')
  const [target, setTarget] = useState<Float32Array>(() => paintPicture('sunset'))
  const [depth, setDepth] = useState(5)
  const [width, setWidth] = useState(20)
  const [lr, setLr] = useState(0.01)
  const [running, setRunning] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [stats, setStats] = useState({ steps: 0, loss: 0 })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const original = useRef<HTMLCanvasElement>(null)
  const painting = useRef<HTMLCanvasElement>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const net = useRef<Net | null>(null)
  const opt = useRef<SGDTrainer | null>(null)
  const steps = useRef(0)
  const ema = useRef(0)
  const row = useRef(0)
  const targetRef = useRef(target)
  const rng = useRef(mulberry(5))

  const restart = useCallback(() => {
    net.current = makeNet(new Array<number>(depth).fill(width), 'relu', 'regression', 3)
    opt.current = makeTrainer(net.current, lr, 0.9, BATCH)
    steps.current = 0
    ema.current = 0
    row.current = 0
    setStats({ steps: 0, loss: 0 })
    // lr has its own effect, and a new rate should not throw the painting away
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depth, width])

  useEffect(() => {
    restart()
  }, [restart])
  useEffect(() => {
    if (opt.current) opt.current.learning_rate = lr
  }, [lr])

  // the picture to copy
  useEffect(() => {
    targetRef.current = target
    const ctx = original.current?.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(SIZE, SIZE)
    for (let i = 0; i < SIZE * SIZE; i++) {
      for (let c = 0; c < 3; c++) img.data[i * 4 + c] = target[i * 3 + c] * 255
      img.data[i * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [target])

  const paintRows = useCallback((count: number) => {
    const m = net.current
    const ctx = painting.current?.getContext('2d')
    if (!m || !ctx) return
    const img = ctx.createImageData(SIZE, count)
    for (let r = 0; r < count; r++) {
      const y = (row.current + r) % SIZE
      for (let x = 0; x < SIZE; x++) {
        const p = outputOf(m, [coord(x), coord(y)])
        const i = (r * SIZE + x) * 4
        for (let c = 0; c < 3; c++) img.data[i + c] = p[c] * 255
        img.data[i + 3] = 255
      }
    }
    // rows wrap around: write each where it belongs
    for (let r = 0; r < count; r++) ctx.putImageData(img, 0, (row.current + r) % SIZE - r, 0, r, SIZE, 1)
    row.current = (row.current + count) % SIZE
  }, [])

  useEffect(() => {
    paintRows(SIZE)
  }, [paintRows, depth, width])

  useEffect(() => {
    if (!running) return
    let id = 0
    let frame = 0
    const tick = () => {
      const m = net.current
      const o = opt.current
      if (m && o && !document.hidden) {
        const t0 = performance.now()
        const tgt = targetRef.current
        do {
          for (let b = 0; b < BATCH; b++) {
            const x = Math.floor(rng.current.next() * SIZE)
            const y = Math.floor(rng.current.next() * SIZE)
            const i = (y * SIZE + x) * 3
            const l = o.train(vol([coord(x), coord(y)]), [tgt[i], tgt[i + 1], tgt[i + 2]]).cost_loss
            ema.current = steps.current ? ema.current * 0.999 + l * 0.001 : l
          }
          steps.current++
        } while (performance.now() - t0 < TRAIN_MS)
        paintRows(ROWS_PER_FRAME)
        if (++frame % 15 === 0) setStats({ steps: steps.current, loss: ema.current })
      }
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [running, paintRows])

  const onFile = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const rgba = await readUploadSquare(file, SIZE)
      setKind(null)
      setTarget(rgbFromRgba(rgba, SIZE * SIZE))
      restart()
    } catch (e) {
      setError(t(`lab.paint.err_${e instanceof ImageInputError ? e.code : 'decode'}`))
    } finally {
      setBusy(false)
    }
  }

  const params = 3 * width + (depth - 1) * (width * width + width) + 3 * width + 3

  return (
    <LabPage id="nn-paint">
      <Workspace
        wide
        controls={
          <>
            <Presets
              title={t('lab.paint.picture')}
              items={KINDS.map((k) => ({ id: k, title: t(`lab.paint.pic.${k}.title`), note: t(`lab.paint.pic.${k}.note`) }))}
              value={kind}
              onPick={(k) => {
                setKind(k)
                setError(null)
                setTarget(paintPicture(k))
                restart()
              }}
            >
              <button type="button" onClick={() => fileInput.current?.click()} disabled={busy} className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-[13px] font-semibold text-slate-700 hover:bg-slate-50">
                {busy ? <LoaderCircle size={14} className="animate-spin" aria-hidden /> : <ImageUp size={14} aria-hidden />}
                {t('lab.paint.upload')}
              </button>
              <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp" className="hidden" onChange={(e) => { void onFile(e.target.files?.[0]); e.target.value = '' }} />
              <p className="mt-1.5 text-[11.5px] leading-snug text-slate-500">{t('lab.paint.upload_note')}</p>
              {error && <p role="alert" className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700">{error}</p>}
            </Presets>
            <Card title={t('lab.paint.net_title')}>
              <div className="grid gap-3">
                <Seg label={t('lab.paint.depth')} value={depth} onChange={setDepth} options={[1, 2, 3, 5, 7].map((v) => ({ v, label: v }))} />
                <Seg label={t('lab.paint.width')} value={width} onChange={setWidth} options={[5, 10, 20, 40].map((v) => ({ v, label: v }))} />
                <Slider label={t('lab.paint.lr')} value={lr} min={0.001} max={0.03} step={0.001} onChange={setLr} display={fmt(lr, 3)} />
                <div className="flex flex-wrap gap-2">
                  <Btn primary onClick={() => setRunning((r) => !r)}>{running ? t('lab.paint.pause') : t('lab.paint.run')}</Btn>
                  <Btn onClick={restart}>{t('lab.paint.restart')}</Btn>
                </div>
              </div>
            </Card>
          </>
        }
      >
        <div className="grid grid-cols-3 gap-2">
          <Stat k={t('lab.paint.steps')} v={stats.steps.toLocaleString('en-US')} />
          <Stat k={t('lab.paint.loss')} v={fmt(stats.loss, 4)} />
          <Stat k={t('lab.paint.params')} v={params.toLocaleString('en-US')} d={t('lab.paint.pixels', { n: (SIZE * SIZE * 3).toLocaleString('en-US') })} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <Card title={t('lab.paint.original')}>
            <canvas ref={original} width={SIZE} height={SIZE} role="img" aria-label={t('lab.paint.original')} className="w-full rounded-xl [image-rendering:pixelated]" />
          </Card>
          <Card title={t('lab.paint.painted')}>
            <canvas ref={painting} width={SIZE} height={SIZE} role="img" aria-label={t('lab.paint.painted')} className="w-full rounded-xl [image-rendering:pixelated]" />
          </Card>
        </div>
        <Card title={t('lab.paint.read_title')}>
          <ul className="grid gap-2 text-[13.5px] leading-relaxed text-slate-700">
            {(['what', 'edges', 'depth', 'compress'] as const).map((k) => <li key={k}><Trans i18nKey={`lab.paint.read.${k}`} components={{ b: <b className="font-extrabold text-slate-900" /> }} /></li>)}
          </ul>
        </Card>
        <p className="text-[11.5px] leading-relaxed text-slate-500">
          <Trans i18nKey="lab.paint.credit" components={{ a: <a className="underline" href="https://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html" target="_blank" rel="noopener noreferrer" /> }} />
        </p>
      </Workspace>
    </LabPage>
  )
}
