import { useCallback, useEffect, useRef, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { lastHidden, makeNet, makeTrainer, mulberry, outputLine, outputOf, toyData, vol } from '../../lib/nn'
import type { NnAct, ToyKind, ToyPoint } from '../../lib/nn'
import type { Net, SGDTrainer } from '../../lib/vendor/convnet.js'
import { fmt } from '../format'
import { Btn, Card, LabPage, Note, Presets, Seg, Slider, Stat, Workspace } from '../ui'

const STRUCTURES = [
  { id: 'none', hidden: [] as number[] },
  { id: 'two', hidden: [2] },
  { id: 'default', hidden: [6, 2] },
  { id: 'deep', hidden: [16, 16, 2] },
] as const
type StructureId = (typeof STRUCTURES)[number]['id']
const KINDS: ToyKind[] = ['blobs', 'xor', 'circle', 'spiral']
const BATCH = 10
const STEPS_PER_FRAME = 8
const CLS = ['#e11d48', '#2563eb']
const FIELD = 64
const VIEW = 400

const toCanvas = (v: number) => ((v + 1) / 2) * VIEW

/** the background of the input panel: each cell tinted by the class the network gives it */
function paintField(ctx: CanvasRenderingContext2D, net: Net, off: HTMLCanvasElement) {
  const o = off.getContext('2d')!
  const img = o.createImageData(FIELD, FIELD)
  for (let y = 0; y < FIELD; y++) {
    for (let x = 0; x < FIELD; x++) {
      const p = outputOf(net, [((x + 0.5) / FIELD) * 2 - 1, ((y + 0.5) / FIELD) * 2 - 1])[1]
      const i = (y * FIELD + x) * 4
      // rose-200 to blue-200, through white where the network is unsure
      const q = Math.abs(p - 0.5) * 2
      img.data[i] = p > 0.5 ? 255 - 64 * q : 255 - q
      img.data[i + 1] = p > 0.5 ? 255 - 36 * q : 255 - 50 * q
      img.data[i + 2] = p > 0.5 ? 255 - q : 255 - 44 * q
      img.data[i + 3] = 255
    }
  }
  o.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(off, 0, 0, VIEW, VIEW)
}

function dots(ctx: CanvasRenderingContext2D, pts: { at: readonly number[]; y: number }[]) {
  for (const { at, y } of pts) {
    ctx.beginPath()
    ctx.arc(at[0], at[1], 4.5, 0, 2 * Math.PI)
    ctx.fillStyle = CLS[y]
    ctx.fill()
    ctx.lineWidth = 1.25
    ctx.strokeStyle = '#fff'
    ctx.stroke()
  }
}

/** the plane as the last hidden layer sees it: the input's grid lines bent by the network, with the data on top */
function paintHidden(ctx: CanvasRenderingContext2D, net: Net, act: NnAct, pts: ToyPoint[]) {
  // ReLU layers have no fixed range, so the view follows the data
  let reach = act === 'tanh' ? 1.05 : 0.5
  const mapped = pts.map((p) => Array.from(lastHidden(net, p.x)))
  if (act !== 'tanh') for (const h of mapped) reach = Math.max(reach, Math.abs(h[0]) * 1.1, Math.abs(h[1]) * 1.1)
  const px = (v: number) => ((v + reach) / (2 * reach)) * VIEW
  ctx.fillStyle = '#f8fafc'
  ctx.fillRect(0, 0, VIEW, VIEW)
  // what the output layer decides: a straight line in this space
  const [a, b, c] = outputLine(net)
  ctx.lineWidth = 1
  ctx.strokeStyle = '#cbd5e1'
  const N = 10
  for (let k = 0; k <= N; k++) {
    for (const horizontal of [true, false]) {
      ctx.beginPath()
      for (let s = 0; s <= 40; s++) {
        const u = (k / N) * 2 - 1
        const v = (s / 40) * 2 - 1
        const h = lastHidden(net, horizontal ? [v, u] : [u, v])
        if (s === 0) ctx.moveTo(px(h[0]), px(h[1]))
        else ctx.lineTo(px(h[0]), px(h[1]))
      }
      ctx.stroke()
    }
  }
  if (Math.abs(a) + Math.abs(b) > 1e-9) {
    const along = Math.abs(b) > Math.abs(a)
    const f = (t: number) => (along ? [t, -(a * t + c) / b] : [-(b * t + c) / a, t])
    const [p0, p1] = [f(-reach), f(reach)]
    ctx.beginPath()
    ctx.moveTo(px(p0[0]), px(p0[1]))
    ctx.lineTo(px(p1[0]), px(p1[1]))
    ctx.lineWidth = 2.5
    ctx.strokeStyle = '#0f172a'
    ctx.setLineDash([7, 5])
    ctx.stroke()
    ctx.setLineDash([])
  }
  dots(ctx, mapped.map((h, i) => ({ at: [px(h[0]), px(h[1])], y: pts[i].y })))
}

export default function NnClassify() {
  const { t } = useTranslation()
  const [kind, setKind] = useState<ToyKind | null>('circle')
  const [points, setPoints] = useState<ToyPoint[]>(() => toyData('circle', mulberry(1)))
  const [structure, setStructure] = useState<StructureId>('default')
  const [act, setAct] = useState<NnAct>('tanh')
  const [lr, setLr] = useState(0.03)
  const [running, setRunning] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [brush, setBrush] = useState<0 | 1>(1)
  const [stats, setStats] = useState({ steps: 0, loss: 0, acc: 0 })
  const left = useRef<HTMLCanvasElement>(null)
  const right = useRef<HTMLCanvasElement>(null)
  const off = useRef<HTMLCanvasElement | null>(null)
  const net = useRef<Net | null>(null)
  const opt = useRef<SGDTrainer | null>(null)
  const steps = useRef(0)
  const pointsRef = useRef(points)
  const lrRef = useRef(lr)
  const rng = useRef(mulberry(11))
  const hidden = STRUCTURES.find((s) => s.id === structure)!.hidden
  const hiddenRef = useRef(hidden.length > 0)
  const actRef = useRef(act)
  hiddenRef.current = hidden.length > 0
  actRef.current = act

  useEffect(() => {
    pointsRef.current = points
  }, [points])
  useEffect(() => {
    lrRef.current = lr
    if (opt.current) opt.current.learning_rate = lr
  }, [lr])

  const draw = useCallback(() => {
    const m = net.current
    const a = left.current?.getContext('2d')
    if (!m || !a) return
    off.current ??= Object.assign(document.createElement('canvas'), { width: FIELD, height: FIELD })
    paintField(a, m, off.current)
    dots(a, pointsRef.current.map((p) => ({ at: [toCanvas(p.x[0]), toCanvas(p.x[1])], y: p.y })))
    const b = right.current?.getContext('2d')
    if (b && hiddenRef.current) paintHidden(b, m, actRef.current, pointsRef.current)
  }, [])

  const measure = useCallback(() => {
    const m = net.current
    const pts = pointsRef.current
    if (!m || !pts.length) {
      setStats({ steps: steps.current, loss: 0, acc: 0 })
      return
    }
    let loss = 0
    let good = 0
    for (const p of pts) {
      const q = outputOf(m, p.x)
      loss -= Math.log(Math.max(q[p.y], 1e-12))
      if ((q[1] > q[0] ? 1 : 0) === p.y) good++
    }
    setStats({ steps: steps.current, loss: loss / pts.length, acc: good / pts.length })
  }, [])

  const restart = useCallback(() => {
    net.current = makeNet(hidden, act, 'softmax', 2)
    opt.current = makeTrainer(net.current, lrRef.current, 0.9, BATCH, 1e-3)
    steps.current = 0
    draw()
    measure()
    // hidden is a constant array of the chosen structure
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structure, act, draw, measure])

  useEffect(() => {
    restart()
  }, [restart])

  useEffect(() => {
    draw()
    measure()
  }, [points, draw, measure])

  useEffect(() => {
    if (!running) return
    let id = 0
    let frame = 0
    const tick = () => {
      const m = net.current
      const o = opt.current
      const pts = pointsRef.current
      if (m && o && pts.length && !document.hidden) {
        for (let s = 0; s < STEPS_PER_FRAME * BATCH; s++) {
          const p = pts[Math.floor(rng.current.next() * pts.length)]
          o.train(vol(p.x), p.y)
        }
        steps.current += STEPS_PER_FRAME
        draw()
        if (++frame % 12 === 0) measure()
      }
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [running, draw, measure])

  const addPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x: [number, number] = [((e.clientX - r.left) / r.width) * 2 - 1, ((e.clientY - r.top) / r.height) * 2 - 1]
    setKind(null)
    setPoints((p) => [...p, { x, y: brush }])
  }

  return (
    <LabPage id="nn-classify">
      <Workspace
        wide
        controls={
          <>
            <Presets
              title={t('lab.toy.data_title')}
              items={KINDS.map((k) => ({ id: k, title: t(`lab.toy.data.${k}.title`), note: t(`lab.toy.data.${k}.note`) }))}
              value={kind}
              onPick={(k) => {
                setKind(k)
                setPoints(toyData(k, mulberry(1)))
                restart()
              }}
            />
            <Card title={t('lab.toy.net_title')}>
              <div className="grid gap-3">
                <Seg label={t('lab.toy.hidden')} value={structure} onChange={setStructure} options={STRUCTURES.map((s) => ({ v: s.id, label: s.hidden.length ? s.hidden.join('-') : t('lab.toy.none') }))} />
                <Seg label={t('lab.toy.activation')} value={act} onChange={setAct} options={[{ v: 'tanh', label: 'tanh' }, { v: 'relu', label: 'ReLU' }]} />
                <Slider label={t('lab.toy.lr')} value={lr} min={0.002} max={0.1} step={0.002} onChange={setLr} display={fmt(lr, 3)} />
                <div className="flex flex-wrap gap-2">
                  <Btn primary onClick={() => setRunning((r) => !r)}>{running ? t('lab.toy.pause') : t('lab.toy.run')}</Btn>
                  <Btn onClick={restart}>{t('lab.toy.restart')}</Btn>
                </div>
              </div>
            </Card>
            <Card title={t('lab.toy.add_title')}>
              <div className="grid gap-3">
                <Seg label={t('lab.toy.brush')} value={brush} onChange={setBrush} options={[{ v: 0, label: <><i className="mr-1 inline-block h-2.5 w-2.5 rounded-full bg-rose-600 ring-1 ring-white" aria-hidden />{t('lab.toy.red')}</> }, { v: 1, label: <><i className="mr-1 inline-block h-2.5 w-2.5 rounded-full bg-blue-600 ring-1 ring-white" aria-hidden />{t('lab.toy.blue')}</> }]} />
                <Btn onClick={() => { setKind(null); setPoints([]) }}>{t('lab.toy.clear')}</Btn>
              </div>
            </Card>
          </>
        }
      >
        <div className="grid grid-cols-3 gap-2">
          <Stat k={t('lab.toy.steps')} v={stats.steps.toLocaleString('en-US')} />
          <Stat k={t('lab.toy.loss')} v={fmt(stats.loss, 3)} />
          <Stat k={t('lab.toy.acc')} v={`${fmt(stats.acc * 100, 0)}%`} tone={stats.acc > 0.97 ? 'good' : 'none'} />
        </div>
        <div className="grid gap-5 md:grid-cols-2">
          <Card title={t('lab.toy.input_title')} sub={t('lab.toy.input_sub')}>
            <canvas ref={left} width={VIEW} height={VIEW} onPointerDown={addPoint} role="img" aria-label={t('lab.toy.input_aria', { n: points.length })} className="w-full cursor-crosshair touch-manipulation rounded-xl border border-slate-200" />
          </Card>
          <Card title={t('lab.toy.hidden_title')} sub={t('lab.toy.hidden_sub')}>
            {hidden.length ? (
              <canvas ref={right} width={VIEW} height={VIEW} role="img" aria-label={t('lab.toy.hidden_aria')} className="w-full rounded-xl border border-slate-200" />
            ) : (
              <Note>{t('lab.toy.no_hidden')}</Note>
            )}
          </Card>
        </div>
        <Card title={t('lab.toy.read_title')}>
          <ul className="grid gap-2 text-[13.5px] leading-relaxed text-slate-700">
            {(['line', 'warp', 'two', 'relu', 'spiral'] as const).map((k) => <li key={k}><Trans i18nKey={`lab.toy.read.${k}`} components={{ b: <b className="font-extrabold text-slate-900" /> }} /></li>)}
          </ul>
        </Card>
        <p className="text-[11.5px] leading-relaxed text-slate-500">
          <Trans i18nKey="lab.toy.credit" components={{ a: <a className="underline" href="https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html" target="_blank" rel="noopener noreferrer" /> }} />
        </p>
      </Workspace>
    </LabPage>
  )
}
