/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, seg } from '../../../lib/explainer'
import { convolve, outputSize } from '../../../lib/conv2d'
import { analyse, totalParams } from '../../../lib/cnnShapes'
import type { Layer } from '../../../lib/cnnShapes'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "From a sliding window to a CNN's shape table" — lectures 7–8; the shape/parameter table is in every final.

const IMG = [
  [1, 2, 3, 4, 5, 6, 7],
  [2, 3, 4, 5, 6, 7, 8],
  [3, 4, 5, 6, 7, 8, 9],
  [4, 5, 6, 7, 8, 9, 8],
  [5, 6, 7, 8, 9, 8, 7],
  [6, 7, 8, 9, 8, 7, 6],
  [7, 8, 9, 8, 7, 6, 5],
]
const KER = [
  [0, 1, 0],
  [1, 1, 1],
  [0, 1, 0],
]

const conv = (filters: number, k: number, stride = 1, pad = 0): Layer => ({ kind: 'conv', filters, k, stride, pad, bias: true })
// Final 2022 Part B Q1: a deep stem and a 1000-way classifier on 224 × 224 × 3 images.
const STACK: Layer[] = [conv(64, 7, 2, 3), { kind: 'pool', k: 3, stride: 2, pad: 1, op: 'max' }, conv(128, 3, 2, 0), conv(256, 3, 2, 1), conv(512, 3, 2, 1), { kind: 'globalpool', op: 'avg' }, { kind: 'dense', units: 1000, bias: true }]
const INFOS = analyse([224, 224, 3], STACK)
const TOTAL = totalParams(INFOS)
const MLP = (224 * 224 * 3 + 1) * 1000

const n0 = (v: number) => v.toLocaleString('en-US')

/**
 * An n × n image (plus p rings of padding) at (x, y) with cell size s, the window at (r0, c0) of size k,
 * and optionally cells never covered by any window (`missed`).
 */
function ImageGrid({ x, y, s, img, p = 0, win, k = 3, missed, o = 1, values = true }: { x: number; y: number; s: number; img: number[][]; p?: number; win?: [number, number] | null; k?: number; missed?: (r: number, c: number) => boolean; o?: number; values?: boolean }) {
  const n = img.length + 2 * p
  const cells = []
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const inside = r >= p && c >= p && r < n - p && c < n - p
      const miss = inside && missed?.(r - p, c - p)
      cells.push(
        <g key={r * n + c}>
          <rect x={x + c * s} y={y + r * s} width={s - 3} height={s - 3} rx={5} fill={miss ? '#3b1d22' : inside ? '#1a2130' : C.bg} stroke={miss ? C.red : inside ? '#2a3446' : C.axis} strokeDasharray={inside ? undefined : '4 4'} strokeWidth={1.5} />
          {values && (
            <text x={x + c * s + (s - 3) / 2} y={y + r * s + s / 2 + 6} textAnchor="middle" fill={inside ? C.text : C.muted} fontSize={s * 0.38} fontFamily="ui-monospace, monospace">
              {inside ? img[r - p][c - p] : 0}
            </text>
          )}
        </g>,
      )
    }
  return (
    <g opacity={o}>
      {cells}
      {win && <rect x={x + win[1] * s - 4} y={y + win[0] * s - 4} width={k * s + 5} height={k * s + 5} rx={8} fill={C.yellow} fillOpacity={0.12} stroke={C.yellow} strokeWidth={4} />}
    </g>
  )
}

function OutGrid({ x, y, s, out, filled, cur, o = 1 }: { x: number; y: number; s: number; out: number[][]; filled: number; cur?: number; o?: number }) {
  const w = out[0].length
  return (
    <g opacity={o}>
      {out.flatMap((row, r) =>
        row.map((v, c) => {
          const i = r * w + c
          const on = i < filled
          return (
            <g key={i}>
              <rect x={x + c * s} y={y + r * s} width={s - 3} height={s - 3} rx={5} fill={i === cur ? C.yellow : on ? '#173a35' : '#141a24'} stroke={on ? C.teal : '#2a3446'} strokeWidth={1.5} />
              {on && (
                <text x={x + c * s + (s - 3) / 2} y={y + r * s + s / 2 + 6} textAnchor="middle" fill={i === cur ? C.bg : C.text} fontSize={s * 0.36} fontWeight={i === cur ? 800 : 400} fontFamily="ui-monospace, monospace">
                  {v}
                </text>
              )}
            </g>
          )
        }),
      )}
    </g>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.cnn.kicker')} title={t('watch.cnn.title')} sub={t('watch.cnn.sub')} />
}

const RES = convolve(IMG, KER, { pad: 0, mode: 'zero', stride: 1, flip: false })
function SlideScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const total = 25
  const k = Math.min(total - 1, Math.floor(seg(p, 0.12, 0.85) * total))
  const r = Math.floor(k / 5)
  const c = k % 5
  const terms = KER.flatMap((row, i) => row.map((kv, j) => ({ kv, v: IMG[r + i][c + j] }))).filter((x) => x.kv)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <ImageGrid x={90} y={180} s={70} img={IMG} win={p > 0.08 ? [r, c] : null} />
        <OutGrid x={1000} y={250} s={80} out={RES.out} filled={p > 0.12 ? k + 1 : 0} cur={p > 0.12 ? k : undefined} />
      </Svg>
      <At x={335} y={140} size={26} color={C.muted}>{t('watch.cnn.label.image', { n: 7 })}</At>
      <Tex f="K = \begin{bmatrix} 0&1&0\\1&1&1\\0&1&0 \end{bmatrix}" x={760} y={330} size={30} o={eseg(p, 0, 0.1)} />
      <At x={1196} y={200} size={26} color={C.muted}>{t('watch.cnn.label.output')}</At>
      <At x={800} y={760} size={30} className="font-mono" o={eseg(p, 0.12, 0.2)}>
        {terms.map((x) => x.v).join(' + ')} = <b style={{ color: C.yellow }}>{RES.out[r][c]}</b>
      </At>
    </>
  )
}

function SizeScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const pos = Math.min(4, Math.floor(seg(p, 0.08, 0.45) * 5))
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {Array.from({ length: 7 }, (_, i) => <rect key={i} x={250 + i * 160} y={250} width={150} height={80} rx={10} fill="#1a2130" stroke="#2a3446" strokeWidth={2} />)}
        {Array.from({ length: pos + 1 }, (_, j) => (
          <rect key={j} x={250 + j * 160 - 4} y={360 + j * 26} width={3 * 160 - 2} height={18} rx={9} fill={j === pos ? C.yellow : C.teal} opacity={j === pos ? 1 : 0.5} />
        ))}
      </Svg>
      {Array.from({ length: 7 }, (_, i) => <At key={i} x={325 + i * 160} y={290} size={28} color={C.muted} className="font-mono">{i + 1}</At>)}
      <At x={800} y={170} size={32} className="font-black" o={eseg(p, 0, 0.08)}>{t('watch.cnn.label.count_positions')}</At>
      <Tex f="N - K + 1 = 7 - 3 + 1 = 5" x={800} y={600} size={42} color={C.teal} o={eseg(p, 0.45, 0.55)} />
      <Tex f="\left\lfloor \dfrac{N - K + 2P}{S} \right\rfloor + 1" x={800} y={740} size={50} color={C.yellow} o={eseg(p, 0.68, 0.8)} />
    </>
  )
}

const PADDED = convolve(IMG, KER, { pad: 1, mode: 'zero', stride: 1, flip: false })
function PaddingScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const k = Math.min(48, Math.floor(seg(p, 0.15, 0.7) * 49))
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <ImageGrid x={90} y={130} s={68} img={IMG} p={1} win={[Math.floor(k / 7), k % 7]} />
        <OutGrid x={900} y={180} s={64} out={PADDED.out} filled={k + 1} cur={k} />
      </Svg>
      <At x={1124} y={140} size={26} color={C.muted}>{t('watch.cnn.label.output_same', { n: 7 })}</At>
      <Tex f="\left\lfloor \dfrac{7 - 3 + 2\cdot 1}{1} \right\rfloor + 1 = 7" x={1124} y={680} size={36} color={C.teal} o={eseg(p, 0.6, 0.7)} />
      <At x={1124} y={780} size={26} color={C.muted} w={600} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.cnn.label.same')}</At>
    </>
  )
}

const STRIDED = convolve(IMG, KER, { pad: 0, mode: 'zero', stride: 3, flip: false })
function StrideScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const k = Math.min(3, Math.floor(seg(p, 0.1, 0.5) * 4))
  const os = outputSize(7, 3, 0, 3)
  const showMiss = p > 0.55
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <ImageGrid x={90} y={180} s={70} img={IMG} win={[Math.floor(k / 2) * 3, (k % 2) * 3]} missed={showMiss ? (r, c) => r === 6 || c === 6 : undefined} />
        <OutGrid x={1050} y={280} s={90} out={STRIDED.out} filled={k + 1} cur={k} />
      </Svg>
      <At x={1135} y={230} size={26} color={C.muted}>{t('watch.cnn.label.stride3')}</At>
      <Tex f={`\\left\\lfloor \\dfrac{7 - 3 + 0}{3} \\right\\rfloor + 1 = \\lfloor 1.33 \\rfloor + 1 = ${os.size}`} x={1135} y={560} size={32} color={C.teal} o={eseg(p, 0.4, 0.5)} />
      <At x={1135} y={680} size={28} color={C.red} w={560} o={eseg(p, 0.58, 0.68)} style={{ textAlign: 'center' }}>{t('watch.cnn.label.never_seen')}</At>
    </>
  )
}

/** A volume drawn as an oblique box. */
function Box({ x, y, w, h, d, color, o = 1 }: { x: number; y: number; w: number; h: number; d: number; color: string; o?: number }) {
  const k = 0.5
  return (
    <g opacity={o}>
      <polygon points={`${x},${y} ${x + d * k},${y - d * k} ${x + w + d * k},${y - d * k} ${x + w},${y}`} fill={color} fillOpacity={0.35} stroke={color} strokeWidth={2} />
      <polygon points={`${x + w},${y} ${x + w + d * k},${y - d * k} ${x + w + d * k},${y + h - d * k} ${x + w},${y + h}`} fill={color} fillOpacity={0.25} stroke={color} strokeWidth={2} />
      <rect x={x} y={y} width={w} height={h} fill={color} fillOpacity={0.18} stroke={color} strokeWidth={2} />
    </g>
  )
}

function ChannelsScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const maps = Math.floor(seg(p, 0.45, 0.75) * 10)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Box x={120} y={320} w={300} h={300} d={40} color={C.blue} o={eseg(p, 0, 0.1)} />
        <Box x={520} y={420} w={60} h={60} d={40} color={C.yellow} o={eseg(p, 0.15, 0.25)} />
        {Array.from({ length: maps }, (_, i) => (
          <Box key={i} x={760 + i * 16} y={320 - i * 16} w={300} h={300} d={4} color={C.teal} o={0.9} />
        ))}
      </Svg>
      <At x={290} y={680} size={30} className="font-mono" o={eseg(p, 0, 0.1)}>32 × 32 × 3</At>
      <At x={570} y={560} size={26} className="font-mono" color={C.yellow} o={eseg(p, 0.15, 0.25)}>5 × 5 × 3</At>
      <At x={575} y={640} size={22} color={C.muted} w={240} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.cnn.label.depth_matches')}</At>
      <At x={990} y={680} size={30} className="font-mono" color={C.teal} o={eseg(p, 0.45, 0.55)}>32 × 32 × {Math.max(1, maps)}</At>
      <At x={800} y={110} size={28} color={C.muted} w={1300} o={eseg(p, 0.3, 0.4)} style={{ textAlign: 'center' }}>{t('watch.cnn.label.one_map')}</At>
      <Tex f="(5\cdot 5\cdot 3 + 1)\times 10 = 760" x={800} y={800} size={40} color={C.yellow} o={eseg(p, 0.78, 0.88)} />
    </>
  )
}

function SharingScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const dense = (32 * 32 * 3 + 1) * (32 * 32 * 10)
  const L = (v: number) => (Math.log10(v) / Math.log10(dense)) * 1100
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <rect x={300} y={330} width={L(760) * eseg(p, 0.15, 0.3)} height={60} rx={10} fill={C.teal} />
        <rect x={300} y={500} width={L(dense) * eseg(p, 0.35, 0.55)} height={60} rx={10} fill={C.red} />
      </Svg>
      <At x={800} y={170} size={34} className="font-black" o={eseg(p, 0, 0.1)} w={1300} style={{ textAlign: 'center' }}>{t('watch.cnn.label.same_job')}</At>
      <At x={280} y={360} size={26} anchor="r" color={C.muted} o={eseg(p, 0.15, 0.25)}>Conv</At>
      <At x={280} y={530} size={26} anchor="r" color={C.muted} o={eseg(p, 0.35, 0.45)}>Dense</At>
      <At x={310 + L(760)} y={360} size={34} anchor="l" className="font-mono font-black" color={C.teal} o={eseg(p, 0.25, 0.32)}>760</At>
      <At x={1380} y={600} size={34} anchor="r" className="font-mono font-black" color={C.red} o={eseg(p, 0.5, 0.57)}>{n0(dense)}</At>
      <At x={800} y={680} size={22} color={C.muted} o={eseg(p, 0.5, 0.6)}>{t('watch.cnn.label.log_scale')}</At>
      <At x={800} y={770} size={28} color={C.yellow} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.cnn.label.sharing')}</At>
    </>
  )
}

function StackScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const shown = Math.floor(seg(p, 0.05, 0.6) * (INFOS.length + 1))
  const shapes = [[224, 224, 3], ...INFOS.map((i) => i.output)]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {shapes.slice(0, shown + 1).map((s, i) => {
          const flat = s.length === 1
          const h = flat ? Math.min(260, 40 + Math.log2(s[0]) * 18) : 30 + Math.log2(s[0]) * 26
          const d = flat ? 6 : 8 + Math.log2(s[2]) * 9
          return flat ? (
            <rect key={i} x={90 + i * 180} y={480 - h / 2} width={22} height={h} rx={5} fill={C.purple} fillOpacity={0.5} stroke={C.purple} strokeWidth={2} />
          ) : (
            <Box key={i} x={70 + i * 180} y={480 - h / 2} w={h * 0.5} h={h} d={d} color={i === 0 ? C.blue : C.teal} />
          )
        })}
      </Svg>
      {shapes.slice(0, shown + 1).map((s, i) => (
        <Fragment key={i}>
          <At x={120 + i * 180} y={700} size={21} className="font-mono" color={C.text}>{s.join('×')}</At>
          <At x={120 + i * 180} y={735} size={18} color={C.muted}>{i === 0 ? t('watch.cnn.label.input') : t(`lab.cnn.kinds.${INFOS[i - 1].layer.kind}`)}</At>
          {i > 0 && <At x={120 + i * 180} y={770} size={18} className="font-mono" color={C.yellow}>{n0(INFOS[i - 1].params)}</At>}
        </Fragment>
      ))}
      <At x={800} y={130} size={30} color={C.muted} o={eseg(p, 0, 0.08)}>{t('watch.cnn.label.final22')}</At>
      <At x={560} y={850} size={30} o={eseg(p, 0.66, 0.74)}>
        {t('watch.cnn.label.total')} <b style={{ color: C.teal }}>{n0(TOTAL)}</b>
      </At>
      <At x={1180} y={850} size={30} o={eseg(p, 0.76, 0.84)}>
        {t('watch.cnn.label.mlp')} <b style={{ color: C.red }}>{n0(MLP)}</b>
      </At>
    </>
  )
}

function Recap({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.3} />
      </Svg>
      <At x={800} y={150} size={30} color={C.blue} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>{t('watch.ui.recap')}</At>
      {[1, 2, 3].map((k) => (
        <At key={k} x={220} y={240 + k * 120} size={36} anchor="l" w={1180} o={eseg(p, 0.08 + k * 0.12, 0.16 + k * 0.12)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.cnn.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const cnn: Episode = {
  id: 'cnn',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'slide', dur: 17000, cues: [0, 0.15, 0.6], render: (p) => <SlideScene p={p} /> },
    { id: 'size', dur: 15000, cues: [0, 0.45, 0.68], render: (p) => <SizeScene p={p} /> },
    { id: 'padding', dur: 14000, cues: [0, 0.6], render: (p) => <PaddingScene p={p} /> },
    { id: 'stride', dur: 15000, cues: [0, 0.4, 0.58], ponder: true, render: (p) => <StrideScene p={p} /> },
    { id: 'channels', dur: 17000, cues: [0, 0.2, 0.45, 0.78], render: (p) => <ChannelsScene p={p} /> },
    { id: 'sharing', dur: 14000, cues: [0, 0.35, 0.65], render: (p) => <SharingScene p={p} /> },
    { id: 'stack', dur: 18000, cues: [0, 0.3, 0.66], render: (p) => <StackScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
