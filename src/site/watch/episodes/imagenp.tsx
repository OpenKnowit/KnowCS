/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { useTranslation } from 'react-i18next'
import { eseg, seg } from '../../../lib/explainer'
import { circleMask, flattenConv1d, separableBlur } from '../../../lib/imageNp'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Image processing without loops" — Final 2024 Q2: a circular mask by broadcasting, a blur by flattening.

const N = 14
// a small procedural greyscale picture: soft diagonal gradient plus a bright blob
const IMG: number[][] = Array.from({ length: N }, (_, i) =>
  Array.from({ length: N }, (_, j) => Math.round(Math.min(255, 40 + (i + j) * 6 + 150 * Math.exp(-((i - 8) ** 2 + (j - 4) ** 2) / 10)))),
)
const CX = 6
const CY = 5
const R = 4
const MASK = circleMask(N, N, CX, CY, R)
const THIRD = 1 / 3
const BLUR = separableBlur(IMG, [THIRD, THIRD, THIRD])
const TINY = [[3, 6, 9, 3], [0, 3, 0, 6], [9, 0, 3, 3]]
const TINY_RES = flattenConv1d(TINY, [1, 1, 1])

function Pixels({ x, y, v, cs = 34, o = 1, mask, ring }: { x: number; y: number; v: number[][]; cs?: number; o?: number; mask?: number[][]; ring?: boolean }) {
  return (
    <g opacity={o}>
      {v.flatMap((row, i) =>
        row.map((val, j) => {
          const g = Math.round(mask && !mask[i][j] ? 0 : val)
          return <rect key={`${i}-${j}`} x={x + j * cs} y={y + i * cs} width={cs - 2} height={cs - 2} rx={3} fill={`rgb(${g},${g},${g})`} />
        }),
      )}
      {ring && <circle cx={x + CX * cs + cs / 2 - 1} cy={y + CY * cs + cs / 2 - 1} r={R * cs} fill="none" stroke={C.yellow} strokeWidth={3} strokeDasharray="8 6" />}
    </g>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.imagenp.kicker')} title={t('watch.imagenp.title')} sub={t('watch.imagenp.sub')} />
}

function MaskScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const step = p < 0.25 ? 0 : p < 0.5 ? 1 : p < 0.72 ? 2 : 3
  const cs = 34
  const x0 = 120
  const y0 = 220
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {step === 0 && <Pixels x={x0} y={y0} v={IMG} />}
        {step >= 1 &&
          MASK.dist.flatMap((row, i) =>
            row.map((d, j) => {
              const inside = d <= R
              const shade = Math.max(0, 1 - d / 10)
              return <rect key={`${i}-${j}`} x={x0 + j * cs} y={y0 + i * cs} width={cs - 2} height={cs - 2} rx={3} fill={step >= 2 ? (inside ? C.yellow : '#1a2130') : C.blue} fillOpacity={step >= 2 ? (inside ? 0.85 : 1) : shade} opacity={step === 3 ? 0 : 1} />
            }),
          )}
        {step === 3 && <Pixels x={x0} y={y0} v={IMG} mask={MASK.mask} ring />}
      </Svg>
      <At x={x0 + 7 * cs} y={180} size={24} className="font-mono" color={C.muted}>
        {['img', 'dist', 'dist <= r', 'img * (dist <= r)'][step]}
      </At>
      <At x={1150} y={190} size={24} className="font-mono" o={eseg(p, 0.2, 0.3)}>x_dist = np.arange(N) - center_x</At>
      <At x={1150} y={240} size={24} className="font-mono" o={eseg(p, 0.25, 0.35)}>y_dist = np.arange(N) - center_y</At>
      <At x={1150} y={300} size={24} className="font-mono" color={C.blue} o={eseg(p, 0.3, 0.4)}>dist = np.sqrt(x_dist ** 2 + y_dist[:, None] ** 2)</At>
      <At x={1150} y={360} size={22} color={C.muted} w={640} o={eseg(p, 0.35, 0.45)} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.broadcast_dist')}</At>
      <At x={1150} y={470} size={24} className="font-mono" color={C.yellow} o={eseg(p, 0.5, 0.6)}>mask = dist &lt;= 100</At>
      <At x={1150} y={540} size={24} className="font-mono" color={C.teal} o={eseg(p, 0.72, 0.8)}>img_masked = img * mask</At>
      <At x={1150} y={640} size={22} color={C.muted} w={640} o={eseg(p, 0.8, 0.88)} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.boundary')}</At>
    </>
  )
}

function SeparableScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
      </Svg>
      <Tex f="\frac19\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix} = \frac13\begin{bmatrix}1\\1\\1\end{bmatrix}\cdot\frac13\begin{bmatrix}1&1&1\end{bmatrix}" x={800} y={250} size={46} o={eseg(p, 0.05, 0.2)} />
      <At x={800} y={450} size={30} w={1300} o={eseg(p, 0.3, 0.4)} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.two_passes')}</At>
      <At x={800} y={580} size={26} className="font-mono" color={C.muted} o={eseg(p, 0.5, 0.6)}>v = blur_filter.sum(0)   # [1/3, 1/3, 1/3]</At>
      <At x={800} y={700} size={26} color={C.yellow} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.cheaper')}</At>
    </>
  )
}

function FlattenScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const cs = 62
  const stage = p < 0.2 ? 0 : p < 0.4 ? 1 : p < 0.7 ? 2 : 3
  const flatX = (k: number) => 110 + k * 46
  const k = Math.min(TINY_RES.flat.length - 3, Math.floor(seg(p, 0.42, 0.68) * (TINY_RES.flat.length - 2)))
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {/* the image, padded */}
        {(stage === 0 ? TINY : TINY_RES.padded).flatMap((row, i) =>
          row.map((v, j) => {
            const isPad = stage > 0 && (j === 0 || j === row.length - 1)
            return (
              <g key={`${i}-${j}`}>
                <rect x={160 + j * cs} y={150 + i * cs} width={cs - 6} height={cs - 6} rx={8} fill={isPad ? C.bg : '#1a2130'} stroke={isPad ? C.axis : i === 0 ? C.blue : i === 1 ? C.orange : C.teal} strokeDasharray={isPad ? '5 4' : undefined} strokeWidth={2} />
                <text x={160 + j * cs + (cs - 6) / 2} y={150 + i * cs + cs / 2 + 6} textAnchor="middle" fill={isPad ? C.muted : C.text} fontSize={22} fontFamily="ui-monospace, monospace">{v}</text>
              </g>
            )
          }),
        )}
        {/* flattened strip */}
        {stage >= 2 &&
          TINY_RES.flat.map((v, i) => {
            const row = Math.floor(i / 6)
            const isPad = i % 6 === 0 || i % 6 === 5
            return (
              <g key={'f' + i}>
                <rect x={flatX(i)} y={480} width={40} height={50} rx={6} fill={isPad ? C.bg : '#1a2130'} stroke={isPad ? C.axis : [C.blue, C.orange, C.teal][row]} strokeDasharray={isPad ? '4 3' : undefined} strokeWidth={2} />
                <text x={flatX(i) + 20} y={512} textAnchor="middle" fill={isPad ? C.muted : C.text} fontSize={18} fontFamily="ui-monospace, monospace">{v}</text>
              </g>
            )
          })}
        {stage === 2 && <rect x={flatX(k) - 4} y={472} width={46 * 3 + 2} height={66} rx={10} fill="none" stroke={C.yellow} strokeWidth={4} />}
        {stage >= 3 &&
          TINY_RES.conv.map((v, i) => {
            const isPad = i % 6 === 0 || i % 6 === 5
            return (
              <g key={'c' + i}>
                <rect x={flatX(i)} y={600} width={40} height={50} rx={6} fill={isPad ? '#33191d' : '#173a35'} stroke={isPad ? C.red : C.teal} strokeWidth={2} />
                <text x={flatX(i) + 20} y={632} textAnchor="middle" fill={isPad ? C.red : C.text} fontSize={18} fontFamily="ui-monospace, monospace" textDecoration={isPad ? 'line-through' : undefined}>{v}</text>
              </g>
            )
          })}
      </Svg>
      <At x={1150} y={180} size={24} className="font-mono" o={stage >= 1 ? 1 : 0}>np.concatenate((zeros, img, zeros), axis=1)</At>
      <At x={1150} y={240} size={24} className="font-mono" o={stage >= 2 ? 1 : 0}>img_padded.reshape(-1)</At>
      <At x={1150} y={300} size={24} className="font-mono" o={stage >= 2 ? eseg(p, 0.42, 0.5) : 0}>np.convolve(img_flat, v, 'valid')</At>
      <At x={1150} y={360} size={24} className="font-mono" o={stage >= 3 ? 1 : 0}>.reshape(H, W + 2)[:, 1:-1]</At>
      <At x={800} y={720} size={24} color={C.muted} w={1400} o={stage >= 3 ? eseg(p, 0.75, 0.85) : 0} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.drop_pad')}</At>
      <At x={800} y={800} size={22} color={C.muted} w={1400} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.ones_note')}</At>
    </>
  )
}

function ResultScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Pixels x={140} y={200} v={IMG} cs={34} />
        <Pixels x={820} y={200} v={BLUR} cs={34} o={eseg(p, 0.15, 0.3)} />
      </Svg>
      <At x={378} y={170} size={24} className="font-mono" color={C.muted}>img</At>
      <At x={1058} y={170} size={24} className="font-mono" color={C.muted} o={eseg(p, 0.15, 0.3)}>img_blur</At>
      <At x={800} y={740} size={24} className="font-mono" o={eseg(p, 0.3, 0.4)}>img_blur = img_flatten_conv_1d(img, v)</At>
      <At x={800} y={790} size={24} className="font-mono" o={eseg(p, 0.4, 0.5)}>img_blur = img_flatten_conv_1d(img_blur.T, v).T</At>
      <At x={800} y={850} size={22} color={C.yellow} w={1400} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.imagenp.label.edges_dark')}</At>
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
          {t(`watch.imagenp.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const imagenp: Episode = {
  id: 'imagenp',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'mask', dur: 22000, cues: [0, 0.25, 0.5, 0.72], ponder: true, render: (p) => <MaskScene p={p} /> },
    { id: 'separable', dur: 14000, cues: [0, 0.3, 0.65], render: (p) => <SeparableScene p={p} /> },
    { id: 'flatten', dur: 26000, cues: [0, 0.2, 0.42, 0.7], render: (p) => <FlattenScene p={p} /> },
    { id: 'result', dur: 13000, cues: [0, 0.3, 0.6], render: (p) => <ResultScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
