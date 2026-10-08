import { useState } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { MODULES, moduleUrl } from '../../lib/sitemap'
import type { ModuleId, ModuleInfo } from '../../lib/sitemap'
import { readVisited, usePaperLabel } from '../ui'

// Home page, second view: a toolkit-style grid of every lab page, filtered by area, each card with a line sketch.

// color draws the sketch; ink is a darker shade of it for small text (WCAG AA on white)
const AREAS = [
  { id: 'numpy', lecs: [1], color: '#0284c7', ink: '#0369a1' },
  { id: 'prob', lecs: [2], color: '#7c3aed', ink: '#6d28d9' },
  { id: 'classic', lecs: [3, 4], color: '#059669', ink: '#047857' },
  { id: 'neural', lecs: [5, 6, 9], color: '#d97706', ink: '#b45309' },
  { id: 'vision', lecs: [7, 8], color: '#e11d48', ink: '#be123c' },
  { id: 'search', lecs: [10], color: '#4f46e5', ink: '#4338ca' },
] as const
type AreaId = (typeof AREAS)[number]['id']
const areaOf = (m: ModuleInfo) => AREAS.find((a) => (a.lecs as readonly number[]).includes(m.lec))!

/** Short code names in the mono subtitle (the same in every language, like a product line). */
const CODE: Record<ModuleId, string> = {
  numpy: 'BROADCAST LAB', 'bayes-basics': 'BAYES RULE', 'bayes-virus': 'BASE-RATE LAB', 'naive-bayes': 'NAIVE BAYES TABLE', 'gaussian-nb': 'GAUSSIAN NB',
  knn: 'NEIGHBOUR LAB', evaluation: 'CONFUSION MATRIX', 'cross-validation': 'D-FOLD CV', kmeans: 'CLUSTER LAB', 'kmeans-table': 'K-MEANS TABLE',
  perceptron: 'PERCEPTRON TABLE', backprop: 'BACKPROP LAB', 'xor-mlp': 'XOR · MLP', kernel: 'EDGE KERNELS', convolution: 'CONV LAB',
  otsu: 'HISTOGRAM · OTSU', affine: 'AFFINE LAB', 'cnn-shapes': 'SHAPE CALCULATOR', pytorch: 'TENSOR LAB', alphabeta: 'GAME TREE',
}

// ---------------------------------------------------------------- line sketches (320 × 150)
const dash = '4 4'
const box = (x: number, y: number, n: number, m: number, s: number, c: string, hot?: (i: number, j: number) => boolean) =>
  Array.from({ length: n * m }, (_, k) => {
    const i = Math.floor(k / m), j = k % m
    return <rect key={k} x={x + j * s} y={y + i * s} width={s - 3} height={s - 3} rx={2} fill={hot?.(i, j) ? c : 'none'} fillOpacity={0.25} stroke={c} strokeWidth={1.4} />
  })

function Sketch({ id, c }: { id: ModuleId; c: string }) {
  const S: Record<ModuleId, ReactNode> = {
    numpy: (
      <>
        {box(40, 40, 3, 1, 22, c)}
        <text x={82} y={78} fill={c} fontSize={18}>+</text>
        {box(100, 62, 1, 4, 22, c)}
        <path d="M200 75 h22" stroke={c} strokeWidth={1.4} strokeDasharray={dash} />
        {box(234, 40, 3, 4, 18, c, (i, j) => i === 1 && j === 2)}
      </>
    ),
    'bayes-basics': (
      <>
        <circle cx={135} cy={75} r={42} fill="none" stroke={c} strokeWidth={1.6} />
        <circle cx={185} cy={75} r={42} fill="none" stroke={c} strokeWidth={1.6} strokeDasharray={dash} />
        <path d="M160 41 a42 42 0 0 1 0 68 a42 42 0 0 1 0 -68" fill={c} fillOpacity={0.2} />
        <text x={150} y={80} fill={c} fontSize={12} fontFamily="ui-monospace,monospace">A∩B</text>
      </>
    ),
    'bayes-virus': (
      <>
        {Array.from({ length: 60 }, (_, k) => <circle key={k} cx={62 + (k % 15) * 14} cy={40 + Math.floor(k / 15) * 18} r={4} fill={k === 22 || k === 41 ? c : 'none'} stroke={c} strokeWidth={1.2} opacity={k === 22 || k === 41 ? 1 : 0.55} />)}
        <path d="M40 118 h240" stroke={c} strokeWidth={1} strokeDasharray={dash} />
      </>
    ),
    'naive-bayes': (
      <>
        {[0, 1, 2, 3].map((r) => <path key={r} d={`M60 ${40 + r * 20} h130`} stroke={c} strokeWidth={1} opacity={0.6} />)}
        {[0, 1, 2].map((r) => <text key={r} x={70} y={56 + r * 20} fill={c} fontSize={11} fontFamily="ui-monospace,monospace">P(e{r + 1}|B)</text>)}
        <text x={150} y={116} fill={c} fontSize={14}>∏</text>
        <rect x={220} y={50} width={22} height={60} fill={c} fillOpacity={0.25} stroke={c} />
        <rect x={250} y={80} width={22} height={30} fill="none" stroke={c} strokeDasharray={dash} />
      </>
    ),
    'gaussian-nb': (
      <>
        <path d="M40 118 C 100 118, 110 40, 140 40 S 180 118, 240 118" fill="none" stroke={c} strokeWidth={1.6} />
        <path d="M80 118 C 140 118, 160 60, 190 60 S 230 118, 290 118" fill="none" stroke={c} strokeWidth={1.4} strokeDasharray={dash} />
        <path d="M30 118 h270" stroke={c} strokeWidth={1} />
        <path d="M168 118 v-58" stroke={c} strokeWidth={1.2} />
      </>
    ),
    knn: (
      <>
        {[[90, 60], [110, 90], [75, 100], [130, 50], [200, 70], [225, 100], [245, 55], [190, 110]].map(([x, y], k) => (k < 4 ? <circle key={k} cx={x} cy={y} r={5} fill={c} fillOpacity={0.35} stroke={c} /> : <rect key={k} x={x - 5} y={y - 5} width={10} height={10} fill="none" stroke={c} />))}
        <circle cx={160} cy={80} r={4} fill={c} />
        <circle cx={160} cy={80} r={46} fill="none" stroke={c} strokeDasharray={dash} />
      </>
    ),
    evaluation: (
      <>
        {box(115, 30, 2, 2, 46, c, (i, j) => i === j)}
        <text x={124} y={60} fill={c} fontSize={12} fontFamily="ui-monospace,monospace">TP</text>
        <text x={170} y={106} fill={c} fontSize={12} fontFamily="ui-monospace,monospace">TN</text>
      </>
    ),
    'cross-validation': (
      <>
        {[0, 1, 2, 3, 4].map((r) => (
          <g key={r}>
            {[0, 1, 2, 3, 4].map((k) => <rect key={k} x={70 + k * 38} y={30 + r * 19} width={34} height={14} rx={2} fill={k === r ? c : 'none'} fillOpacity={0.3} stroke={c} strokeWidth={1.2} />)}
          </g>
        ))}
      </>
    ),
    kmeans: (
      <>
        {[[80, 60], [95, 75], [70, 85], [170, 45], [185, 60], [160, 55], [240, 100], [255, 85], [230, 110]].map(([x, y], k) => <circle key={k} cx={x} cy={y} r={4} fill="none" stroke={c} strokeWidth={1.3} />)}
        {[[82, 73], [172, 53], [242, 98]].map(([x, y], k) => <path key={k} d={`M${x - 7} ${y} h14 M${x} ${y - 7} v14`} stroke={c} strokeWidth={2} />)}
        <path d="M120 95 C 150 120, 200 120, 215 105" fill="none" stroke={c} strokeDasharray={dash} />
      </>
    ),
    'kmeans-table': (
      <>
        {[0, 1, 2, 3].map((r) => <path key={r} d={`M50 ${35 + r * 22} h220`} stroke={c} strokeWidth={1} opacity={0.5} />)}
        {[90, 150, 210].map((x) => <path key={x} d={`M${x} 30 v80`} stroke={c} strokeWidth={1} opacity={0.5} />)}
        <path d="M120 70 h30 M144 65 l6 5 l-6 5" fill="none" stroke={c} strokeWidth={1.6} />
        <circle cx={240} cy={68} r={6} fill={c} fillOpacity={0.3} stroke={c} />
      </>
    ),
    perceptron: (
      <>
        <path d="M60 120 L 260 30" stroke={c} strokeWidth={1.6} />
        {[[80, 60], [110, 45], [100, 80]].map(([x, y], k) => <circle key={k} cx={x} cy={y} r={5} fill={c} fillOpacity={0.35} stroke={c} />)}
        {[[200, 110], [230, 95], [250, 115]].map(([x, y], k) => <rect key={k} x={x - 5} y={y - 5} width={10} height={10} fill="none" stroke={c} />)}
        <path d="M170 75 l12 -5" stroke={c} strokeDasharray="3 3" />
      </>
    ),
    backprop: (
      <>
        {[[70, 45], [70, 105], [160, 35], [160, 75], [160, 115], [250, 75]].map(([x, y], k) => <circle key={k} cx={x} cy={y} r={9} fill="none" stroke={c} strokeWidth={1.4} />)}
        {[[70, 45, 160, 35], [70, 45, 160, 75], [70, 45, 160, 115], [70, 105, 160, 35], [70, 105, 160, 75], [70, 105, 160, 115], [160, 35, 250, 75], [160, 75, 250, 75], [160, 115, 250, 75]].map(([a, b, d, e], k) => <path key={k} d={`M${a + 9} ${b} L${d - 9} ${e}`} stroke={c} strokeWidth={1} opacity={0.6} />)}
        <path d="M245 130 C 200 140, 120 140, 75 128" fill="none" stroke={c} strokeDasharray={dash} />
      </>
    ),
    'xor-mlp': (
      <>
        <circle cx={100} cy={45} r={6} fill={c} fillOpacity={0.35} stroke={c} />
        <circle cx={220} cy={110} r={6} fill={c} fillOpacity={0.35} stroke={c} />
        <rect x={94} y={104} width={12} height={12} fill="none" stroke={c} />
        <rect x={214} y={39} width={12} height={12} fill="none" stroke={c} />
        <path d="M70 90 C 130 60, 140 120, 250 60" fill="none" stroke={c} strokeWidth={1.4} strokeDasharray={dash} />
      </>
    ),
    kernel: (
      <>
        {box(60, 30, 5, 5, 18, c)}
        <rect x={78} y={48} width={51} height={51} fill={c} fillOpacity={0.18} stroke={c} strokeWidth={2} />
        <path d="M175 75 h25" stroke={c} strokeDasharray={dash} />
        {box(215, 48, 3, 3, 18, c, (i, j) => i === 1 && j === 1)}
      </>
    ),
    convolution: (
      <>
        <rect x={80} y={25} width={110} height={110} fill="none" stroke={c} strokeDasharray={dash} />
        {box(98, 43, 4, 4, 19, c)}
        <rect x={98} y={43} width={54} height={54} fill="none" stroke={c} strokeWidth={2} />
        <path d="M210 80 h25" stroke={c} strokeDasharray={dash} />
        {box(245, 55, 2, 2, 22, c, (i, j) => i === 0 && j === 0)}
      </>
    ),
    otsu: (
      <>
        {[12, 26, 40, 30, 16, 8, 6, 14, 32, 46, 36, 18].map((h, k) => <rect key={k} x={60 + k * 17} y={120 - h * 1.8} width={13} height={h * 1.8} fill={c} fillOpacity={k < 6 ? 0.2 : 0.4} stroke={c} strokeWidth={1} />)}
        <path d="M158 25 v100" stroke={c} strokeWidth={2} />
        <text x={163} y={34} fill={c} fontSize={11} fontFamily="ui-monospace,monospace">T</text>
      </>
    ),
    affine: (
      <>
        <rect x={70} y={45} width={60} height={60} fill="none" stroke={c} strokeDasharray={dash} />
        <path d="M200 105 L 230 45 L 280 55 L 250 115 Z" fill={c} fillOpacity={0.18} stroke={c} strokeWidth={1.6} />
        <path d="M140 75 h45" stroke={c} strokeWidth={1.2} />
        <path d="M180 70 l6 5 l-6 5" fill="none" stroke={c} strokeWidth={1.2} />
      </>
    ),
    'cnn-shapes': (
      <>
        {[[50, 30, 80, 90], [120, 45, 55, 60], [180, 55, 38, 40], [232, 63, 22, 24]].map(([x, y, w, h], k) => (
          <g key={k}>
            <rect x={x} y={y} width={w * 0.55} height={h} fill="none" stroke={c} strokeWidth={1.3} />
            <path d={`M${x} ${y} l${w * 0.3} -10 h${w * 0.55} l-${w * 0.3} 10 M${x + w * 0.55} ${y} l${w * 0.3} -10 v${h} l-${w * 0.3} 10`} fill="none" stroke={c} strokeWidth={1} opacity={0.7} />
          </g>
        ))}
        <path d="M270 75 h20" stroke={c} strokeDasharray={dash} />
      </>
    ),
    pytorch: (
      <>
        {[[70, 45, 'w'], [70, 105, 'b'], [165, 75, 'z'], [255, 75, 'L']].map(([x, y, l], k) => (
          <g key={k}>
            <rect x={(x as number) - 22} y={(y as number) - 14} width={44} height={28} rx={6} fill={k === 3 ? c : 'none'} fillOpacity={0.2} stroke={c} strokeWidth={1.4} />
            <text x={x as number} y={(y as number) + 4} textAnchor="middle" fill={c} fontSize={12} fontFamily="ui-monospace,monospace">{l}</text>
          </g>
        ))}
        <path d="M92 50 L143 70 M92 100 L143 80 M187 75 H233" stroke={c} strokeWidth={1.2} />
        <path d="M233 92 C 200 120, 120 125, 92 112" fill="none" stroke={c} strokeDasharray={dash} />
      </>
    ),
    alphabeta: (
      <>
        <circle cx={160} cy={30} r={8} fill="none" stroke={c} strokeWidth={1.4} />
        {[[100, 75], [220, 75]].map(([x, y], k) => <rect key={k} x={x - 8} y={y - 8} width={16} height={16} fill="none" stroke={c} strokeWidth={1.4} />)}
        <path d="M154 36 L104 67 M166 36 L216 67" stroke={c} strokeWidth={1.2} />
        {[70, 100, 130, 190, 220, 250].map((x, k) => <circle key={k} cx={x} cy={122} r={5} fill={k === 5 ? 'none' : c} fillOpacity={0.3} stroke={c} strokeDasharray={k === 5 ? '2 2' : undefined} />)}
        <path d="M96 83 L70 116 M100 83 V116 M104 83 L130 116 M216 83 L190 116 M220 83 V116" stroke={c} strokeWidth={1} />
        <path d="M224 83 L250 116" stroke={c} strokeWidth={1} strokeDasharray="3 3" />
        <path d="M232 96 l12 6 M244 96 l-12 6" stroke={c} strokeWidth={1.6} />
      </>
    ),
  }
  return (
    <svg viewBox="0 0 320 150" className="h-full w-full" aria-hidden>
      {S[id]}
    </svg>
  )
}

const GRID = { backgroundImage: 'linear-gradient(#e5e9f0 1px, transparent 1px), linear-gradient(90deg, #e5e9f0 1px, transparent 1px)', backgroundSize: '24px 24px' }

function LabCard({ m, seen }: { m: ModuleInfo; seen: boolean }) {
  const { t } = useTranslation()
  const paper = usePaperLabel()
  const area = areaOf(m)
  return (
    <a href={moduleUrl(m.id)} className="group flex flex-col border border-slate-200 bg-white transition hover:-translate-y-0.5 hover:shadow-lg hover:shadow-slate-200" style={{ ['--c' as string]: area.ink }}>
      <div className="relative h-44 border-b border-slate-100 bg-slate-50/60" style={GRID}>
        <span className="absolute left-3 top-3 border bg-white px-1.5 py-0.5 text-[11px] font-bold" style={{ color: area.ink, borderColor: area.color }}>{t(`site.home.lab.areas.${area.id}`)}</span>
        <div className="absolute inset-x-6 bottom-2 top-8">
          <Sketch id={m.id} c={area.color} />
        </div>
      </div>
      <div className="flex flex-1 flex-col gap-2.5 p-5">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <b className="text-[19px] font-black tracking-tight text-slate-900 group-hover:text-[var(--c)]">{t(`site.modules.${m.id}.title`)}</b>
          <span className="font-mono text-[10.5px] font-semibold uppercase tracking-[0.14em] text-slate-500">{CODE[m.id]}</span>
        </div>
        <p className="text-[13.5px] leading-relaxed text-slate-600">{t(`site.modules.${m.id}.blurb`)}</p>
        {m.exams.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {m.exams.slice(0, 3).map((p) => (
              <span key={p} className="border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11.5px] font-semibold text-slate-600">{paper(p)}</span>
            ))}
            {m.exams.length > 3 && <span className="px-1 py-0.5 text-[11.5px] font-semibold text-slate-500">+{m.exams.length - 3}</span>}
          </div>
        )}
        <div className="mt-auto flex items-center justify-between pt-3">
          <span className={`font-mono text-[11px] font-semibold uppercase tracking-[0.18em] ${seen ? 'text-emerald-700' : 'text-slate-500'}`}>{seen ? t('site.home.lab.visited') : t('site.home.lab.ready')}</span>
          <span className="text-[13px] font-bold" style={{ color: area.ink }}>{t('site.home.lab.open')} →</span>
        </div>
      </div>
    </a>
  )
}

/** The toolkit view of the home page. */
export default function HomeLab() {
  const { t } = useTranslation()
  const [area, setArea] = useState<AreaId | 'all'>('all')
  const [visited] = useState(readVisited)
  const shown = MODULES.filter((m) => area === 'all' || areaOf(m).id === area)
  const tab = (id: AreaId | 'all', label: string, n: number, color?: string) => (
    <button
      key={id}
      type="button"
      onClick={() => setArea(id)}
      aria-pressed={area === id}
      className={`flex items-center gap-2 border px-3.5 py-2 text-sm font-bold transition ${area === id ? 'border-blue-600 bg-blue-600 text-white shadow-md shadow-blue-200' : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'}`}
    >
      <i className="inline-block h-2 w-2 rounded-full" style={{ background: area === id ? '#fff' : color ?? '#0f172a' }} aria-hidden />
      {label}
      <span className={`font-mono text-[11px] ${area === id ? 'text-white' : 'text-slate-500'}`}>{n}</span>
    </button>
  )
  return (
    <div className="mx-auto max-w-[1280px] px-4 sm:px-8">
      <h1 className="text-[clamp(26px,3.2vw,36px)] font-black tracking-tight text-slate-900">{t('site.home.lab.title')}</h1>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-5">
        <p className="max-w-3xl text-[15px] text-slate-600">{t('site.home.lab.lead')}</p>
        <span className="flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
          <i className="inline-block h-2 w-2 rounded-full bg-emerald-500" aria-hidden />
          {t('site.home.lab.online', { n: MODULES.length })}
        </span>
      </div>
      <div className="mt-5 flex flex-wrap gap-2" role="group" aria-label={t('site.home.lab.filter')}>
        {tab('all', t('site.home.lab.all'), MODULES.length)}
        {AREAS.map((a) => tab(a.id, t(`site.home.lab.areas.${a.id}`), MODULES.filter((m) => areaOf(m).id === a.id).length, a.color))}
      </div>
      <div className="mt-5 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {shown.map((m) => <LabCard key={m.id} m={m} seen={!!visited[m.id]} />)}
      </div>
    </div>
  )
}
