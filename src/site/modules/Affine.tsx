import { useMemo, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { applyPoint, doableBy3x3, properties, reach, reflectX, reflectY, rotate, scale, shear, translate, warpAffine } from '../../lib/affine'
import type { Affine as M23 } from '../../lib/affine'
import { fmt } from '../format'
import { Ans, Card, LabPage, Note, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

// --- the test picture: an "F" and a dot on an 8 × 10 grid (rows ≠ cols, so rows − 1 and cols − 1 differ) ---
const ROWS = 8
const COLS = 10
const IMG: number[][] = Array.from({ length: ROWS }, (_, y) =>
  Array.from({ length: COLS }, (_, x) => {
    if (x === 2 && y >= 1 && y <= 6) return 230 // stem
    if (y === 1 && x >= 3 && x <= 6) return 180 // top bar
    if (y === 3 && x >= 3 && x <= 5) return 120 // middle bar
    if (x === 8 && y === 6) return 90 // dot
    return 40
  }),
)
const X0 = Math.floor(COLS / 2)
const Y0 = Math.floor(ROWS / 2)

type Kind = 'translate' | 'reflectX' | 'reflectY' | 'rotate' | 'scale' | 'shear'
const KINDS: Kind[] = ['translate', 'reflectX', 'reflectY', 'rotate', 'scale', 'shear']

const num = (v: number) => fmt(Math.abs(v) < 1e-9 ? 0 : v, 3)
const code = (M: M23) => `[[${M[0].map(num).join(', ')}], [${M[1].map(num).join(', ')}]]`

/** Image-coordinate pixel grid. `pick` makes the cells buttons. */
function Grid({ img, mark, ring, hatch, pick, label, centre }: { img: number[][]; mark?: [number, number] | null; ring: string; hatch?: (x: number, y: number) => boolean; pick?: (x: number, y: number) => void; label: string; centre?: boolean }) {
  const { t } = useTranslation()
  const cols = img[0].length
  return (
    <div role="group" aria-label={label}>
      <div className="grid gap-px font-mono text-[10px] text-slate-500" style={{ gridTemplateColumns: `16px repeat(${cols}, minmax(0, 1fr))` }}>
        <span className="text-center italic">y\x</span>
        {img[0].map((_, x) => <span key={x} className="text-center">{x}</span>)}
        {img.map((row, y) => [
          <span key={`r${y}`} className="grid place-items-center">{y}</span>,
          ...row.map((v, x) => {
            const on = mark && mark[0] === x && mark[1] === y
            const style: CSSProperties = { background: hatch?.(x, y) ? 'repeating-linear-gradient(45deg,#0f172a 0 3px,#334155 3px 6px)' : `rgb(${v},${v},${v})` }
            const cls = `relative aspect-square rounded-[3px] ${on ? `z-10 ring-[3px] ${ring}` : ''}`
            const dot = centre && x === X0 && y === Y0 ? <i className="absolute inset-0 m-auto h-1.5 w-1.5 rounded-full bg-rose-500" aria-hidden /> : null
            return pick ? (
              <button key={x} type="button" className={`${cls} focus-visible:outline-2 focus-visible:outline-blue-500`} style={style} onClick={() => pick(x, y)} aria-pressed={!!on} aria-label={t('lab.affine.pixel', { x, y })}>{dot}</button>
            ) : (
              <span key={x} className={cls} style={style}>{dot}</span>
            )
          }),
        ])}
      </div>
    </div>
  )
}

/** The 2 × 2 part acting on a lattice and a right triangle — what survives, drawn. */
function Lattice({ M }: { M: M23 }) {
  const { t } = useTranslation()
  const S = 22, W = 260, H = 220, cx = W / 2, cy = H / 2
  const lin = (x: number, y: number): [number, number] => [M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y]
  const P = (p: [number, number]) => `${cx + p[0] * S},${cy + p[1] * S}`
  const lines: ReactNode[] = []
  for (let k = -8; k <= 8; k++) {
    lines.push(<line key={`a${k}`} x1={cx + k * S} y1={0} x2={cx + k * S} y2={H} stroke="#e2e8f0" />, <line key={`b${k}`} x1={0} y1={cy + k * S} x2={W} y2={cy + k * S} stroke="#e2e8f0" />)
    const [a1, a2, b1, b2] = [lin(k, -12), lin(k, 12), lin(-12, k), lin(12, k)]
    lines.push(<line key={`c${k}`} x1={cx + a1[0] * S} y1={cy + a1[1] * S} x2={cx + a2[0] * S} y2={cy + a2[1] * S} stroke="#93c5fd" strokeWidth={k ? 1 : 2} />, <line key={`d${k}`} x1={cx + b1[0] * S} y1={cy + b1[1] * S} x2={cx + b2[0] * S} y2={cy + b2[1] * S} stroke="#93c5fd" strokeWidth={k ? 1 : 2} />)
  }
  const A: [number, number] = [0, 0], B: [number, number] = [3, 0], C: [number, number] = [0, -2]
  const [a, b, c] = [lin(...A), lin(...B), lin(...C)]
  const len = (p: [number, number], q: [number, number]) => Math.hypot(p[0] - q[0], p[1] - q[1])
  const ang = (Math.acos(Math.max(-1, Math.min(1, ((b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1])) / (len(a, b) * len(a, c) || 1)))) * 180) / Math.PI
  return (
    <figure className="grid gap-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[320px] rounded-xl border border-slate-200 bg-white" role="img" aria-label={t('lab.affine.lattice_aria')}>
        {lines}
        <polygon points={[A, B, C].map(P).join(' ')} fill="none" stroke="#94a3b8" strokeDasharray="4 3" strokeWidth={1.5} />
        <polygon points={[a, b, c].map(P).join(' ')} fill="#2563eb22" stroke="#2563eb" strokeWidth={2.5} />
      </svg>
      <figcaption className="font-mono text-[12px] leading-relaxed text-slate-600">
        |AB| 3 → {fmt(len(a, b), 2)} · |AC| 2 → {fmt(len(a, c), 2)} · ∠A 90° → {fmt(ang, 1)}°
      </figcaption>
    </figure>
  )
}

// --- augmentation: does the label survive? ---
type Aug = 'hflip' | 'vflip' | 'rot180' | 'rot15' | 'shift' | 'zoom' | 'bright'
const AUGS: { id: Aug; css: string; filter?: string; cat: boolean; digit: boolean }[] = [
  { id: 'hflip', css: 'scaleX(-1)', cat: true, digit: false },
  { id: 'vflip', css: 'scaleY(-1)', cat: false, digit: false },
  { id: 'rot180', css: 'rotate(180deg)', cat: false, digit: false },
  { id: 'rot15', css: 'rotate(15deg)', cat: true, digit: true },
  { id: 'shift', css: 'translate(10%, 6%)', cat: true, digit: true },
  { id: 'zoom', css: 'scale(1.25)', cat: true, digit: true },
  { id: 'bright', css: 'none', filter: 'brightness(1.6)', cat: true, digit: true },
]
const IDENTITY = { ...AUGS[0], css: 'none' }

const Cat = () => (
  <g fill="#334155">
    <ellipse cx="58" cy="68" rx="24" ry="19" />
    <circle cx="34" cy="42" r="14" />
    <polygon points="22,34 24,16 33,29" />
    <polygon points="36,28 45,15 47,34" />
    <rect x="34" y="70" width="7" height="20" rx="3" />
    <rect x="48" y="74" width="7" height="16" rx="3" />
    <path d="M80 76 C 97 70, 98 48, 86 40" fill="none" stroke="#334155" strokeWidth="6" strokeLinecap="round" />
    <circle cx="28" cy="40" r="2.4" fill="#fbbf24" />
  </g>
)

function Thumb({ children, aug, label }: { children: ReactNode; aug: (typeof AUGS)[number]; label: string }) {
  return (
    <svg viewBox="0 0 100 100" className="aspect-square w-full overflow-hidden rounded-xl border border-slate-200 bg-amber-50" role="img" aria-label={label}>
      <g style={{ transform: aug.css, transformOrigin: '50px 50px', filter: aug.filter, transition: 'transform .5s ease, filter .5s ease' }}>{children}</g>
    </svg>
  )
}

const Digit = ({ d }: { d: string }) => (
  <text x="50" y="74" textAnchor="middle" fontSize="70" fontWeight="700" fill="#1e293b" fontFamily="'Comic Sans MS','Chalkboard SE','Segoe Print',cursive">{d}</text>
)

export default function AffinePage() {
  const { t } = useTranslation()
  const [kind, setKind] = useState<Kind>('reflectY')
  const [tx, setTx] = useState(2)
  const [ty, setTy] = useState(1)
  const [deg, setDeg] = useState(90)
  const [s, setS] = useState(1.5)
  const [k, setK] = useState(0.5)
  const [pick, setPick] = useState<[number, number]>([7, 1])
  const [aug, setAug] = useState<Aug>('hflip')

  const M: M23 = useMemo(() => {
    switch (kind) {
      case 'translate': return translate(tx, ty)
      case 'reflectX': return reflectX(ROWS)
      case 'reflectY': return reflectY(COLS)
      case 'rotate': return rotate(deg, X0, Y0)
      case 'scale': return scale(s, s, X0, Y0)
      case 'shear': return shear(k)
    }
  }, [kind, tx, ty, deg, s, k])
  const { out, from } = useMemo(() => warpAffine(IMG, M), [M])
  const src = from[pick[1]][pick[0]]
  const props = properties(M)
  const far = reach(from)
  const by3 = doableBy3x3(M)
  const corners: [number, number][] = [[0, 0], [COLS - 1, 0], [0, ROWS - 1], [COLS - 1, ROWS - 1]]
  const a = AUGS.find((x) => x.id === aug)!
  const yes = (v: boolean) => <b className={v ? 'text-emerald-700' : 'text-rose-700'}>{v ? '✓' : '✗'}</b>

  return (
    <LabPage id="affine" quiz lead={t('lab.affine.lead')}>
      <Workspace
        wide
        controls={
          <>
            <Card title={t('lab.affine.transform')}>
              <Presets items={KINDS.map((id) => ({ id, title: t(`lab.affine.kinds.${id}.title`), note: t(`lab.affine.kinds.${id}.note`) }))} value={kind} onPick={setKind} />
            </Card>
            {(kind === 'translate' || kind === 'rotate' || kind === 'scale' || kind === 'shear') && (
              <Card title={t('lab.common.settings')}>
                <div className="grid gap-3">
                  {kind === 'translate' && (
                    <>
                      <Slider label={<>t<sub>x</sub></>} value={tx} min={-5} max={5} step={1} onChange={setTx} />
                      <Slider label={<>t<sub>y</sub></>} value={ty} min={-4} max={4} step={1} onChange={setTy} />
                    </>
                  )}
                  {kind === 'rotate' && <Slider label="θ" value={deg} min={-180} max={180} step={15} onChange={setDeg} display={`${deg}°`} />}
                  {kind === 'scale' && <Slider label="s" value={s} min={0.5} max={2} step={0.25} onChange={setS} />}
                  {kind === 'shear' && <Slider label="k" value={k} min={-1} max={1} step={0.25} onChange={setK} />}
                  {kind === 'rotate' && <p className="text-xs text-slate-500">{t('lab.affine.rotate_dir')}</p>}
                </div>
              </Card>
            )}
            <Card title={t('lab.affine.matrix')}>
              <div className="formula !text-[13px]">M = {code(M)}</div>
              <pre className="mt-2 overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] leading-relaxed text-slate-100">
{`rows, cols = img.shape   # ${ROWS}, ${COLS}
M = np.float32(${code(M)})
out = cv2.warpAffine(img, M, (cols, rows))`}
              </pre>
              <p className="mt-2 text-xs text-slate-500">{t('lab.affine.dsize')}</p>
            </Card>
          </>
        }
      >
        <Card step={1} title={t('lab.affine.pixels_title')} sub={t('lab.affine.pixels_sub')}>
          <div className="grid gap-5 md:grid-cols-2">
            <div className="grid gap-2">
              <div className="text-xs font-extrabold uppercase tracking-wider text-slate-500">{t('lab.affine.before')}</div>
              <Grid img={IMG} mark={src} ring="ring-amber-400" label={t('lab.affine.before')} centre={kind === 'rotate' || kind === 'scale'} />
            </div>
            <div className="grid gap-2">
              <div className="text-xs font-extrabold uppercase tracking-wider text-slate-500">{t('lab.affine.after')}</div>
              <Grid img={out} mark={pick} ring="ring-blue-500" hatch={(x, y) => !from[y][x]} pick={(x, y) => setPick([x, y])} label={t('lab.affine.after')} />
            </div>
          </div>
          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            <Note>
              {src ? (
                <Trans i18nKey="lab.affine.came_from" values={{ x: pick[0], y: pick[1], sx: src[0], sy: src[1], v: IMG[src[1]][src[0]] }} components={{ 1: <b className="text-blue-700" />, 3: <b className="text-amber-700" /> }} />
              ) : (
                <Trans i18nKey="lab.affine.outside" values={{ x: pick[0], y: pick[1] }} components={{ 1: <b className="text-blue-700" /> }} />
              )}
            </Note>
            {src && (
              <div className="formula !text-[12.5px] leading-relaxed">
                x′ = {num(M[0][0])}·{src[0]} + {num(M[0][1])}·{src[1]} + {num(M[0][2])} = {num(applyPoint(M, ...src)[0])}
                <br />
                y′ = {num(M[1][0])}·{src[0]} + {num(M[1][1])}·{src[1]} + {num(M[1][2])} = {num(applyPoint(M, ...src)[1])}
              </div>
            )}
          </div>
          <p className="mt-3 text-[13px] text-slate-600">{t('lab.affine.legend')}</p>
        </Card>

        <Card step={2} title={t('lab.affine.corners_title')} sub={t('lab.common.quiz_sub')}>
          <TableWrap>
            <tr><th>P (x, y)</th><th>x′</th><th>y′</th></tr>
            {corners.map(([x, y]) => {
              const [px, py] = applyPoint(M, x, y)
              return (
                <tr key={`${x},${y}`}>
                  <td>({x}, {y})</td>
                  <Ans k={`x_${x}_${y}`} v={num(px)} />
                  <Ans k={`y_${x}_${y}`} v={num(py)} />
                </tr>
              )
            })}
          </TableWrap>
          <p className="mt-2 text-xs text-slate-500">{t('lab.affine.corners_note')}</p>
        </Card>

        <div className="grid gap-5 xl:grid-cols-2">
          <Card step={3} title={t('lab.affine.keeps_title')} sub="Final 2024 Q1(f)">
            <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-1 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <Lattice M={M} />
              <TableWrap>
                <tr><th className="left">{t('lab.affine.property')}</th><th>{t('lab.affine.kept')}</th></tr>
                <tr><td className="left font-sans">{t('lab.affine.props.parallel')}</td><td>{yes(true)}</td></tr>
                <tr><td className="left font-sans">{t('lab.affine.props.distances')}</td><td>{yes(props.distances)}</td></tr>
                <tr><td className="left font-sans">{t('lab.affine.props.angles')}</td><td>{yes(props.angles)}</td></tr>
                <tr><td className="left font-sans">{t('lab.affine.props.handed')}</td><td>{yes(!props.mirrored)}</td></tr>
              </TableWrap>
            </div>
            <div className="mt-3"><Note tone="good">{t('lab.affine.keeps_note')}</Note></div>
          </Card>

          <Card step={4} title={t('lab.affine.local_title')} sub="Final 2022 Q1(f) · Final 2024 Q5(e)">
            <div className="grid gap-3">
              <div className="flex flex-wrap gap-3">
                <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
                  <div className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">{t('lab.affine.reach')}</div>
                  <div className="font-mono text-[22px] font-extrabold text-slate-900">{t('lab.affine.reach_px', { n: far })}</div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
                  <div className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">{t('lab.affine.by3')}</div>
                  <div className={`font-mono text-[22px] font-extrabold ${by3 ? 'text-emerald-700' : 'text-rose-700'}`}>{by3 ? t('lab.affine.yes') : t('lab.affine.no')}</div>
                </div>
              </div>
              <Note>{t(by3 ? 'lab.affine.by3_yes' : 'lab.affine.by3_no')}</Note>
              <Note tone="warn" title={t('lab.common.exam_traps')}>{t('lab.affine.not_point')}</Note>
            </div>
          </Card>
        </div>

        <Card step={5} title={t('lab.affine.aug_title')} sub="Final 2022 B Q1(b)">
          <Seg value={aug} onChange={setAug} label={t('lab.affine.aug_pick')} options={AUGS.map((x) => ({ v: x.id, label: t(`lab.affine.augs.${x.id}.name`) }))} />
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {(['cat', 'digit'] as const).map((who) => (
              <div key={who} className="grid gap-2">
                <div className="grid grid-cols-3 gap-2">
                  {who === 'cat' ? (
                    <>
                      <Thumb aug={IDENTITY} label={t('lab.affine.cat_orig')}><Cat /></Thumb>
                      <span className="grid place-items-center text-2xl font-black text-slate-400" aria-hidden>→</span>
                      <Thumb aug={a} label={t('lab.affine.cat_aug', { aug: t(`lab.affine.augs.${a.id}.name`) })}><Cat /></Thumb>
                    </>
                  ) : (
                    ['6', '2', '7'].map((d) => <Thumb key={d} aug={a} label={t('lab.affine.digit_aug', { d, aug: t(`lab.affine.augs.${a.id}.name`) })}><Digit d={d} /></Thumb>)
                  )}
                </div>
                <div className={`rounded-xl border px-3 py-2 text-[13.5px] leading-relaxed ${a[who] ? 'border-emerald-200 bg-emerald-50 text-emerald-950' : 'border-rose-200 bg-rose-50 text-rose-950'}`}>
                  <b>{t(`lab.affine.${who}_task`)}: {a[who] ? t('lab.affine.keeps_label') : t('lab.affine.breaks_label')}</b> — {t(`lab.affine.augs.${a.id}.${who}`)}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4"><Note tone="warn" title={t('lab.common.exam_pattern')}>{t('lab.affine.aug_exam')}</Note></div>
        </Card>
      </Workspace>
    </LabPage>
  )
}
