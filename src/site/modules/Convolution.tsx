import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { LabError } from '../../lib/labError'
import { convolve, outputSize, pad, type PadMode } from '../../lib/conv2d'
import { errorText, fmt, int } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  img: string
  kernel: string
  pad: number
  mode: PadMode
  stride: number
  flip: boolean
}

const PRESETS: Preset[] = [
  { id: 'lecture7', img: '10 1 3 2 6\n4 3 5 8 0\n8 7 9 6 5', kernel: '-1 0 1\n-1 0 1\n-1 0 1', pad: 0, mode: 'zero', stride: 1, flip: true },
  { id: 'lecture8', img: '5 1 3 7 8\n2 4 6 1 7\n3 4 9 7 2\n4 9 4 5 9\n7 9 8 7 9', kernel: '3 0 0\n0 0 0\n0 0 -3', pad: 1, mode: 'zero', stride: 1, flip: true },
  { id: 'stride', img: '1 3\n2 4', kernel: '2 0 1\n1 3 0\n0 1 2', pad: 1, mode: 'zero', stride: 2, flip: false },
  { id: 'reflect', img: '0 6\n12 18', kernel: '1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9', pad: 1, mode: 'reflect', stride: 1, flip: false },
  { id: 'big', img: '1 2 3 4 5 6 7\n2 3 4 5 6 7 8\n3 4 5 6 7 8 9\n4 5 6 7 8 9 8\n5 6 7 8 9 8 7\n6 7 8 9 8 7 6\n7 8 9 8 7 6 5', kernel: '0 1 0\n1 1 1\n0 1 0', pad: 0, mode: 'zero', stride: 3, flip: false },
]

const GALLERY: { id: string; k: string }[] = [
  { id: 'average', k: '1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9' },
  { id: 'sharpen', k: '-1 -1 -1\n-1 9 -1\n-1 -1 -1' },
  { id: 'prewitt_x', k: '-1 0 1\n-1 0 1\n-1 0 1' },
  { id: 'sobel_x', k: '-1 0 1\n-2 0 2\n-1 0 1' },
  { id: 'sobel_y', k: '-1 -2 -1\n0 0 0\n1 2 1' },
  { id: 'identity', k: '0 0 0\n0 1 0\n0 0 0' },
]

const parseNum = (t: string) => {
  const m = t.match(/^(-?\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/)
  return m ? Number(m[1]) / Number(m[2]) : Number(t)
}
function parseGrid(text: string, what: string): number[][] {
  const rows = text.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => l.split(/[\s,]+/).map(parseNum))
  if (!rows.length) throw new LabError('grid_empty', { what }, `The ${what} is empty.`)
  if (rows.some((r) => r.length !== rows[0].length)) throw new LabError('grid_rows', { what }, `Every ${what} row needs the same number of values.`)
  if (rows.flat().some((v) => !Number.isFinite(v))) throw new LabError('grid_nan', { what }, `The ${what} has a value that is not a number.`)
  return rows
}

const MODES: { v: PadMode; demo: string }[] = [
  { v: 'zero', demo: '0 0 | a b c | 0 0' },
  { v: 'replicate', demo: 'a a | a b c | c c' },
  { v: 'reflect', demo: 'b a | a b c | c b' },
  { v: 'mirror', demo: 'c b | a b c | b a' },
]

function Grid({ cells, cellClass, onPick, sel, label }: { cells: { text: string; cls?: string; title?: string }[][]; cellClass?: string; onPick?: (r: number, c: number) => void; sel?: [number, number]; label: string }) {
  const rows = cells.length
  const cols = cells[0]?.length ?? 1
  return (
    <div className="inline-grid gap-[3px] font-mono text-[13px]" style={{ gridTemplateColumns: `repeat(${cols}, minmax(34px, auto))` }} role="group" aria-label={label}>
      {cells.map((row, r) =>
        row.map((c, j) => {
          const current = !!sel && sel[0] === r && sel[1] === j
          return (
            <button
              key={`${r}-${j}`}
              type="button"
              title={c.title}
              data-cell={`${r},${j}`}
              aria-label={`${label} (${r}, ${j}): ${c.text}`}
              aria-pressed={onPick ? current : undefined}
              onClick={onPick ? () => onPick(r, j) : undefined}
              // pickable grids are one tab stop; arrow keys move the selection (roving tabindex)
              tabIndex={onPick ? (current || (!sel && r === 0 && j === 0) ? 0 : -1) : -1}
              onKeyDown={
                onPick
                  ? (e) => {
                      const d = ({ ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] } as Record<string, [number, number]>)[e.key]
                      if (!d) return
                      e.preventDefault()
                      const nr = Math.min(rows - 1, Math.max(0, r + d[0]))
                      const nc = Math.min(cols - 1, Math.max(0, j + d[1]))
                      onPick(nr, nc)
                      e.currentTarget.parentElement?.querySelector<HTMLButtonElement>(`[data-cell="${nr},${nc}"]`)?.focus()
                    }
                  : undefined
              }
              className={`grid h-[34px] place-items-center rounded-md border px-1 transition ${cellClass ?? ''} ${c.cls ?? 'border-slate-200 bg-slate-50'} ${onPick ? 'cursor-pointer hover:border-blue-400' : 'cursor-default'}`}
            >
              {c.text}
            </button>
          )
        }),
      )}
    </div>
  )
}

export default function Convolution() {
  const { t } = useTranslation()
  const [preset, setPreset] = useState<string | null>('lecture8')
  const p0 = PRESETS[1]
  const [imgText, setImgText] = useState(p0.img)
  const [kText, setKText] = useState(p0.kernel)
  const [padN, setPadN] = useState(p0.pad)
  const [mode, setMode] = useState<PadMode>(p0.mode)
  const [stride, setStride] = useState(p0.stride)
  const [flip, setFlip] = useState(p0.flip)
  const [sel, setSel] = useState<[number, number]>([0, 0])
  const [playing, setPlaying] = useState(false)

  const parsed = useMemo(() => {
    try {
      const img = parseGrid(imgText, t('lab.conv.image'))
      const kernel = parseGrid(kText, t('lab.conv.kernel'))
      if (kernel.length > img.length + 2 * padN || kernel[0].length > img[0].length + 2 * padN) throw new LabError('kernel_big', {}, 'The kernel is larger than the padded image.')
      return { img, kernel, res: convolve(img, kernel, { pad: padN, mode, stride, flip }), error: null as unknown }
    } catch (e) {
      return { error: e, img: null, kernel: null, res: null }
    }
  }, [imgText, kText, padN, mode, stride, flip, t])

  const oh = parsed.res?.out.length ?? 0
  const ow = parsed.res?.out[0]?.length ?? 0
  const r = Math.min(sel[0], Math.max(0, oh - 1))
  const c = Math.min(sel[1], Math.max(0, ow - 1))
  useEffect(() => {
    if (!playing || !oh) return
    const timer = setTimeout(() => {
      const next = r * ow + c + 1
      if (next >= oh * ow) setPlaying(false)
      else setSel([Math.floor(next / ow), next % ow])
    }, 700)
    return () => clearTimeout(timer)
  }, [playing, r, c, oh, ow])

  const custom = () => setPreset(null)
  const load = (p: Preset) => {
    setPreset(p.id)
    setImgText(p.img)
    setKText(p.kernel)
    setPadN(p.pad)
    setMode(p.mode)
    setStride(p.stride)
    setFlip(p.flip)
    setSel([0, 0])
    setPlaying(false)
  }

  const controls = (
    <>
      <Card title={t('lab.common.example')}>
        <Presets items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.conv.presets.${p.id}.title`), note: t(`lab.conv.presets.${p.id}.note`) }))} value={preset} onPick={(id) => load(PRESETS.find((p) => p.id === id)!)} />
      </Card>
      <Card title={t('lab.conv.inputs')} sub={t('lab.conv.inputs_sub')}>
        <div className="grid grid-cols-[1.3fr_1fr] gap-2">
          <label className="grid gap-1">
            <span className="text-xs font-bold text-slate-500">{t('lab.conv.image')}</span>
            <textarea value={imgText} spellCheck={false} onChange={(e) => { setImgText(e.target.value); custom() }} className="min-h-[130px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
          </label>
          <label className="grid gap-1">
            <span className="text-xs font-bold text-slate-500">{t('lab.conv.kernel')}</span>
            <textarea value={kText} spellCheck={false} onChange={(e) => { setKText(e.target.value); custom() }} className="min-h-[130px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
          </label>
        </div>
        <div className="mt-3 grid gap-3">
          <Slider label={t('lab.conv.padding')} value={padN} min={0} max={3} step={1} onChange={(v) => { setPadN(v); custom() }} />
          <Seg label={t('lab.conv.pad_mode')} value={mode} onChange={(v) => { setMode(v); custom() }} options={MODES.map((m) => ({ v: m.v, label: t(`lab.conv.modes.${m.v}`) }))} />
          <Slider label={t('lab.conv.stride')} value={stride} min={1} max={3} step={1} onChange={(v) => { setStride(v); custom() }} />
          <Seg label={t('lab.conv.kernel')} value={flip ? 'flip' : 'raw'} onChange={(v) => { setFlip(v === 'flip'); custom() }} options={[{ v: 'flip', label: t('lab.conv.flip') }, { v: 'raw', label: t('lab.conv.raw') }]} />
        </div>
      </Card>
    </>
  )

  if (!parsed.res || !parsed.img || !parsed.kernel) {
    return (
      <LabPage id="convolution" quiz>
        <Workspace controls={controls}>
          <Note tone="bad" title={t('lab.common.cant_read')}>{errorText(parsed.error, t)}</Note>
        </Workspace>
      </LabPage>
    )
  }

  const { img, kernel, res } = parsed
  const kh = kernel.length
  const kw = kernel[0].length
  const [r0, c0] = res.origin(r, c)
  const sizeH = outputSize(img.length, kh, padN, stride)
  const sizeW = outputSize(img[0].length, kw, padN, stride)
  const products = res.kernel.flatMap((row, i) => row.map((kv, j) => ({ kv, pv: res.padded[r0 + i][c0 + j].value })))
  const posSum = kernel.flat().filter((v) => v > 0).reduce((a, b) => a + b, 0)
  const negSum = kernel.flat().filter((v) => v < 0).reduce((a, b) => a + b, 0)
  const vmax = 255 * posSum
  const vmin = 255 * negSum
  const kstr = (v: number) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : fmt(v, 3))

  return (
    <LabPage id="convolution" quiz lead={t('lab.conv.lead')}>
      <Workspace wide controls={controls}>
        <Card step={1} title={t('lab.conv.slide_title')} sub={t('lab.conv.slide_sub')}>
          <div className="flex flex-wrap items-start gap-6">
            <div>
              <div className="mb-1.5 text-xs font-bold text-slate-500">{t('lab.conv.padded_label', { h: res.padded.length, w: res.padded[0].length })}</div>
              <Grid
                label={t('lab.conv.padded')}
                cells={res.padded.map((row, i) =>
                  row.map((cell, j) => {
                    const inWin = i >= r0 && i < r0 + kh && j >= c0 && j < c0 + kw
                    const base = cell.inside ? 'border-slate-200 bg-slate-50' : 'border-dashed border-slate-300 bg-[repeating-linear-gradient(45deg,#f8fafc,#f8fafc_4px,#eef2f7_4px,#eef2f7_8px)] text-slate-500'
                    return { text: kstr(cell.value), cls: inWin ? 'border-blue-500 bg-blue-50 ring-2 ring-inset ring-blue-500' : base, title: cell.from ? t('lab.conv.copied_from', { at: cell.from.join(', ') }) : t('lab.conv.zero_padding') }
                  }),
                )}
              />
            </div>
            <div>
              <div className="mb-1.5 text-xs font-bold text-slate-500">{flip ? t('lab.conv.kernel_flipped') : t('lab.conv.kernel_raw')}</div>
              <Grid label={t('lab.conv.kernel')} cells={res.kernel.map((row) => row.map((v) => ({ text: kstr(v), cls: 'border-indigo-200 bg-indigo-50 text-indigo-800' })))} />
              {flip && (
                <p className="mt-1.5 text-[11px] text-slate-500">
                  {t('lab.conv.written', { k: kernel.map((row) => row.map(kstr).join(' ')).join(' / ') })}
                </p>
              )}
            </div>
            <div>
              <div className="mb-1.5 text-xs font-bold text-slate-500">{t('lab.conv.output_label', { h: oh, w: ow })}</div>
              <Grid label={t('lab.conv.output')} sel={[r, c]} onPick={(i, j) => { setSel([i, j]); setPlaying(false) }} cells={res.out.map((row, i) => row.map((v, j) => ({ text: kstr(v), cls: i === r && j === c ? 'border-blue-600 bg-blue-600 text-white' : 'border-indigo-200 bg-indigo-50' })))} />
            </div>
          </div>
          <div className="formula mt-4">
            {`out[${r}][${c}] = `}
            {products.map((p, i) => (
              <span key={i} className={p.kv === 0 ? 'text-slate-300' : ''}>
                {i ? ' + ' : ''}
                {kstr(p.kv)}·{kstr(p.pv)}
              </span>
            ))}
            {' = '}
            <b className="text-blue-700">{kstr(res.out[r][c])}</b>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Btn onClick={() => setPlaying((p) => !p)} primary>{playing ? t('lab.common.pause') : t('lab.conv.play_all')}</Btn>
            <Btn onClick={() => { const n = Math.max(0, r * ow + c - 1); setSel([Math.floor(n / ow), n % ow]) }}>{t('lab.common.prev')}</Btn>
            <Btn onClick={() => { const n = Math.min(oh * ow - 1, r * ow + c + 1); setSel([Math.floor(n / ow), n % ow]) }}>{t('lab.common.next')}</Btn>
          </div>
        </Card>

        <div className="grid gap-5 xl:grid-cols-2">
          <Card step={2} title={t('lab.conv.size_title')} sub={t('lab.conv.size_sub')}>
            <div className="formula">
              {`${t('lab.conv.height')} = ⌊(${img.length} − ${kh} + 2·${padN}) / ${stride}⌋ + 1 = `}<b className="text-blue-700">{sizeH.size}</b>
              {`\n${t('lab.conv.width')} = ⌊(${img[0].length} − ${kw} + 2·${padN}) / ${stride}⌋ + 1 = `}<b className="text-blue-700">{sizeW.size}</b>
            </div>
            <div className="mt-3">
              {sizeH.exact && sizeW.exact ? (
                <Note tone="good">{t('lab.conv.fits')}</Note>
              ) : (
                <Note tone="warn" title={t('lab.conv.nofit_title')}>{t(stride > 1 ? 'lab.conv.nofit_many' : 'lab.conv.nofit_one')}</Note>
              )}
            </div>
          </Card>
          <Card step={3} title={t('lab.conv.table_title')} sub={t('lab.common.quiz_sub')}>
            <TableWrap>
              {res.out.map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => <Ans key={j} k={`o${i}_${j}`} v={kstr(Math.round(v * 1000) / 1000)} className={i === r && j === c ? 'win' : ''} />)}
                </tr>
              ))}
            </TableWrap>
            {kernel.flat().some((v) => !Number.isInteger(v)) && <p className="mt-2 text-xs text-slate-500">{t('lab.conv.rounded')}</p>}
          </Card>
        </div>

        <Card step={4} title={t('lab.conv.modes_title')} sub={t('lab.conv.modes_sub')}>
          <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
            {MODES.map((m) => {
              const g = pad(img, 2, m.v)
              return (
                <button key={m.v} type="button" onClick={() => { setMode(m.v); if (padN === 0) setPadN(1); custom() }} className={`rounded-xl border p-3 text-left ${m.v === mode ? 'border-blue-500 bg-blue-50/50' : 'border-slate-200 hover:border-slate-300'}`}>
                  <div className="mb-1 flex items-baseline justify-between gap-2">
                    <b className="text-[13px]">{t(`lab.conv.modes.${m.v}`)}</b>
                    <code className="text-[11px] text-slate-500">{m.demo}</code>
                  </div>
                  <div className="overflow-x-auto">
                    <div className="inline-grid gap-[2px] font-mono text-[11px]" style={{ gridTemplateColumns: `repeat(${g[0].length}, 26px)` }}>
                      {g.flatMap((row, i) => row.map((cell, j) => (
                        <span key={`${i}-${j}`} className={`grid h-[22px] place-items-center rounded ${cell.inside ? 'bg-slate-200 font-bold text-slate-800' : 'bg-slate-50 text-slate-500'}`}>{kstr(cell.value)}</span>
                      )))}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
          <p className="mt-2 text-xs text-slate-500">{t('lab.conv.modes_note')}</p>
        </Card>

        <div className="grid gap-5 xl:grid-cols-2">
          <Card title={t('lab.conv.range_title')} sub="Final 2022 A Q4(a)">
            <div className="formula">
              {`max = 255 × (${t('lab.conv.pos_sum')}) = 255 × ${kstr(posSum)} = `}<b className="text-blue-700">{kstr(vmax)}</b>
              {`\nmin = 255 × (${t('lab.conv.neg_sum')}) = 255 × ${kstr(negSum)} = `}<b className="text-rose-600">{kstr(vmin)}</b>
              {vmax !== vmin && vmin < 0
                ? `\n${t('lab.conv.back_to')}:  I_out = 255 · (I − (${kstr(vmin)})) / (${kstr(vmax)} − (${kstr(vmin)}))`
                : vmax > 255
                  ? `\n${t('lab.conv.back_to')}:  I_out = 255 · I / ${kstr(vmax)}`
                  : `\n${t('lab.conv.inside')}`}
            </div>
          </Card>
          <Card title={t('lab.conv.gallery_title')} sub={t('lab.conv.gallery_sub')}>
            <div className="grid gap-1.5 sm:grid-cols-2">
              {GALLERY.map((g) => (
                <button key={g.id} type="button" onClick={() => { setKText(g.k); custom() }} className="rounded-xl border border-slate-200 px-3 py-2 text-left text-[13px] hover:border-blue-400">
                  <b>{t(`lab.conv.gallery.${g.id}.name`)}</b> <span className="text-slate-500">— {t(`lab.conv.gallery.${g.id}.effect`)}</span>
                </button>
              ))}
            </div>
          </Card>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title={t('lab.conv.flip_title')}>{t('lab.conv.flip_note')}</Note>
          <Note title={t('lab.conv.local_title')}>{t('lab.conv.local_note', { n: int(63) })}</Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
