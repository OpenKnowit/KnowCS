import { useEffect, useMemo, useState } from 'react'
import { convolve, outputSize, pad, type PadMode } from '../../lib/conv2d'
import { fmt, int } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  title: string
  note: string
  img: string
  kernel: string
  pad: number
  mode: PadMode
  stride: number
  flip: boolean
}

const PRESETS: Preset[] = [
  { id: 'lecture7', title: 'Lecture 7: edge kernel', note: '3×5 image, [−1 0 1] kernel, no padding', img: '10 1 3 2 6\n4 3 5 8 0\n8 7 9 6 5', kernel: '-1 0 1\n-1 0 1\n-1 0 1', pad: 0, mode: 'zero', stride: 1, flip: true },
  { id: 'lecture8', title: 'Lecture 8 practice: flip the kernel', note: '5×5, zero padding 1 — first row 12 18 3 21 0', img: '5 1 3 7 8\n2 4 6 1 7\n3 4 9 7 2\n4 9 4 5 9\n7 9 8 7 9', kernel: '3 0 0\n0 0 0\n0 0 -3', pad: 1, mode: 'zero', stride: 1, flip: true },
  { id: 'stride', title: 'Exam-style: stride 2, padding 1', note: 'Pattern of Final 2022 B Q1(a) — flipped vs not', img: '1 3\n2 4', kernel: '2 0 1\n1 3 0\n0 1 2', pad: 1, mode: 'zero', stride: 2, flip: false },
  { id: 'reflect', title: 'Exam-style: reflect, then average', note: 'Pattern of Final 2022 A Q4(b) / Final 2024 Q5(c)', img: '0 6\n12 18', kernel: '1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9', pad: 1, mode: 'reflect', stride: 1, flip: false },
  { id: 'big', title: '7×7 with stride 3', note: 'Lecture 8: “does not fit”', img: '1 2 3 4 5 6 7\n2 3 4 5 6 7 8\n3 4 5 6 7 8 9\n4 5 6 7 8 9 8\n5 6 7 8 9 8 7\n6 7 8 9 8 7 6\n7 8 9 8 7 6 5', kernel: '0 1 0\n1 1 1\n0 1 0', pad: 0, mode: 'zero', stride: 3, flip: false },
]

const GALLERY: { name: string; effect: string; k: string }[] = [
  { name: 'Average', effect: 'smoothing / blur', k: '1/9 1/9 1/9\n1/9 1/9 1/9\n1/9 1/9 1/9' },
  { name: 'Sharpen', effect: 'boosts the centre against its neighbours', k: '-1 -1 -1\n-1 9 -1\n-1 -1 -1' },
  { name: 'Prewitt x', effect: 'vertical edges (left–right change)', k: '-1 0 1\n-1 0 1\n-1 0 1' },
  { name: 'Sobel x', effect: 'vertical edges, centre row weighted 2', k: '-1 0 1\n-2 0 2\n-1 0 1' },
  { name: 'Sobel y', effect: 'horizontal edges', k: '-1 -2 -1\n0 0 0\n1 2 1' },
  { name: 'Identity', effect: 'does nothing', k: '0 0 0\n0 1 0\n0 0 0' },
]

const parseNum = (t: string) => {
  const m = t.match(/^(-?\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/)
  return m ? Number(m[1]) / Number(m[2]) : Number(t)
}
function parseGrid(text: string, what: string): number[][] {
  const rows = text.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => l.split(/[\s,]+/).map(parseNum))
  if (!rows.length) throw new Error(`The ${what} is empty.`)
  if (rows.some((r) => r.length !== rows[0].length)) throw new Error(`Every ${what} row needs the same number of values.`)
  if (rows.flat().some((v) => !Number.isFinite(v))) throw new Error(`The ${what} has a value that is not a number.`)
  return rows
}

const MODES: { v: PadMode; label: string; demo: string }[] = [
  { v: 'zero', label: 'Zero', demo: '0 0 | a b c | 0 0' },
  { v: 'replicate', label: 'Replicate', demo: 'a a | a b c | c c' },
  { v: 'reflect', label: 'Reflect', demo: 'b a | a b c | c b' },
  { v: 'mirror', label: 'Mirror', demo: 'c b | a b c | b a' },
]

function Grid({ cells, cellClass, onPick, label }: { cells: { text: string; cls?: string; title?: string }[][]; cellClass?: string; onPick?: (r: number, c: number) => void; label: string }) {
  return (
    <div className="inline-grid gap-[3px] font-mono text-[13px]" style={{ gridTemplateColumns: `repeat(${cells[0]?.length ?? 1}, minmax(34px, auto))` }} role="grid" aria-label={label}>
      {cells.map((row, r) =>
        row.map((c, j) => (
          <button
            key={`${r}-${j}`}
            type="button"
            title={c.title}
            onClick={onPick ? () => onPick(r, j) : undefined}
            tabIndex={onPick ? 0 : -1}
            className={`grid h-[34px] place-items-center rounded-md border px-1 transition ${cellClass ?? ''} ${c.cls ?? 'border-slate-200 bg-slate-50'} ${onPick ? 'cursor-pointer hover:border-blue-400' : 'cursor-default'}`}
          >
            {c.text}
          </button>
        )),
      )}
    </div>
  )
}

export default function Convolution() {
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
      const img = parseGrid(imgText, 'image')
      const kernel = parseGrid(kText, 'kernel')
      if (kernel.length > img.length + 2 * padN || kernel[0].length > img[0].length + 2 * padN) throw new Error('The kernel is larger than the padded image.')
      return { img, kernel, res: convolve(img, kernel, { pad: padN, mode, stride, flip }), error: null }
    } catch (e) {
      return { error: (e as Error).message, img: null, kernel: null, res: null }
    }
  }, [imgText, kText, padN, mode, stride, flip])

  const oh = parsed.res?.out.length ?? 0
  const ow = parsed.res?.out[0]?.length ?? 0
  const r = Math.min(sel[0], Math.max(0, oh - 1))
  const c = Math.min(sel[1], Math.max(0, ow - 1))
  useEffect(() => {
    if (!playing || !oh) return
    const t = setTimeout(() => {
      const next = r * ow + c + 1
      if (next >= oh * ow) setPlaying(false)
      else setSel([Math.floor(next / ow), next % ow])
    }, 700)
    return () => clearTimeout(t)
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
      <Card title="Example">
        <Presets items={PRESETS} value={preset} onPick={(id) => load(PRESETS.find((p) => p.id === id)!)} />
      </Card>
      <Card title="Image and kernel" sub="numbers or fractions like 1/9">
        <div className="grid grid-cols-[1.3fr_1fr] gap-2">
          <label className="grid gap-1">
            <span className="text-xs font-bold text-slate-500">Image</span>
            <textarea value={imgText} spellCheck={false} onChange={(e) => { setImgText(e.target.value); custom() }} className="min-h-[130px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
          </label>
          <label className="grid gap-1">
            <span className="text-xs font-bold text-slate-500">Kernel</span>
            <textarea value={kText} spellCheck={false} onChange={(e) => { setKText(e.target.value); custom() }} className="min-h-[130px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
          </label>
        </div>
        <div className="mt-3 grid gap-3">
          <Slider label="Padding p" value={padN} min={0} max={3} step={1} onChange={(v) => { setPadN(v); custom() }} />
          <Seg label="Padding mode" value={mode} onChange={(v) => { setMode(v); custom() }} options={MODES.map((m) => ({ v: m.v, label: m.label }))} />
          <Slider label="Stride s" value={stride} min={1} max={3} step={1} onChange={(v) => { setStride(v); custom() }} />
          <Seg label="Kernel" value={flip ? 'flip' : 'raw'} onChange={(v) => { setFlip(v === 'flip'); custom() }} options={[{ v: 'flip', label: 'Flip 180° (convolution)' }, { v: 'raw', label: 'As written (correlation)' }]} />
        </div>
      </Card>
    </>
  )

  if (!parsed.res || !parsed.img || !parsed.kernel) {
    return (
      <LabPage id="convolution" quiz>
        <Workspace controls={controls}>
          <Note tone="bad" title="Can’t read the input.">{parsed.error}</Note>
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
    <LabPage id="convolution" quiz lead="Slide the kernel over the padded image, multiply cell by cell, add up. Then answer the three things exams always ask: what padding does at the border, whether the kernel is flipped, and how big the output is.">
      <Workspace wide controls={controls}>
        <Card step={1} title="Slide the window" sub="click any output cell, or play">
          <div className="flex flex-wrap items-start gap-6">
            <div>
              <div className="mb-1.5 text-xs font-bold text-slate-500">Padded input ({res.padded.length}×{res.padded[0].length}) — hatched = padding</div>
              <Grid
                label="Padded input"
                cells={res.padded.map((row, i) =>
                  row.map((cell, j) => {
                    const inWin = i >= r0 && i < r0 + kh && j >= c0 && j < c0 + kw
                    const base = cell.inside ? 'border-slate-200 bg-slate-50' : 'border-dashed border-slate-300 bg-[repeating-linear-gradient(45deg,#f8fafc,#f8fafc_4px,#eef2f7_4px,#eef2f7_8px)] text-slate-500'
                    return { text: kstr(cell.value), cls: inWin ? 'border-blue-500 bg-blue-50 ring-2 ring-inset ring-blue-500' : base, title: cell.from ? `copied from (${cell.from.join(', ')})` : 'zero padding' }
                  }),
                )}
              />
            </div>
            <div>
              <div className="mb-1.5 text-xs font-bold text-slate-500">Kernel {flip ? 'after flipping' : 'as written'}</div>
              <Grid label="Kernel" cells={res.kernel.map((row) => row.map((v) => ({ text: kstr(v), cls: 'border-indigo-200 bg-indigo-50 text-indigo-800' })))} />
              {flip && (
                <p className="mt-1.5 text-[11px] text-slate-400">
                  written: {kernel.map((row) => row.map(kstr).join(' ')).join(' / ')}
                </p>
              )}
            </div>
            <div>
              <div className="mb-1.5 text-xs font-bold text-slate-500">Output ({oh}×{ow})</div>
              <Grid label="Output" onPick={(i, j) => { setSel([i, j]); setPlaying(false) }} cells={res.out.map((row, i) => row.map((v, j) => ({ text: kstr(v), cls: i === r && j === c ? 'border-blue-600 bg-blue-600 text-white' : 'border-indigo-200 bg-indigo-50' })))} />
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
            <Btn onClick={() => setPlaying((p) => !p)} primary>{playing ? 'Pause' : 'Play all positions'}</Btn>
            <Btn onClick={() => { const n = Math.max(0, r * ow + c - 1); setSel([Math.floor(n / ow), n % ow]) }}>← Prev</Btn>
            <Btn onClick={() => { const n = Math.min(oh * ow - 1, r * ow + c + 1); setSel([Math.floor(n / ow), n % ow]) }}>Next →</Btn>
          </div>
        </Card>

        <div className="grid gap-5 xl:grid-cols-2">
          <Card step={2} title="Output size" sub="lecture 8 formula">
            <div className="formula">
              {`height = ⌊(${img.length} − ${kh} + 2·${padN}) / ${stride}⌋ + 1 = `}<b className="text-blue-700">{sizeH.size}</b>
              {`\nwidth  = ⌊(${img[0].length} − ${kw} + 2·${padN}) / ${stride}⌋ + 1 = `}<b className="text-blue-700">{sizeW.size}</b>
            </div>
            <div className="mt-3">
              {sizeH.exact && sizeW.exact ? (
                <Note tone="good">The window lands exactly on the last column and row.</Note>
              ) : (
                <Note tone="warn" title="Does not fit exactly.">
                  (n − k + 2p) is not divisible by the stride, so the last {stride > 1 ? 'pixels are' : 'pixel is'} never covered. Exams floor the result — the lecture calls 7×7 with a 3×3 kernel and stride 3 “does not fit”.
                </Note>
              )}
            </div>
          </Card>
          <Card step={3} title="Output table" sub="Turn on “Quiz me”">
            <TableWrap>
              {res.out.map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => <Ans key={j} k={`o${i}_${j}`} v={kstr(Math.round(v * 1000) / 1000)} className={i === r && j === c ? 'win' : ''} />)}
                </tr>
              ))}
            </TableWrap>
            {kernel.flat().some((v) => !Number.isInteger(v)) && <p className="mt-2 text-xs text-slate-400">Final 2022 rounded averages to integers — check what the paper says.</p>}
          </Card>
        </div>

        <Card step={4} title="The four padding modes on this image" sub="padding 2 so the pattern is visible">
          <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">
            {MODES.map((m) => {
              const g = pad(img, 2, m.v)
              return (
                <button key={m.v} type="button" onClick={() => { setMode(m.v); if (padN === 0) setPadN(1); custom() }} className={`rounded-xl border p-3 text-left ${m.v === mode ? 'border-blue-500 bg-blue-50/50' : 'border-slate-200 hover:border-slate-300'}`}>
                  <div className="mb-1 flex items-baseline justify-between gap-2">
                    <b className="text-[13px]">{m.label}</b>
                    <code className="text-[11px] text-slate-400">{m.demo}</code>
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
          <p className="mt-2 text-xs text-slate-400">Final 2022 and Final 2024 called the edge-repeating version “reflection padding”. If a paper is ambiguous, write which one you used.</p>
        </Card>

        <div className="grid gap-5 xl:grid-cols-2">
          <Card title="Range of this kernel on an 8-bit image" sub="Final 2022 A Q4(a)">
            <div className="formula">
              {`max = 255 × (sum of positive weights) = 255 × ${kstr(posSum)} = `}<b className="text-blue-700">{kstr(vmax)}</b>
              {`\nmin = 255 × (sum of negative weights) = 255 × ${kstr(negSum)} = `}<b className="text-rose-600">{kstr(vmin)}</b>
              {vmax !== vmin && vmin < 0
                ? `\nback to 0…255:  I_out = 255 · (I − (${kstr(vmin)})) / (${kstr(vmax)} − (${kstr(vmin)}))`
                : vmax > 255
                  ? `\nback to 0…255:  I_out = 255 · I / ${kstr(vmax)}`
                  : '\nalready inside 0…255'}
            </div>
          </Card>
          <Card title="Name that kernel" sub="Final 2024 Q5(d) — click to load">
            <div className="grid gap-1.5 sm:grid-cols-2">
              {GALLERY.map((g) => (
                <button key={g.name} type="button" onClick={() => { setKText(g.k); custom() }} className="rounded-xl border border-slate-200 px-3 py-2 text-left text-[13px] hover:border-blue-400">
                  <b>{g.name}</b> <span className="text-slate-500">— {g.effect}</span>
                </button>
              ))}
            </div>
          </Card>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title="Flip or not?">
            Lecture 7’s formula O(x,y) = Σ K(m,n)·I(x−m, y−n) is true convolution, which rotates the kernel by 180°; lecture 8’s practice says “don’t forget to flip”. Final 2022 accepted both answers (73 unflipped, 27 flipped). Symmetric kernels give the same result either way.
          </Note>
          <Note title="Local, not global.">
            A 3×3 kernel only sees a 3×3 neighbourhood, so no kernel can flip a 64×64 image — a pixel’s new value would have to come from up to {int(63)} pixels away (Final 2024 Q5e).
          </Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
