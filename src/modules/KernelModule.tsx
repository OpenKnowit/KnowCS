import { useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { Dices, ImageUp, LoaderCircle } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { KERNEL_PRESETS } from '../data/constants'
import { ImageInputError, readRandom, readUpload } from '../lib/imageInput'
import { IMAGE_PRESETS, IMAGE_SIZES, KERNEL_SIZES, convolve, kernelGain, kernelPreset, presetImage, presetNormalized, resample, resizeKernel } from '../lib/kernel'
import type { ImagePreset } from '../lib/kernel'
import type { ActivePreset, KernelPresetName } from '../types'

// The convolution lab: any picture (a built-in one, your own or a random photo) through a 3 × 3, 5 × 5 or 7 × 7 kernel.
// Everything runs in this browser. An imported picture is never sent anywhere: see src/lib/imageInput.ts.

const DISPLAY_PX = 288
type Source = { kind: 'preset'; id: ImagePreset } | { kind: 'photo'; grey: number[][]; origin: 'upload' | 'random' }

/** a grid of grey values, one canvas pixel each, scaled up sharply; cell lines while the pixels are big enough */
function PixelCanvas({ image, cell, label, thumb = false }: { image: number[][]; cell: number; label?: string; thumb?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const n = image.length
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !n) return
    canvas.width = canvas.height = n
    const data = ctx.createImageData(n, n)
    image.forEach((row, r) =>
      row.forEach((v, c) => {
        const i = (r * n + c) * 4
        data.data[i] = data.data[i + 1] = data.data[i + 2] = v
        data.data[i + 3] = 255
      }),
    )
    ctx.putImageData(data, 0, 0)
  }, [image, n])
  const size = n * cell
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <canvas ref={ref} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true} style={{ width: size, height: size, imageRendering: 'pixelated' }} className="block" />
      {!thumb && cell >= 16 && (
        <div aria-hidden className="pointer-events-none absolute inset-0" style={{ backgroundImage: 'linear-gradient(to right, rgb(203 213 225 / 0.8) 1px, transparent 1px), linear-gradient(to bottom, rgb(203 213 225 / 0.8) 1px, transparent 1px)', backgroundSize: `${cell}px ${cell}px` }} />
      )}
    </div>
  )
}

/** keeps what is typed while it is not a number yet, so "-" can start a negative */
function KernelCell({ value, label, small, onChange }: { value: number; label: string; small: boolean; onChange: (v: number) => void }) {
  const [text, setText] = useState<string | null>(null)
  return (
    <input
      type="number"
      value={text ?? value}
      onChange={(e) => {
        setText(e.target.value)
        const n = Number(e.target.value)
        if (e.target.value.trim() !== '' && Number.isFinite(n)) onChange(n)
      }}
      onBlur={() => setText(null)}
      aria-label={label}
      className={`rounded-lg border border-indigo-200 bg-white text-center font-mono font-bold text-indigo-700 px-0 transition [appearance:textfield] focus:outline-none focus:ring-2 focus:ring-indigo-400 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none ${small ? 'h-9 w-9 text-[10.5px]' : 'h-10 w-10 sm:h-12 sm:w-12'}`}
    />
  )
}

const chip = (on: boolean) => `rounded-lg px-3 py-1.5 text-xs font-bold shadow-sm transition-all ${on ? 'bg-indigo-600 text-white shadow-indigo-200' : 'border bg-white text-slate-600 hover:bg-slate-100'}`

export const KernelModule = () => {
  const { t } = useTranslation()
  const [k, setK] = useState<number>(3)
  const [kernel, setKernel] = useState<number[][]>(KERNEL_PRESETS['Sobel-X'])
  const [activePreset, setActivePreset] = useState<ActivePreset>('Sobel-X')
  const [normalize, setNormalize] = useState(false)
  const [resolution, setResolution] = useState<number>(8)
  const [source, setSource] = useState<Source>({ kind: 'preset', id: 'square' })
  const [busy, setBusy] = useState<'upload' | 'random' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  // only the latest request may land; leaving the page cancels a download
  const request = useRef(0)
  const remote = useRef<AbortController | null>(null)
  useEffect(() => () => remote.current?.abort(), [])

  const image = useMemo(() => (source.kind === 'preset' ? presetImage(source.id, resolution) : resample(source.grey, resolution)), [source, resolution])
  const divisor = normalize ? kernelGain(kernel) : 1
  const output = useMemo(() => convolve(image, kernel, divisor), [image, kernel, divisor])
  const cell = DISPLAY_PX / resolution

  const handleKernelChange = (r: number, c: number, val: number) => {
    setKernel(kernel.map((row, ri) => row.map((v, ci) => (ri === r && ci === c ? val : v))))
    setActivePreset('Custom')
  }
  const applyPreset = (name: KernelPresetName, size = k) => {
    setKernel(kernelPreset(name, size))
    setActivePreset(name)
    setNormalize(presetNormalized(name, size))
  }
  const changeSize = (size: number) => {
    setK(size)
    if (activePreset === 'Custom') setKernel(resizeKernel(kernel, size))
    else applyPreset(activePreset, size)
    // a 7 × 7 kernel needs more than 8 pixels a side to show anything
    if (size >= 7 && resolution < 16) setResolution(16)
  }

  /** a photo is unreadable at 8 × 8 */
  const showPhoto = (grey: number[][], origin: 'upload' | 'random') => {
    setSource({ kind: 'photo', grey, origin })
    if (resolution < 64) setResolution(64)
  }
  const load = async (origin: 'upload' | 'random', read: () => Promise<number[][]>) => {
    const id = ++request.current
    setBusy(origin)
    setError(null)
    try {
      const grey = await read()
      if (id === request.current) showPhoto(grey, origin)
    } catch (e) {
      if (id === request.current) setError(t(`kernel_module.err.${e instanceof ImageInputError ? e.code : 'decode'}`))
    } finally {
      if (id === request.current) setBusy(null)
    }
  }
  const onFile = (file: File | undefined) => {
    if (file) void load('upload', () => readUpload(file))
  }
  const onRandom = () => {
    remote.current?.abort()
    const ctl = new AbortController()
    remote.current = ctl
    void load('random', () => readRandom(crypto.getRandomValues(new Uint32Array(1))[0], ctl.signal))
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    onFile(e.dataTransfer.files[0])
  }
  const pickPreset = (id: ImagePreset) => {
    // ignore a download or a decode still under way
    request.current++
    setBusy(null)
    setError(null)
    setSource({ kind: 'preset', id })
  }

  const shapeOf = (n: number) => `(${n}, ${n})`
  const seg = (on: boolean) => `rounded-md px-2.5 py-1 font-mono text-xs font-bold ${on ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`

  return (
    <div className="space-y-6">
      {/* the kernel */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border bg-slate-50 p-4">
        <div className="flex flex-wrap items-center gap-2">
          {(Object.keys(KERNEL_PRESETS) as KernelPresetName[]).map((name) => (
            <button key={name} type="button" onClick={() => applyPreset(name)} aria-pressed={activePreset === name} className={chip(activePreset === name)}>
              {name}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-500">
            {t('kernel_module.kernel_size')}
            <div className="flex gap-0.5 rounded-lg bg-slate-200/70 p-0.5" role="group" aria-label={t('kernel_module.kernel_size')}>
              {KERNEL_SIZES.map((s) => (
                <button key={s} type="button" onClick={() => changeSize(s)} aria-pressed={k === s} className={seg(k === s)}>
                  {s}×{s}
                </button>
              ))}
            </div>
          </div>
          <label className="flex cursor-pointer select-none items-center gap-1.5 text-xs font-bold text-slate-500" title={t('kernel_module.normalize_hint')}>
            <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} className="accent-indigo-600" />
            {t('kernel_module.normalize')}
          </label>
          <div className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
            {t('kernel_module.operation')} <Latex formula="(I * K)_{x,y}" />
          </div>
        </div>
      </div>

      {/* the picture */}
      <div className="rounded-xl border bg-slate-50 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {IMAGE_PRESETS.map((id) => {
              const on = source.kind === 'preset' && source.id === id
              return (
                <button key={id} type="button" onClick={() => pickPreset(id)} aria-pressed={on} className={`flex items-center gap-1.5 ${chip(on)}`}>
                  <span className="overflow-hidden rounded-[3px]">
                    <PixelCanvas image={presetImage(id, 16)} cell={1} thumb />
                  </span>
                  {t(`kernel_module.image.${id}`)}
                </button>
              )
            })}
            <span className="mx-1 hidden h-5 w-px bg-slate-300 sm:block" aria-hidden />
            <button type="button" onClick={() => fileInput.current?.click()} disabled={busy === 'upload'} aria-pressed={source.kind === 'photo' && source.origin === 'upload'} className={`flex items-center gap-1.5 ${chip(source.kind === 'photo' && source.origin === 'upload')}`}>
              {busy === 'upload' ? <LoaderCircle size={14} className="animate-spin" aria-hidden /> : <ImageUp size={14} aria-hidden />}
              {t('kernel_module.upload')}
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/gif,image/bmp"
              className="hidden"
              onChange={(e) => {
                onFile(e.target.files?.[0])
                e.target.value = ''
              }}
            />
            <button type="button" onClick={onRandom} disabled={busy === 'random'} aria-pressed={source.kind === 'photo' && source.origin === 'random'} className={`flex items-center gap-1.5 ${chip(source.kind === 'photo' && source.origin === 'random')}`}>
              {busy === 'random' ? <LoaderCircle size={14} className="animate-spin" aria-hidden /> : <Dices size={14} aria-hidden />}
              {t('kernel_module.random')}
            </button>
          </div>
          <div className="flex items-center gap-2 text-xs font-bold text-slate-500">
            {t('kernel_module.resolution')}
            <div className="flex gap-0.5 rounded-lg bg-slate-200/70 p-0.5" role="group" aria-label={t('kernel_module.resolution')}>
              {IMAGE_SIZES.map((s) => (
                <button key={s} type="button" onClick={() => setResolution(s)} disabled={s <= k} aria-pressed={resolution === s} className={`${seg(resolution === s)} disabled:opacity-30`}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>
        <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
          <Trans i18nKey="kernel_module.privacy" components={{ a: <a href="https://picsum.photos/" target="_blank" rel="noopener noreferrer" className="underline hover:text-slate-700" /> }} />
        </p>
        {error && (
          <p role="alert" className="mt-2 rounded-lg bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700">
            {error}
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
        <div className="flex flex-col items-center">
          <span className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-400">
            {t('kernel_module.input_image')} <Latex formula="I" /> <span className="font-mono normal-case">{shapeOf(image.length)}</span>
          </span>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            title={t('kernel_module.drop_hint')}
            className={`rounded border-2 shadow-sm ${dragging ? 'border-indigo-500 ring-4 ring-indigo-200' : 'border-slate-200'}`}
          >
            <PixelCanvas image={image} cell={cell} label={t('kernel_module.input_label', { n: image.length })} />
          </div>
          <p className="mt-2 max-w-[288px] text-center text-[11px] text-slate-500">{t('kernel_module.drop_hint')}</p>
        </div>

        <div className="relative flex flex-col items-center justify-center">
          <div className="absolute -left-6 top-1/2 hidden -translate-y-1/2 text-3xl font-bold text-slate-200 lg:block">×</div>
          <span className="mb-3 text-xs font-bold uppercase tracking-wider text-slate-400">
            {t('kernel_module.kernel')} <Latex formula="K" /> <span className="font-mono normal-case">{shapeOf(k)}</span>
          </span>
          <div className="grid gap-1.5 rounded-xl border-2 border-indigo-100 bg-indigo-50 p-3 shadow-inner" style={{ gridTemplateColumns: `repeat(${k}, minmax(0, 1fr))` }}>
            {kernel.map((row, ri) =>
              row.map((val, ci) => (
                // a preset replaces the numbers: the key drops any half-typed text
                <KernelCell key={`${k}-${activePreset}-${ri}-${ci}`} value={val} small={k > 3} label={t('kernel_module.cell_label', { r: ri + 1, c: ci + 1 })} onChange={(v) => handleKernelChange(ri, ci, v)} />
              )),
            )}
          </div>
        </div>

        <div className="relative flex flex-col items-center">
          <div className="absolute -left-6 top-1/2 hidden -translate-y-1/2 text-3xl font-bold text-slate-200 lg:block">=</div>
          <span className="mb-3 text-xs font-bold uppercase tracking-wider text-green-600">
            {t('kernel_module.output_feature_map')} <span className="font-mono normal-case">{shapeOf(output.length)}</span>
          </span>
          <div className="rounded border-2 border-slate-200 shadow-sm">
            <PixelCanvas image={output} cell={cell} label={t('kernel_module.output_label', { n: output.length })} />
          </div>
          <p className="mt-2 max-w-[288px] text-center text-[11px] text-slate-500">{t('kernel_module.size_note', { n: image.length, k, out: output.length })}</p>
          <p className="mt-1 max-w-[288px] text-center text-[11px] text-slate-500">{normalize && divisor > 1 ? t('kernel_module.output_note_scaled', { d: divisor }) : t('kernel_module.output_note')}</p>
        </div>
      </div>
      <SeniorAdvice content={
        <Trans i18nKey="kernel_module.advice">
          A convolution kernel... <Latex formula="\text{Laplacian}" /> ...
        </Trans>
      } />
    </div>
  )
}
