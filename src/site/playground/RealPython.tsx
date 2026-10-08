import { useEffect, useRef, useState } from 'react'
import { FlaskConical, Loader2, Play, Square } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { RealLoadError, isRealWarm, runReal, stopReal } from '../../lib/pyodide/client'
import type { RealPhase, RealResult } from '../../lib/pyodide/client'
import { pyodideBase } from '../../lib/pyodide/config'

// "Run in real Python": the same editor code with CPython and the real libraries (Pyodide in a worker), on request only.
// The step-through views stay minipy's; this is the check that the real library prints the same thing.

type State =
  | { s: 'idle' }
  | { s: 'busy'; code: string; phase: RealPhase }
  | { s: 'done'; code: string; r: RealResult }
  | { s: 'failed'; code: string; why: 'load' | 'stopped' }

const VERSION_NAMES: Record<string, string> = { python: 'Python', numpy: 'NumPy', pandas: 'pandas', matplotlib: 'matplotlib', sklearn: 'scikit-learn', scipy: 'SciPy', cv2: 'OpenCV', seaborn: 'seaborn' }

export function RealPython({ code }: { code: string }) {
  const { t } = useTranslation()
  const [st, setSt] = useState<State>({ s: 'idle' })
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  const run = () => {
    const ran = code
    setSt({ s: 'busy', code: ran, phase: { phase: isRealWarm() ? 'running' : 'runtime' } })
    runReal(ran, (phase) => alive.current && setSt({ s: 'busy', code: ran, phase }))
      .then((r) => alive.current && setSt({ s: 'done', code: ran, r }))
      .catch((e: unknown) => alive.current && setSt({ s: 'failed', code: ran, why: e instanceof RealLoadError ? 'load' : 'stopped' }))
  }

  const busy = st.s === 'busy'
  const stale = st.s !== 'idle' && st.s !== 'busy' && st.code !== code
  const phaseText = (p: RealPhase) => (p.phase === 'packages' ? t('playground.ui.real.phase_packages', { names: p.names.join(', ') }) : t(`playground.ui.real.phase_${p.phase}`))

  return (
    <section aria-label={t('playground.ui.real.title')} className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-amber-100 text-amber-800" aria-hidden>
            <FlaskConical size={15} />
          </span>
          <div className="min-w-0">
            <h4 className="text-sm font-black text-slate-900">{t('playground.ui.real.title')}</h4>
            <p className="text-[11px] text-slate-500">{t('playground.ui.real.engine')}</p>
          </div>
        </div>
        {busy ? (
          <button type="button" onClick={() => stopReal()} className="flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-xs font-bold text-slate-700 hover:border-rose-400 hover:text-rose-700">
            <Square size={11} aria-hidden /> {t('playground.ui.real.stop')}
          </button>
        ) : (
          <button type="button" onClick={run} className="flex items-center gap-1.5 rounded-lg bg-amber-700 px-2.5 py-1 text-xs font-bold text-white hover:bg-amber-800">
            <Play size={11} aria-hidden /> {st.s === 'idle' ? t('playground.ui.real.run') : t('playground.ui.real.rerun')}
          </button>
        )}
      </div>

      {st.s === 'idle' && (
        <div className="mt-2 space-y-1 text-xs leading-relaxed text-slate-600">
          <p>{t('playground.ui.real.what')}</p>
          {!isRealWarm() && <p className="text-slate-500">{t('playground.ui.real.size')}</p>}
        </div>
      )}

      <div aria-live="polite">
        {busy && (
          <p className="mt-2 flex items-center gap-2 text-xs font-bold text-amber-800">
            <Loader2 size={13} className="animate-spin motion-reduce:animate-none" aria-hidden /> {phaseText(st.phase)}
          </p>
        )}
        {st.s === 'failed' && (
          <p className={`mt-2 text-xs ${st.why === 'load' ? 'text-rose-700' : 'text-slate-500'}`}>
            {st.why === 'load' ? t('playground.ui.real.failed', { host: new URL(pyodideBase()).host }) : t('playground.ui.real.stopped')}
          </p>
        )}
        {st.s === 'done' && <RealOutput r={st.r} stale={stale} />}
      </div>
    </section>
  )
}

function RealOutput({ r, stale }: { r: RealResult; stale: boolean }) {
  const { t } = useTranslation()
  const versions = Object.entries(r.versions)
    .filter(([k]) => k in VERSION_NAMES)
    .map(([k, v]) => `${VERSION_NAMES[k]} ${v}`)
  const empty = !r.stdout && !r.value && !r.error && !r.figures.length && !r.stderr
  return (
    <div className={`mt-2 space-y-2 ${stale ? 'opacity-60' : ''}`}>
      <p className="font-mono text-[10px] text-slate-500">
        {versions.join(' · ')} · {t('playground.ui.real.took', { ms: r.ms })}
      </p>
      {stale && <p className="text-[11px] font-bold text-slate-600">{t('playground.ui.real.stale')}</p>}
      {(r.stdout || r.value !== null || r.stderr || r.error || empty) && (
        <div className="rounded-lg bg-slate-900 p-2.5">
          {r.stdout && <pre className="max-h-72 overflow-auto whitespace-pre font-mono text-xs leading-5 text-slate-100">{r.stdout}</pre>}
          {r.value !== null && <pre className="max-h-72 overflow-auto whitespace-pre font-mono text-xs leading-5 text-sky-300">{r.value}</pre>}
          {r.stderr && <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-amber-300">{r.stderr}</pre>}
          {r.error && <pre className="max-h-72 overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-rose-300">{r.error.trace}</pre>}
          {empty && <p className="text-xs italic text-slate-400">{t('playground.ui.no_output')}</p>}
        </div>
      )}
      {r.figures.map((src, k) => (
        <img key={k} src={src} alt={t('playground.ui.real.figure', { n: k + 1 })} className="w-full rounded-lg border border-slate-200 bg-white" />
      ))}
    </div>
  )
}
