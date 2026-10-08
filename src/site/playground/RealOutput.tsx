import { Loader2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { pyodideBase } from '../../lib/pyodide/config'
import { RUN_LIMIT_MS } from '../../lib/pyodide/client'
import type { RealPhase, RealResult } from '../../lib/pyodide/client'
import type { RealRun } from './useEngine'

// What real Python printed (the output box) and drew (the figures), for the "Real Python" engine.

const VERSION_NAMES: Record<string, string> = { python: 'Python', numpy: 'NumPy', pandas: 'pandas', matplotlib: 'matplotlib', sklearn: 'scikit-learn', scipy: 'SciPy', cv2: 'OpenCV', seaborn: 'seaborn' }

const shown = (run: RealRun): RealResult | null => (run.s === 'done' ? run.r : run.s === 'busy' ? run.prev : null)

export function RealConsole({ run }: { run: RealRun }) {
  const { t } = useTranslation()
  const r = shown(run)
  const phaseText = (p: RealPhase) => (p.phase === 'packages' ? t('playground.ui.real.phase_packages', { names: p.names.join(', ') }) : t(`playground.ui.real.phase_${p.phase}`))
  const versions = r ? Object.entries(r.versions).filter(([k]) => k in VERSION_NAMES).map(([k, v]) => `${VERSION_NAMES[k]} ${v}`) : []
  const empty = r && !r.stdout && r.value === null && !r.error && !r.stderr
  return (
    <div aria-live="polite" className="rounded-xl border border-amber-200 bg-slate-900 p-3">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-[11px] font-bold text-slate-400">
        <span>{t('playground.ui.real.output')}</span>
        {r && <span className="font-mono font-normal">{versions.join(' · ')} · {t('playground.ui.real.took', { ms: r.ms })}</span>}
      </div>
      {run.s === 'busy' && (
        <p className="mb-1 flex items-center gap-2 text-xs font-bold text-amber-300">
          <Loader2 size={13} className="animate-spin motion-reduce:animate-none" aria-hidden /> {phaseText(run.phase)}
        </p>
      )}
      {run.s === 'failed' && (
        <p className={`text-xs ${run.why === 'stopped' ? 'text-slate-300' : 'text-rose-300'}`}>
          {run.why === 'load' ? t('playground.ui.real.failed', { host: new URL(pyodideBase()).host }) : run.why === 'timeout' ? t('playground.ui.real.timeout', { s: RUN_LIMIT_MS / 1000 }) : t('playground.ui.real.stopped')}
        </p>
      )}
      {r && (
        <div className={run.s === 'busy' ? 'opacity-50' : ''}>
          {r.stdout && <pre className="max-h-72 overflow-auto whitespace-pre font-mono text-xs leading-5 text-slate-100">{r.stdout}</pre>}
          {r.value !== null && <pre className="max-h-72 overflow-auto whitespace-pre font-mono text-xs leading-5 text-sky-300">{r.value}</pre>}
          {r.stderr && <pre className="max-h-40 overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-amber-300">{r.stderr}</pre>}
          {r.error && <pre className="max-h-72 overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-rose-300">{r.error.trace}</pre>}
          {empty && <p className="text-xs italic text-slate-400">{r.figures.length ? t('playground.ui.real.see_figures') : t('playground.ui.no_output')}</p>}
        </div>
      )}
    </div>
  )
}

export function RealFigures({ run }: { run: RealRun }) {
  const { t } = useTranslation()
  const r = shown(run)
  return (
    <div className="space-y-3">
      {r?.figures.map((src, k) => (
        <img key={k} src={src} alt={t('playground.ui.real.figure', { n: k + 1 })} className={`w-full rounded-xl border border-slate-200 bg-white ${run.s === 'busy' ? 'opacity-50' : ''}`} />
      ))}
      <p className="rounded-xl bg-amber-50 px-3 py-2 text-xs leading-relaxed text-amber-950">{t('playground.ui.engine.steps_off')}</p>
    </div>
  )
}

