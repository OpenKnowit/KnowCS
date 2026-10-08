import { useId } from 'react'
import { Cpu, FlaskConical } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { Engine } from './useEngine'
import { useEngineState } from './useEngine'
import { engineSeenReady } from '../../lib/pyodide/client'

// The engine radio: step-through (minipy, draws every step) or real Python (Pyodide, the real libraries).

interface Props {
  engine: Engine
  onChange: (e: Engine) => void
  /** false for PyTorch / Keras / TensorFlow (no Pyodide build): real Python is shown but disabled */
  available: boolean
  /** the library's display name, for the "not available" hint */
  lib: string
}

export function EngineSwitch({ engine, onChange, available, lib }: Props) {
  const { t } = useTranslation()
  const state = useEngineState()
  const name = useId()
  const hint = useId()
  const real = available && engine === 'real'
  const warn = real && state !== 'ready' && !engineSeenReady()
  const options = [
    { id: 'mini' as const, icon: Cpu, label: t('playground.ui.engine.mini'), sub: t('playground.ui.engine.mini_sub'), on: 'border-sky-700 bg-sky-50 text-sky-900', dot: 'bg-sky-700' },
    {
      id: 'real' as const,
      icon: FlaskConical,
      label: t('playground.ui.engine.real'),
      sub: available ? t(`playground.ui.engine.real_${state}`) : t('playground.ui.engine.unavailable', { name: lib }),
      on: 'border-amber-700 bg-amber-50 text-amber-950',
      dot: state === 'ready' ? 'bg-emerald-600' : state === 'failed' ? 'bg-rose-600' : state === 'loading' ? 'animate-pulse bg-amber-500 motion-reduce:animate-none' : 'bg-slate-300',
    },
  ]
  return (
    <fieldset className="min-w-0" aria-describedby={warn ? hint : undefined}>
      <legend className="mb-1 text-[11px] font-bold text-slate-500">{t('playground.ui.engine.label')}</legend>
      <div className="grid grid-cols-2 gap-1.5">
        {options.map((o) => {
          const checked = o.id === 'real' ? real : !real
          const disabled = o.id === 'real' && !available
          return (
            <label
              key={o.id}
              className={`relative flex min-w-0 cursor-pointer items-start gap-2 rounded-xl border px-2.5 py-2 transition has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-blue-400 ${checked ? o.on : 'border-slate-200 bg-white text-slate-700 hover:border-slate-400'} ${disabled ? 'cursor-not-allowed opacity-60 hover:border-slate-200' : ''}`}
            >
              <input type="radio" name={name} value={o.id} checked={checked} disabled={disabled} onChange={() => onChange(o.id)} className="sr-only" />
              <o.icon size={15} className="mt-0.5 shrink-0" aria-hidden />
              <span className="min-w-0">
                <span className="block text-xs font-black">{o.label}</span>
                <span className="mt-0.5 flex items-center gap-1 text-[11px] leading-snug text-slate-600">
                  {o.id === 'real' && available && <i className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${o.dot}`} aria-hidden />}
                  {o.sub}
                </span>
              </span>
            </label>
          )
        })}
      </div>
      {warn && (
        <p id={hint} className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
          {t('playground.ui.real.size')}
        </p>
      )}
    </fieldset>
  )
}
