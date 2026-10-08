import { useEffect, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { ArrowRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { noteUrl } from '../../lib/sitemap'
import { LIBS, libFromHash, loadLib } from '../playground/libs'
import type { LibId, LoadedLib } from '../playground/libs'

// /playground/: the library sandboxes (NumPy, matplotlib, pandas, PyTorch, Keras, TensorFlow) on one page, full width.

const GRID = { backgroundImage: 'linear-gradient(#e7ebf1 1px, transparent 1px), linear-gradient(90deg, #e7ebf1 1px, transparent 1px)', backgroundSize: '32px 32px' }

export default function PlaygroundPage() {
  const { t } = useTranslation()
  const [lib, setLib] = useState<LibId>(libFromHash)
  const [loaded, setLoaded] = useState<LoadedLib | null>(null)
  useEffect(() => {
    document.title = `${t('site.playground.title')} · KnowCS`
  }, [t])
  useEffect(() => {
    let live = true
    void loadLib(lib).then((l) => live && setLoaded(l))
    return () => {
      live = false
    }
  }, [lib])
  // back / forward and edited hashes switch the library too (#playground is the full-screen marker, not a library)
  useEffect(() => {
    const onHash = () => {
      if (window.location.hash !== '#playground') setLib(libFromHash())
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const pick = (id: LibId) => {
    setLib(id)
    history.replaceState(null, '', id === 'numpy' ? window.location.pathname : `#${id}`)
  }
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const k = LIBS.findIndex((l) => l.id === lib)
    const next = e.key === 'ArrowRight' ? k + 1 : e.key === 'ArrowLeft' ? k - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? LIBS.length - 1 : null
    if (next === null) return
    e.preventDefault()
    const id = LIBS[(next + LIBS.length) % LIBS.length].id
    pick(id)
    document.getElementById(`lib-${id}`)?.focus()
  }
  const ready = loaded && loaded.id === lib ? loaded : null
  const current = LIBS.find((l) => l.id === lib)!

  return (
    <main className="min-h-screen pb-10 pt-6" style={GRID}>
      <div className="mx-auto max-w-[1280px] px-4 sm:px-8">
        <h1 className="text-[clamp(26px,3.2vw,36px)] font-black tracking-tight text-slate-900">{t('site.playground.title')}</h1>
        <p className="mt-2 max-w-3xl text-[15px] text-slate-600">{t('site.playground.lead')}</p>

        <div className="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6" role="tablist" aria-label={t('site.playground.libs_label')} onKeyDown={onKey}>
          {LIBS.map((l) => {
            const on = l.id === lib
            return (
              <button
                key={l.id}
                id={`lib-${l.id}`}
                type="button"
                role="tab"
                aria-selected={on}
                aria-controls="lib-panel"
                tabIndex={on ? 0 : -1}
                onClick={() => pick(l.id)}
                className={`min-w-0 rounded-2xl border bg-white p-3 text-left transition ${on ? 'border-slate-900 shadow-lg shadow-slate-300/50' : 'border-slate-200 hover:border-slate-300 hover:shadow-sm'}`}
                style={on ? { boxShadow: `inset 0 3px 0 ${l.color}` } : undefined}
              >
                <span className="flex items-center gap-2">
                  <i className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: l.color }} aria-hidden />
                  <b className="text-[15px] font-black text-slate-900">{l.name}</b>
                </span>
                <span className="mt-1 block truncate font-mono text-[11px] text-slate-500">{l.line}</span>
                <span className="mt-1 block text-xs leading-snug text-slate-600">{t(`site.playground.libs.${l.id}`)}</span>
              </button>
            )
          })}
        </div>

        <div className="mt-3 flex justify-end">
          <a href={noteUrl(lib)} className="inline-flex items-center gap-1 text-sm font-bold text-blue-700 hover:underline">
            {t('site.playground.read_note', { name: current.name })} <ArrowRight size={14} aria-hidden />
          </a>
        </div>

        <div id="lib-panel" role="tabpanel" aria-labelledby={`lib-${lib}`} className="mt-3">
          {ready ? (
            ready.kind === 'numpy' ? (
              <ready.View key={lib} wide />
            ) : (
              <ready.View key={lib} config={ready.config} wide />
            )
          ) : (
            <div className="min-h-[640px] animate-pulse rounded-[1.5rem] border border-slate-200 bg-white/80 p-6 text-sm text-slate-500" aria-busy="true">
              {t('playground.ui.loading', { defaultValue: '…' })}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
