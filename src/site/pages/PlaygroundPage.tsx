import { useEffect, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { ArrowRight, PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { noteUrl } from '../../lib/sitemap'
import { LIBS, libFromHash, loadLib } from '../playground/libs'
import type { LibId, LoadedLib } from '../playground/libs'

// /playground/: the library sandboxes (NumPy, matplotlib, pandas, PyTorch, Keras, TensorFlow) behind a library list
// on the left that folds away, like JupyterLab's side panel.

export default function PlaygroundPage() {
  const { t } = useTranslation()
  const [lib, setLib] = useState<LibId>(libFromHash)
  const [loaded, setLoaded] = useState<LoadedLib | null>(null)
  const [side, setSide] = useState(true)
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
    const next = e.key === 'ArrowDown' ? k + 1 : e.key === 'ArrowUp' ? k - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? LIBS.length - 1 : null
    if (next === null) return
    e.preventDefault()
    const id = LIBS[(next + LIBS.length) % LIBS.length].id
    pick(id)
    document.getElementById(`lib-${id}`)?.focus()
  }
  const ready = loaded && loaded.id === lib ? loaded : null
  const current = LIBS.find((l) => l.id === lib)!

  return (
    <main className="mx-auto flex max-w-[1480px] items-start gap-5 px-4 pb-10 pt-5 sm:px-6">
      {side ? (
        <aside className="sticky top-4 hidden w-[248px] shrink-0 rounded-2xl border border-slate-200 bg-white md:block">
          <div className="flex items-center justify-between py-2.5 pl-4 pr-2.5">
            <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-slate-500">{t('site.playground.libraries')}</span>
            <button type="button" onClick={() => setSide(false)} className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900" aria-label={t('site.playground.hide_libs')} title={t('site.playground.hide_libs')}>
              <PanelLeftClose size={16} aria-hidden />
            </button>
          </div>
          <div className="grid gap-0.5 px-2 pb-2" role="tablist" aria-orientation="vertical" aria-label={t('site.playground.libs_label')} onKeyDown={onKey}>
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
                  className={`min-w-0 rounded-lg px-3 py-2 text-left transition ${on ? 'bg-blue-50' : 'hover:bg-slate-50'}`}
                >
                  <span className="flex items-center gap-2">
                    <i className="inline-block h-2 w-2 shrink-0 rounded-full" style={{ background: l.color }} aria-hidden />
                    <b className={`text-[13px] ${on ? 'font-bold text-blue-700' : 'font-semibold text-slate-700'}`}>{l.name}</b>
                  </span>
                  <span className="mt-0.5 block truncate pl-4 font-mono text-[11px] text-slate-500">{l.line}</span>
                </button>
              )
            })}
          </div>
          <a href={noteUrl(lib)} className="flex items-center gap-1 border-t border-slate-200 px-4 py-3 text-[13px] font-semibold text-blue-700 hover:underline">
            {t('site.playground.read_note', { name: current.name })} <ArrowRight size={14} aria-hidden />
          </a>
        </aside>
      ) : (
        <button type="button" onClick={() => setSide(true)} className="sticky top-4 hidden h-9 w-9 shrink-0 place-items-center rounded-[9px] border border-slate-200 bg-white text-slate-600 hover:text-slate-900 md:grid" aria-label={t('site.playground.show_libs')} title={t('site.playground.show_libs')}>
          <PanelLeftOpen size={18} aria-hidden />
        </button>
      )}

      <div className="min-w-0 flex-1">
        <h1 className="sr-only">{t('site.playground.title')}</h1>
        {/* phones get the library list as a row of chips above the sandbox */}
        <div className="mb-3 flex gap-1.5 overflow-x-auto md:hidden">
          {LIBS.map((l) => (
            <button key={l.id} type="button" onClick={() => pick(l.id)} aria-pressed={l.id === lib} className={`shrink-0 rounded-full border px-3 py-1.5 text-[13px] font-semibold ${l.id === lib ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-200 bg-white text-slate-600'}`}>
              {l.name}
            </button>
          ))}
        </div>
        <div id="lib-panel" role="tabpanel" aria-labelledby={`lib-${lib}`}>
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
