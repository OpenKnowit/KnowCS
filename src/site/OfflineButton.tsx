import { useEffect, useState } from 'react'
import { CheckCircle2, Download, RefreshCw } from 'lucide-react'
import { useTranslation } from 'react-i18next'

const KEY = 'knowcs-offline'
const read = () => {
  try {
    return localStorage.getItem(KEY)
  } catch {
    return null
  }
}
const write = (v: string) => {
  try {
    localStorage.setItem(KEY, v)
  } catch {
    /* storage blocked: the copy is still saved, we just cannot remember it */
  }
}

type State = { kind: 'idle' } | { kind: 'saving'; done: number; total: number } | { kind: 'saved' } | { kind: 'stale' } | { kind: 'failed' }

/** Footer button: ask the service worker to keep every page and explainer on this device. */
export function OfflineButton() {
  const { t } = useTranslation()
  const supported = import.meta.env.PROD && typeof navigator !== 'undefined' && 'serviceWorker' in navigator
  const [state, setState] = useState<State>(() => (read() ? { kind: 'saved' } : { kind: 'idle' }))
  const [mb, setMb] = useState<number | null>(null)

  useEffect(() => {
    if (!supported) return
    let live = true
    // the build's size, and whether a saved copy is from an older deploy
    fetch('/offline.json', { cache: 'no-store' })
      .then((r) => r.json() as Promise<{ version: string; bytes: number }>)
      .then((m) => {
        if (!live) return
        setMb(Math.max(1, Math.round(m.bytes / 1e6)))
        const saved = read()
        if (saved && saved !== m.version) setState({ kind: 'stale' })
      })
      .catch(() => {})
    const onMessage = (e: MessageEvent<{ type: string; done?: number; total?: number; version?: string | null; failed?: number }>) => {
      if (e.data?.type === 'offline-progress') setState({ kind: 'saving', done: e.data.done ?? 0, total: e.data.total ?? 0 })
      if (e.data?.type === 'offline-done') {
        if (e.data.version && !e.data.failed) {
          write(e.data.version)
          setState({ kind: 'saved' })
        } else setState({ kind: 'failed' })
      }
    }
    navigator.serviceWorker.addEventListener('message', onMessage)
    return () => {
      live = false
      navigator.serviceWorker.removeEventListener('message', onMessage)
    }
  }, [supported])

  if (!supported) return null
  const save = () => {
    setState({ kind: 'saving', done: 0, total: 0 })
    void navigator.serviceWorker.ready.then((reg) => reg.active?.postMessage({ type: 'save-offline' }))
  }
  const base = 'flex items-center gap-2 rounded-full border px-3 py-1.5 font-bold normal-case tracking-normal transition'
  if (state.kind === 'saving')
    return (
      <span className={`${base} border-blue-200 bg-blue-50 text-blue-800`} role="status">
        <RefreshCw size={14} className="animate-spin" aria-hidden />
        {state.total ? t('app.footer.offline_saving', { done: state.done, total: state.total }) : t('app.footer.offline_starting')}
      </span>
    )
  if (state.kind === 'saved')
    return (
      <span className={`${base} border-emerald-200 bg-emerald-50 text-emerald-800`} role="status">
        <CheckCircle2 size={14} aria-hidden /> {t('app.footer.offline_saved')}
      </span>
    )
  return (
    <button type="button" onClick={save} title={mb ? t('app.footer.offline_hint', { mb }) : undefined} className={`${base} border-slate-300 bg-white text-slate-700 hover:border-blue-400 hover:text-blue-700`}>
      {state.kind === 'stale' ? <RefreshCw size={14} aria-hidden /> : <Download size={14} aria-hidden />}
      {state.kind === 'stale' ? t('app.footer.offline_update') : state.kind === 'failed' ? t('app.footer.offline_retry') : t('app.footer.offline_save')}
    </button>
  )
}
