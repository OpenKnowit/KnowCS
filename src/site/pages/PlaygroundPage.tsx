import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { libFromHash, loadLib } from '../playground/libs'
import type { LibId, LoadedLib } from '../playground/libs'

// /playground/: one workbench for the library sandboxes (NumPy, matplotlib, pandas, PyTorch, Keras, TensorFlow).
// The library sits in the URL hash; the workbench lists its examples on the left.

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
  // back / forward and edited hashes switch the library too
  useEffect(() => {
    const onHash = () => setLib(libFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const pick = (id: LibId) => {
    setLib(id)
    history.replaceState(null, '', id === 'numpy' ? window.location.pathname : `#${id}`)
  }
  const ready = loaded && loaded.id === lib ? loaded : null

  return (
    <main>
      <h1 className="sr-only">{t('site.playground.title')}</h1>
      {ready ? (
        <ready.Bench key={lib} lib={lib} spec={ready.spec} side={side} onSide={setSide} onPickLib={pick} />
      ) : (
        <div className="flex min-h-[620px] items-center justify-center border-t border-slate-200 bg-slate-50 text-sm text-slate-500 lg:h-[calc(100vh-67px)]" aria-busy="true">
          {t('playground.ui.loading', { defaultValue: '…' })}
        </div>
      )}
    </main>
  )
}
