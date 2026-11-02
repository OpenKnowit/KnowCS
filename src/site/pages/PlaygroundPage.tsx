import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowLeft } from 'lucide-react'
import { LIBS, entryFromHash, libFromHash, loadSpec, loadWorkbench, playgroundHash } from '../playground/libs'
import type { LibId } from '../playground/libs'
import type { Specs } from '../playground/Workbench'

// /playground/: one workbench for the library sandboxes (NumPy, matplotlib, pandas, PyTorch, Keras, TensorFlow).
// The library sits in the URL hash. The current one loads first, the rest right after, so switching never waits.

type BenchModule = Awaited<ReturnType<typeof loadWorkbench>>

export default function PlaygroundPage() {
  const { t } = useTranslation()
  const [sel, setSel] = useState<{ lib: LibId; entry: string | null }>(() => ({ lib: libFromHash(), entry: entryFromHash() }))
  const [specs, setSpecs] = useState<Specs>({})
  const [mod, setMod] = useState<BenchModule | null>(null)
  const [side, setSide] = useState(true)
  const [gallery, setGallery] = useState(false)
  // code handed over by a lab page (?code=…): opened once in its library, then dropped from the address bar
  const [seed, setSeed] = useState<{ lib: LibId; code: string } | null>(() => {
    const code = new URLSearchParams(window.location.search).get('code')
    return code ? { lib: libFromHash(), code } : null
  })
  // …and the page it came from (?from=/cnn-shapes/), for the back button; only paths on this site
  const [from] = useState<string | null>(() => {
    const f = new URLSearchParams(window.location.search).get('from')
    return f && /^\/(?!\/)/.test(f) ? f : null
  })
  useEffect(() => {
    if (window.location.search) history.replaceState(null, '', window.location.pathname + window.location.hash)
  }, [])
  const back = () => {
    // the browser's back keeps the lab page as it was left (its edits); a fresh tab has nothing to go back to
    let sameSite = false
    try {
      sameSite = !!document.referrer && new URL(document.referrer).origin === window.location.origin && new URL(document.referrer).pathname === from
    } catch {
      sameSite = false
    }
    if (sameSite && history.length > 1) history.back()
    else window.location.assign(from!)
  }
  useEffect(() => {
    document.title = `${t('site.playground.title')} · KnowCS`
  }, [t])
  useEffect(() => {
    let live = true
    void loadWorkbench().then((m) => live && setMod(m))
    const first = libFromHash()
    void loadSpec(first)
      .then((s) => live && setSpecs((p) => ({ ...p, [first]: s })))
      .then(async () => {
        for (const l of LIBS) {
          if (!live) return
          if (l.id === first) continue
          const s = await loadSpec(l.id)
          if (live) setSpecs((p) => ({ ...p, [l.id]: s }))
        }
      })
    return () => {
      live = false
    }
  }, [])
  // back / forward and edited hashes switch the library too
  useEffect(() => {
    const onHash = () => {
      setSel({ lib: libFromHash(), entry: entryFromHash() })
      setSeed(null)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const pick = useCallback((lib: LibId, entry: string | null) => {
    setSel({ lib, entry })
    setSeed(null)
    void loadSpec(lib).then((s) => setSpecs((p) => (p[lib] ? p : { ...p, [lib]: s })))
    // the address names the example, so it can be shared: /playground/#keras/cnn_summary
    history.replaceState(null, '', lib === 'numpy' && !entry ? window.location.pathname : playgroundHash(lib, entry))
  }, [])
  const closeGallery = useCallback(() => setGallery(false), [])
  // until a library has loaded, the previous one stays on screen (no blank flash in between)
  const [shown, setShown] = useState(sel)
  if (specs[sel.lib] && shown !== sel) setShown(sel)
  const spec = specs[shown.lib]

  return (
    <main>
      <h1 className="sr-only">{t('site.playground.title')}</h1>
      <div className="flex border-t border-slate-200 bg-slate-50 lg:h-[calc(100vh-67px)] lg:min-h-[620px]">
        {mod && side && <mod.ExamplesSidebar lib={shown.lib} entryId={shown.entry} specs={specs} onPick={pick} onCollapse={() => setSide(false)} />}
        {mod && spec ? (
          <mod.default key={shown.lib} lib={shown.lib} spec={spec} entryId={shown.entry} side={side} onSide={setSide} onGallery={() => setGallery(true)} seed={seed?.lib === shown.lib ? seed.code : null} />
        ) : (
          <div className="flex min-h-[620px] flex-1 items-center justify-center text-sm text-slate-500" aria-busy="true">
            {t('playground.ui.loading', { defaultValue: '…' })}
          </div>
        )}
      </div>
      {from && (
        <button
          type="button"
          onClick={back}
          title={from}
          className="fixed bottom-12 left-4 z-40 flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3.5 py-2 text-[13px] font-bold text-slate-700 shadow-lg shadow-slate-900/10 hover:border-blue-300 hover:text-blue-700"
        >
          <ArrowLeft size={15} aria-hidden />
          {t('site.playground.back')}
        </button>
      )}
      {mod && gallery && (
        <mod.Gallery
          lib={shown.lib}
          specs={specs}
          onPick={(lib, entry) => {
            pick(lib, entry)
            setGallery(false)
          }}
          onClose={closeGallery}
        />
      )}
    </main>
  )
}
