import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import { i18nReady } from '../i18n'
import { PAGE_LOADERS, preloadPage } from './pageLoaders'
import { legacyTarget } from '../lib/sitemap'
import { Shell } from './Shell'
import type { Section } from './Shell'

// iOS Safari zooms the whole page when an input smaller than 16px gets focus, and many of the labs' number boxes are.
// maximum-scale stops that; iOS still lets people pinch to zoom, which is why this is not set for other browsers.
if (/iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)) {
  document.querySelector('meta[name="viewport"]')?.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1')
}

// Every HTML page loads this entry; <body data-page data-id> says which page to render.
const { page = 'home', id = '' } = document.body.dataset

// Old single-page links (knowcs.online/#/course/knn …) land on the home page: forward them.
const moved = page === 'home' ? legacyTarget(window.location.hash) : null
if (moved) window.location.replace(moved)


const HomePage = lazy(PAGE_LOADERS.home)
const PlaygroundPage = lazy(PAGE_LOADERS.playground)
const ModulePage = lazy(PAGE_LOADERS.module)
const NotesPage = lazy(PAGE_LOADERS.notes)
const ExtendPage = lazy(PAGE_LOADERS.extend)
const WatchPage = lazy(PAGE_LOADERS.watch)
const DrillPage = lazy(PAGE_LOADERS.drill)
const FormulasPage = lazy(PAGE_LOADERS.formulas)
const PapersPage = lazy(PAGE_LOADERS.papers)
const ResourcesPage = lazy(PAGE_LOADERS.resources)
const AdminPage = lazy(PAGE_LOADERS.admin)
const NotFoundPage = lazy(PAGE_LOADERS.notfound)

const section: Section =
  page === 'notes' || page === 'note' ? 'notes' : page === 'extend' || page === 'extend-item' ? 'extend' : page === 'watch' || page === 'watch-item' ? 'watch' : page === 'drill' || page === 'formulas' ? 'drill' : page === 'playground' ? 'playground' : page === 'resources' || page === 'admin' ? 'resources' : 'course'

const content =
  page === 'module' ? <ModulePage id={id} /> : page === 'playground' ? <PlaygroundPage /> : page === 'drill' ? <DrillPage /> : page === 'formulas' ? <FormulasPage /> : page === 'papers' ? <PapersPage /> : page === 'resources' ? <ResourcesPage /> : page === 'admin' ? <AdminPage /> : page === 'notfound' ? <NotFoundPage /> : page === 'watch' || page === 'watch-item' ? <WatchPage id={id || null} /> : page === 'notes' || page === 'note' ? <NotesPage id={id || null} /> : page === 'extend' || page === 'extend-item' ? <ExtendPage id={id || null} /> : <HomePage />

// cache-first for hashed assets, offline fallback for pages (production only; see src/sw/sw.js)
if (!moved && import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => void navigator.serviceWorker.register('/sw.js').catch(() => {}))
}

if (!moved) {
  // start this page's code and strings now, in parallel with the core strings
  preloadPage(page, id)
  // real Python: once the page is up, download the runtime into its cache and start the worker, so the
  // playground's first real run is quick (not while the page is only being prerendered, not on Save-Data)
  const warmPython = () => {
    if ((navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData) return
    setTimeout(() => void import('../lib/pyodide/client').then((m) => m.warmReal()), 1500)
  }
  const afterLoad = () => (document.readyState === 'complete' ? warmPython() : window.addEventListener('load', warmPython, { once: true }))
  if ((document as Document & { prerendering?: boolean }).prerendering) document.addEventListener('prerenderingchange', afterLoad, { once: true })
  else afterLoad()
  // render once the current language's strings have loaded
  void i18nReady.then(() => {
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <Shell section={section}>
          <Suspense fallback={<div className="min-h-screen" aria-busy="true" />}>{content}</Suspense>
        </Shell>
      </StrictMode>,
    )
  })
}
