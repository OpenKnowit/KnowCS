import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import '../index.css'
import i18n from 'i18next'
import { i18nReady } from '../i18n'
import { PAGE_LOADERS, preloadPage } from './pageLoaders'
import { legacyTarget } from '../lib/sitemap'
import { Shell } from './Shell'
import type { Section } from './Shell'
import { countVisit } from './visit'
import { PageBoundary, PageFailure, PageLoading } from './PageState'
import { markReloaded, wouldReloadFor } from './raise'

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
const AboutPage = lazy(PAGE_LOADERS.about)
const CoveragePage = lazy(PAGE_LOADERS.coverage)
const AdminPage = lazy(PAGE_LOADERS.admin)
const NotFoundPage = lazy(PAGE_LOADERS.notfound)

const section: Section =
  page === 'notes' || page === 'note' ? 'notes' : page === 'extend' || page === 'extend-item' ? 'extend' : page === 'watch' || page === 'watch-item' ? 'watch' : page === 'drill' || page === 'formulas' ? 'drill' : page === 'playground' || page === 'coverage' ? 'playground' : page === 'resources' || page === 'admin' ? 'resources' : page === 'about' ? 'about' : 'course'

const content =
  page === 'module' ? <ModulePage id={id} /> : page === 'playground' ? <PlaygroundPage /> : page === 'coverage' ? <CoveragePage /> : page === 'drill' ? <DrillPage /> : page === 'formulas' ? <FormulasPage /> : page === 'papers' ? <PapersPage /> : page === 'resources' ? <ResourcesPage /> : page === 'about' ? <AboutPage /> : page === 'admin' ? <AdminPage /> : page === 'notfound' ? <NotFoundPage /> : page === 'watch' || page === 'watch-item' ? <WatchPage id={id || null} /> : page === 'notes' || page === 'note' ? <NotesPage id={id || null} /> : page === 'extend' || page === 'extend-item' ? <ExtendPage id={id || null} /> : <HomePage />

// cache-first for hashed assets, offline fallback for pages (production only; see src/sw/sw.js)
if (!moved && import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => void navigator.serviceWorker.register('/sw.js').catch(() => {}))
}

if (!moved) {
  // start this page's code and strings now, in parallel with the core strings
  preloadPage(page, id)
  // render once the current language's strings have loaded
  const root = createRoot(document.getElementById('root')!)
  const show = () => {
    // i18next resolves even when the strings did not arrive: without them the page would be a screen of key names
    if (!i18n.hasResourceBundle(i18n.language, 'translation')) throw new TypeError('Failed to fetch the strings')
    countVisit(page, i18n.language)
    root.render(
      <StrictMode>
        {/* the outer boundary is for the header itself; a page that fails keeps the header and shows its message below */}
        <PageBoundary bare>
          <Shell section={section}>
            <PageBoundary>
              <Suspense fallback={<PageLoading />}>{content}</Suspense>
            </PageBoundary>
          </Shell>
        </PageBoundary>
      </StrictMode>,
    )
  }
  void i18nReady.then(show).catch((error: unknown) => {
    console.error(error)
    if (wouldReloadFor(error)) {
      markReloaded()
      window.location.reload()
    } else root.render(<PageFailure kind="load" bare />)
  })
}
