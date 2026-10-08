import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { LazyMotion, MotionConfig } from 'framer-motion'
import '../index.css'
import { i18nReady } from '../i18n'
import { PAGE_LOADERS, preloadPage } from './pageLoaders'
import { legacyTarget } from '../lib/sitemap'
import { Shell } from './Shell'
import type { Section } from './Shell'

// Every HTML page loads this entry; <body data-page data-id> says which page to render.
const { page = 'home', id = '' } = document.body.dataset

// Old single-page links (knowcs.online/#/course/knn …) land on the home page: forward them.
const moved = page === 'home' ? legacyTarget(window.location.hash) : null
if (moved) window.location.replace(moved)

// animation features load on demand, after first paint (m.* components render statically until then)
const loadMotionFeatures = () => import('framer-motion').then((r) => r.domAnimation)

const HomePage = lazy(PAGE_LOADERS.home)
const ModulePage = lazy(PAGE_LOADERS.module)
const NotesPage = lazy(PAGE_LOADERS.notes)
const ExtendPage = lazy(PAGE_LOADERS.extend)
const WatchPage = lazy(PAGE_LOADERS.watch)
const DrillPage = lazy(PAGE_LOADERS.drill)
const FormulasPage = lazy(PAGE_LOADERS.formulas)
const PapersPage = lazy(PAGE_LOADERS.papers)
const NotFoundPage = lazy(PAGE_LOADERS.notfound)

const section: Section =
  page === 'notes' || page === 'note' ? 'notes' : page === 'extend' || page === 'extend-item' ? 'extend' : page === 'watch' || page === 'watch-item' ? 'watch' : page === 'drill' || page === 'formulas' ? 'drill' : 'course'

const content =
  page === 'module' ? <ModulePage id={id} /> : page === 'drill' ? <DrillPage /> : page === 'formulas' ? <FormulasPage /> : page === 'papers' ? <PapersPage /> : page === 'notfound' ? <NotFoundPage /> : page === 'watch' || page === 'watch-item' ? <WatchPage id={id || null} /> : page === 'notes' || page === 'note' ? <NotesPage id={id || null} /> : page === 'extend' || page === 'extend-item' ? <ExtendPage id={id || null} /> : <HomePage />

// cache-first for hashed assets, offline fallback for pages (production only; see src/sw/sw.js)
if (!moved && import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => void navigator.serviceWorker.register('/sw.js').catch(() => {}))
}

if (!moved) {
  // start this page's code and strings now, in parallel with the core strings
  preloadPage(page, id)
  // render once the current language's strings have loaded
  void i18nReady.then(() => {
    createRoot(document.getElementById('root')!).render(
      <StrictMode>
        <LazyMotion features={loadMotionFeatures} strict>
          <MotionConfig reducedMotion="user">
            <Shell section={section}>
              <Suspense fallback={<div className="min-h-[60vh]" aria-busy="true" />}>{content}</Suspense>
            </Shell>
          </MotionConfig>
        </LazyMotion>
      </StrictMode>,
    )
  })
}
