import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { LazyMotion, MotionConfig, domAnimation } from 'framer-motion'
import '../index.css'
import '../i18n'
import { legacyTarget } from '../lib/sitemap'
import { Shell } from './Shell'
import type { Section } from './Shell'

// Every HTML page loads this entry; <body data-page data-id> says which page to render.
const { page = 'home', id = '' } = document.body.dataset

// Old single-page links (knowcs.online/#/course/knn …) land on the home page: forward them.
const moved = page === 'home' ? legacyTarget(window.location.hash) : null
if (moved) window.location.replace(moved)

const HomePage = lazy(() => import('./pages/HomePage'))
const ModulePage = lazy(() => import('./pages/ModulePage'))
const NotesPage = lazy(() => import('./pages/NotesPage'))
const ExtendPage = lazy(() => import('./pages/ExtendPage'))

const section: Section = page === 'notes' || page === 'note' ? 'notes' : page === 'extend' || page === 'extend-item' ? 'extend' : 'course'

const content =
  page === 'module' ? <ModulePage id={id} /> : page === 'notes' || page === 'note' ? <NotesPage id={id || null} /> : page === 'extend' || page === 'extend-item' ? <ExtendPage id={id || null} /> : <HomePage />

if (!moved) {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <LazyMotion features={domAnimation} strict>
        <MotionConfig reducedMotion="user">
          <Shell section={section}>
            <Suspense fallback={<div className="min-h-[60vh]" aria-busy="true" />}>{content}</Suspense>
          </Shell>
        </MotionConfig>
      </LazyMotion>
    </StrictMode>,
  )
}
