import type { ReactNode } from 'react'
import { LazyMotion, MotionConfig } from 'framer-motion'

// The animation engine arrives after first paint (domAnimation is its own chunk).
const loadFeatures = () => import('framer-motion').then((r) => r.domAnimation)

/**
 * Wraps the parts of the site that animate with m.* (classic modules, notes, the extension page), so pages that
 * never animate do not download Framer Motion at all.
 */
export function Motion({ children }: { children: ReactNode }) {
  return (
    <LazyMotion features={loadFeatures} strict>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </LazyMotion>
  )
}
