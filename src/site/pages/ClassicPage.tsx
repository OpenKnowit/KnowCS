import type { ComponentType, LazyExoticComponent } from 'react'
import { lazy } from 'react'
import { Motion } from '../../components/Motion'
import type { ModuleId } from '../../lib/sitemap'
import { CLASSIC_LOADERS } from '../pageLoaders'
import { PageBar } from '../ui'

const COMPONENTS = Object.fromEntries(Object.entries(CLASSIC_LOADERS).map(([id, load]) => [id, lazy(load!)])) as Partial<Record<ModuleId, LazyExoticComponent<ComponentType>>>

/** The original modules: the bar, then the module itself across the full width. */
export default function ClassicPage({ id }: { id: ModuleId }) {
  const Component = COMPONENTS[id]!
  return (
    <>
      <PageBar id={id} />
      <main className="px-4 pb-10 pt-5 sm:px-6">
        <Motion>
          <Component />
        </Motion>
      </main>
    </>
  )
}
