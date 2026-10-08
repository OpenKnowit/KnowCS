import { lazy } from 'react'
import type { ComponentType, LazyExoticComponent } from 'react'
import type { ModuleId } from '../../lib/sitemap'
import { CLASSIC_LOADERS, LAB_LOADERS, PAGE_LOADERS } from '../pageLoaders'

const LAB = Object.fromEntries(Object.entries(LAB_LOADERS).map(([id, load]) => [id, lazy(load as () => Promise<{ default: ComponentType }>)])) as Partial<Record<ModuleId, LazyExoticComponent<ComponentType>>>
const ClassicPage = lazy(PAGE_LOADERS.classic)

export default function ModulePage({ id }: { id: string }) {
  const Lab = LAB[id as ModuleId]
  if (Lab) return <Lab />
  if (CLASSIC_LOADERS[id as ModuleId]) return <ClassicPage id={id as ModuleId} />
  return <p className="p-10 text-center text-slate-500">404</p>
}
