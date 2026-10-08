import { BarChart3, GitBranch, Grid3x3, Layers, Table2, TrendingDown } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

// one colour per kind of step, shared by the step strip and each step's header
export type StepKind = 'call' | 'figure' | 'frame' | 'flow' | 'graph' | 'history'
export const KIND: Record<StepKind, { icon: LucideIcon; badge: string; icon_: string; on: string }> = {
  call: { icon: Grid3x3, badge: 'bg-sky-100 text-sky-800', icon_: 'text-sky-600', on: 'border-sky-700 bg-sky-700' },
  figure: { icon: BarChart3, badge: 'bg-teal-100 text-teal-800', icon_: 'text-teal-600', on: 'border-teal-700 bg-teal-700' },
  frame: { icon: Table2, badge: 'bg-indigo-100 text-indigo-800', icon_: 'text-indigo-600', on: 'border-indigo-700 bg-indigo-700' },
  flow: { icon: Layers, badge: 'bg-violet-100 text-violet-800', icon_: 'text-violet-600', on: 'border-violet-700 bg-violet-700' },
  graph: { icon: GitBranch, badge: 'bg-rose-100 text-rose-800', icon_: 'text-rose-600', on: 'border-rose-700 bg-rose-700' },
  history: { icon: TrendingDown, badge: 'bg-emerald-100 text-emerald-800', icon_: 'text-emerald-600', on: 'border-emerald-700 bg-emerald-700' },
}

