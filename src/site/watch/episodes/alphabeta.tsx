/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { useTranslation } from 'react-i18next'
import { eseg, seg } from '../../../lib/explainer'
import { generateAbSteps, leavesEvaluated, minimaxValue, treeWithLeaves } from '../../../lib/alphabeta'
import { AB_PRESETS } from '../../../data/constants'
import type { AbNode, AbStep } from '../../../types'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, TitleCard } from '../stage'

// "Alpha-beta: skipping what cannot matter" — lecture 10; asked in the 2022, 2023 and 2024 finals.

const TREE = treeWithLeaves(AB_PRESETS.alpha) // 6 7 8 9 1 2 0 4 — prunes one leaf and one whole subtree
const STEPS = generateAbSteps(TREE)
const ROOT = minimaxValue(TREE)
const WORST = treeWithLeaves([0, 4, 1, 2, 8, 9, 6, 7])
const WORST_STEPS = generateAbSteps(WORST)

const IDS = Object.keys(TREE)
const X = (n: AbNode) => 110 + n.x * 1.0
const Y = (n: AbNode) => 190 + n.y * 1.4
const inf = (v: number) => (v === Infinity ? '+∞' : v === -Infinity ? '−∞' : String(v))

/** Minimax values filled bottom-up: depth 3 = leaves, then 2, 1, 0 as `level` rises. */
function minimaxAt(tree: Record<string, AbNode>, level: number): Record<string, number | null> {
  const depth = (id: string): number => (tree[id].parent ? 1 + depth(tree[id].parent!) : 0)
  return Object.fromEntries(IDS.map((id) => [id, 3 - depth(id) < level ? minimaxValue(tree, id) : null]))
}

function TreeView({ tree, values, step, hl, showAB }: { tree: Record<string, AbNode>; values: Record<string, number | null>; step?: AbStep; hl?: string | null; showAB?: boolean }) {
  const pruned = new Set(step?.prunedNodes ?? [])
  const prunedE = new Set(step?.prunedEdges ?? [])
  const visited = new Set(step?.visited ?? IDS)
  return (
    <g>
      {IDS.map((id) => {
        const n = tree[id]
        if (!n.parent) return null
        const p = tree[n.parent]
        const key = `${n.parent}->${id}`
        const active = step?.activeEdge === key
        return <line key={key} x1={X(p)} y1={Y(p)} x2={X(n)} y2={Y(n)} stroke={prunedE.has(key) ? C.red : active ? C.yellow : visited.has(id) ? '#4b5b78' : C.grid} strokeWidth={active ? 5 : 3} strokeDasharray={prunedE.has(key) ? '8 7' : undefined} />
      })}
      {IDS.map((id) => {
        const n = tree[id]
        const isLeaf = n.type === 'leaf'
        const on = visited.has(id)
        const cut = pruned.has(id)
        const stroke = cut ? C.red : id === hl ? C.yellow : n.type === 'max' ? C.blue : n.type === 'min' ? C.purple : C.muted
        return (
          <g key={id} opacity={cut ? 0.45 : on ? 1 : 0.45}>
            {n.type === 'max' && <polygon points={`${X(n)},${Y(n) - 34} ${X(n) + 36},${Y(n) + 24} ${X(n) - 36},${Y(n) + 24}`} fill={C.bg} stroke={stroke} strokeWidth={4} />}
            {n.type === 'min' && <polygon points={`${X(n)},${Y(n) + 34} ${X(n) + 36},${Y(n) - 24} ${X(n) - 36},${Y(n) - 24}`} fill={C.bg} stroke={stroke} strokeWidth={4} />}
            {isLeaf && <rect x={X(n) - 26} y={Y(n) - 26} width={52} height={52} rx={8} fill={C.bg} stroke={stroke} strokeWidth={3} />}
            {cut && <line x1={X(n) - 30} y1={Y(n) - 30} x2={X(n) + 30} y2={Y(n) + 30} stroke={C.red} strokeWidth={4} />}
            {!isLeaf && <text x={X(n) - 50} y={Y(n) + 6} textAnchor="end" fill={C.muted} fontSize={22} fontWeight={700}>{id}</text>}
          </g>
        )
      })}
      {showAB &&
        step &&
        IDS.filter((id) => tree[id].type !== 'leaf' && visited.has(id) && !pruned.has(id)).map((id) => {
          const n = tree[id]
          return (
            <text key={'ab' + id} x={X(n) + 46} y={Y(n) - 4} fill={C.muted} fontSize={21} fontFamily="ui-monospace, monospace">
              <tspan fill={C.blue}>α {inf(step.nodeAlphas[id])}</tspan>
              <tspan x={X(n) + 46} dy={24} fill={C.purple}>β {inf(step.nodeBetas[id])}</tspan>
            </text>
          )
        })}
      {IDS.map((id) => {
        const n = tree[id]
        const v = n.type === 'leaf' ? (visited.has(id) || !step ? n.value : null) : values[id]
        if (v === null || v === undefined) return null
        return (
          <text key={'v' + id} x={X(n)} y={Y(n) + (n.type === 'max' ? 14 : n.type === 'min' ? 0 : 9)} textAnchor="middle" fill={pruned.has(id) ? C.red : C.text} fontSize={26} fontWeight={800} fontFamily="ui-monospace, monospace">
            {v}
          </text>
        )
      })}
    </g>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.alphabeta.kicker')} title={t('watch.alphabeta.title')} sub={t('watch.alphabeta.sub')} />
}

function TreeScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <g opacity={eseg(p, 0, 0.15)}>
          <TreeView tree={TREE} values={{}} />
        </g>
      </Svg>
      {[['max', 274, C.blue], ['min', 428, C.purple], ['max', 582, C.blue]].map(([k, y, col], i) => (
        <At key={i} x={1230} y={Number(y)} size={26} anchor="l" color={String(col)} o={eseg(p, 0.25 + i * 0.1, 0.35 + i * 0.1)}>
          {t(`watch.alphabeta.label.${k}_row`)}
        </At>
      ))}
      <At x={1230} y={750} size={26} anchor="l" color={C.muted} o={eseg(p, 0.55, 0.65)}>{t('watch.alphabeta.label.leaf_row')}</At>
    </>
  )
}

function MinimaxScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const level = Math.floor(seg(p, 0.05, 0.75) * 4.99)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <TreeView tree={TREE} values={minimaxAt(TREE, level)} />
      </Svg>
      <At x={1330} y={300} size={30} w={460} o={eseg(p, 0.1, 0.2)} style={{ textAlign: 'center' }}>{t('watch.alphabeta.label.bubble')}</At>
      <At x={1330} y={520} size={60} className="font-black font-mono" color={C.yellow} o={eseg(p, 0.75, 0.85)}>{ROOT}</At>
      <At x={1330} y={600} size={26} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.alphabeta.label.root_value')}</At>
    </>
  )
}

function SearchScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const k = Math.min(STEPS.length - 1, Math.floor(seg(p, 0.03, 0.92) * STEPS.length))
  const s = STEPS[k]
  const leaves = s.visited.filter((id) => TREE[id].type === 'leaf').length
  const prunes = STEPS.slice(0, k + 1).filter((x) => x.type === 'prune')
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <TreeView tree={TREE} values={s.nodeValues} step={s} hl={s.nodeId} showAB />
      </Svg>
      <At x={1330} y={200} size={26} color={C.muted}>{t('watch.alphabeta.label.leaves_seen')}</At>
      <At x={1330} y={260} size={56} className="font-black font-mono">{leaves} / 8</At>
      {prunes.map((x, i) => (
        <At key={i} x={1330} y={400 + i * 120} size={24} w={440} color={C.red} style={{ textAlign: 'center' }}>
          {t('watch.alphabeta.label.cut', { node: x.nodeId, a: inf(x.alpha), b: inf(x.beta), n: x.prunedChildren?.length ?? 0 })}
        </At>
      ))}
    </>
  )
}

function WhyScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const prune = STEPS.find((s) => s.type === 'prune' && s.nodeId === 'C')!
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <TreeView tree={TREE} values={prune.nodeValues} step={prune} hl="C" />
      </Svg>
      <At x={1330} y={180} size={28} w={460} o={eseg(p, 0.05, 0.15)} style={{ textAlign: 'center' }}>{t('watch.alphabeta.label.why_1', { a: ROOT })}</At>
      <At x={1330} y={350} size={28} w={460} o={eseg(p, 0.25, 0.35)} style={{ textAlign: 'center' }}>{t('watch.alphabeta.label.why_2')}</At>
      <At x={1330} y={520} size={28} w={460} color={C.yellow} o={eseg(p, 0.45, 0.55)} style={{ textAlign: 'center' }}>{t('watch.alphabeta.label.why_3')}</At>
      <At x={1330} y={700} size={34} className="font-black font-mono" color={C.red} o={eseg(p, 0.65, 0.75)}>α ≥ β ⇒ cut</At>
    </>
  )
}

function OrderScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const showWorst = p > 0.45
  const tree = showWorst ? WORST : TREE
  const steps = showWorst ? WORST_STEPS : STEPS
  const last = steps[steps.length - 1]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <TreeView tree={tree} values={last.nodeValues} step={last} />
      </Svg>
      <At x={1330} y={220} size={30} className="font-black" w={460} style={{ textAlign: 'center' }}>{t(showWorst ? 'watch.alphabeta.label.bad_order' : 'watch.alphabeta.label.good_order')}</At>
      <At x={1330} y={340} size={56} className="font-black font-mono" color={showWorst ? C.red : C.teal}>{leavesEvaluated(steps)} / 8</At>
      <At x={1330} y={410} size={24} color={C.muted}>{t('watch.alphabeta.label.leaves_seen')}</At>
      <At x={1330} y={520} size={28} className="font-mono">{t('watch.alphabeta.label.root_is', { v: minimaxValue(tree) })}</At>
      <At x={1330} y={660} size={26} color={C.yellow} w={460} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.alphabeta.label.order_note')}</At>
    </>
  )
}

function Recap({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.3} />
      </Svg>
      <At x={800} y={150} size={30} color={C.blue} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>{t('watch.ui.recap')}</At>
      {[1, 2, 3].map((k) => (
        <At key={k} x={220} y={240 + k * 120} size={36} anchor="l" w={1180} o={eseg(p, 0.08 + k * 0.12, 0.16 + k * 0.12)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.alphabeta.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const alphabeta: Episode = {
  id: 'alphabeta',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'tree', dur: 12000, cues: [0, 0.25, 0.55], render: (p) => <TreeScene p={p} /> },
    { id: 'minimax', dur: 15000, cues: [0, 0.3, 0.75], ponder: true, render: (p) => <MinimaxScene p={p} /> },
    { id: 'search', dur: 34000, cues: [0, 0.12, 0.3, 0.5, 0.72], render: (p) => <SearchScene p={p} /> },
    { id: 'why', dur: 16000, cues: [0, 0.25, 0.45, 0.65], render: (p) => <WhyScene p={p} /> },
    { id: 'order', dur: 15000, cues: [0, 0.45, 0.7], render: (p) => <OrderScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
