import type { AbNode, AbStep } from '../types'
import { AB_LEAF_IDS, AB_NODES } from '../data/constants'

// Alpha-Beta 剪枝的纯逻辑：给定一棵带叶子值的博弈树，
// 生成完整的逐步追踪（DFS + α/β 剪枝），供可视化按结构化字段渲染。

/** 递归标记被剪掉的子树（节点 + 边），用于可视化高亮。 */
function markSubtreePruned(
  nodes: Record<string, AbNode>,
  subId: string,
  prunedNodes: Set<string>,
  prunedEdges: Set<string>
): void {
  prunedNodes.add(subId)
  const subNode = nodes[subId]
  if (subNode?.children) {
    for (const child of subNode.children) {
      prunedEdges.add(`${subId}->${child}`)
      markSubtreePruned(nodes, child, prunedNodes, prunedEdges)
    }
  }
}

/**
 * 生成 Alpha-Beta 算法的完整步骤序列。
 * @param nodes 节点表（含叶子 value）
 * @param rootId 根节点 id（约定为 MAX 层）
 */
export function generateAbSteps(nodes: Record<string, AbNode>, rootId = 'A'): AbStep[] {
  const steps: AbStep[] = []
  const visited = new Set<string>()
  const prunedNodes = new Set<string>()
  const prunedEdges = new Set<string>()

  const nodeVals: Record<string, number | null> = {}
  const alphas: Record<string, number> = {}
  const betas: Record<string, number> = {}
  for (const id in nodes) {
    nodeVals[id] = null
    alphas[id] = -Infinity
    betas[id] = Infinity
  }

  // 快照当前全局状态到一个步骤里（数组化便于序列化/测试）
  const snapshot = (
    partial: Omit<
      AbStep,
      | 'nodeValues'
      | 'nodeAlphas'
      | 'nodeBetas'
      | 'visited'
      | 'prunedNodes'
      | 'prunedEdges'
    >
  ): AbStep => ({
    ...partial,
    nodeValues: { ...nodeVals },
    nodeAlphas: { ...alphas },
    nodeBetas: { ...betas },
    visited: [...visited],
    prunedNodes: [...prunedNodes],
    prunedEdges: [...prunedEdges],
  })

  function dfs(nodeId: string, alpha: number, beta: number): number {
    visited.add(nodeId)
    alphas[nodeId] = alpha
    betas[nodeId] = beta
    const node = nodes[nodeId]

    steps.push(snapshot({ type: 'enter', nodeId, alpha, beta, activeEdge: null }))

    if (node.type === 'leaf') {
      const leafVal = node.value ?? 0
      nodeVals[nodeId] = leafVal
      steps.push(
        snapshot({
          type: 'leaf_eval',
          nodeId,
          alpha,
          beta,
          activeEdge: node.parent ? `${node.parent}->${nodeId}` : null,
          leafValue: leafVal,
        })
      )
      return leafVal
    }

    const isMax = node.type === 'max'
    let currentBest = isMax ? -Infinity : Infinity
    let currentAlpha = alpha
    let currentBeta = beta
    const children = node.children ?? []

    for (let i = 0; i < children.length; i++) {
      const childId = children[i]

      steps.push(
        snapshot({
          type: 'traverse',
          nodeId,
          alpha: currentAlpha,
          beta: currentBeta,
          activeEdge: `${nodeId}->${childId}`,
          childId,
        })
      )

      const childResult = dfs(childId, currentAlpha, currentBeta)

      const prevBest = currentBest
      if (isMax) {
        currentBest = Math.max(currentBest, childResult)
        nodeVals[nodeId] = currentBest
        const prevBound = currentAlpha
        currentAlpha = Math.max(currentAlpha, currentBest)
        alphas[nodeId] = currentAlpha
        steps.push(
          snapshot({
            type: 'update',
            nodeId,
            alpha: currentAlpha,
            beta: currentBeta,
            activeEdge: null,
            childId,
            childResult,
            prevBest,
            newBest: currentBest,
            prevBound,
            newBound: currentAlpha,
          })
        )
      } else {
        currentBest = Math.min(currentBest, childResult)
        nodeVals[nodeId] = currentBest
        const prevBound = currentBeta
        currentBeta = Math.min(currentBeta, currentBest)
        betas[nodeId] = currentBeta
        steps.push(
          snapshot({
            type: 'update',
            nodeId,
            alpha: currentAlpha,
            beta: currentBeta,
            activeEdge: null,
            childId,
            childResult,
            prevBest,
            newBest: currentBest,
            prevBound,
            newBound: currentBeta,
          })
        )
      }

      // 剪枝判定：β ≤ α
      if (currentBeta <= currentAlpha) {
        const prunedChildren = children.slice(i + 1)
        for (const siblingId of prunedChildren) {
          markSubtreePruned(nodes, siblingId, prunedNodes, prunedEdges)
          prunedEdges.add(`${nodeId}->${siblingId}`)
        }
        steps.push(
          snapshot({
            type: 'prune',
            nodeId,
            alpha: currentAlpha,
            beta: currentBeta,
            activeEdge: null,
            prunedChildren,
          })
        )
        break
      }
    }

    steps.push(
      snapshot({
        type: 'exit',
        nodeId,
        alpha: currentAlpha,
        beta: currentBeta,
        activeEdge: null,
        returnValue: currentBest,
      })
    )

    return currentBest
  }

  dfs(rootId, -Infinity, Infinity)
  return steps
}

/** 直接求博弈树的极小化极大值（不剪枝，用于校验剪枝结果一致性）。 */
export function minimaxValue(nodes: Record<string, AbNode>, nodeId = 'A'): number {
  const node = nodes[nodeId]
  if (node.type === 'leaf') return node.value ?? 0
  const childVals = (node.children ?? []).map((c) => minimaxValue(nodes, c))
  return node.type === 'max' ? Math.max(...childVals) : Math.min(...childVals)
}

/** The standard 3-level tree with the given eight leaf values (left to right). */
export function treeWithLeaves(values: readonly number[]): Record<string, AbNode> {
  const nodes: Record<string, AbNode> = JSON.parse(JSON.stringify(AB_NODES))
  AB_LEAF_IDS.forEach((id, i) => {
    nodes[id].value = values[i]
  })
  return nodes
}

/** How many leaves alpha-beta actually evaluates. */
export const leavesEvaluated = (steps: AbStep[]): number => steps.filter((s) => s.type === 'leaf_eval').length

/** internal node names in breadth-first order: A … Z, then AA, AB … */
const nodeName = (i: number): string => (i < 26 ? String.fromCharCode(65 + i) : nodeName(Math.floor(i / 26) - 1) + String.fromCharCode(65 + (i % 26)))

export const AB_LEAF_GAP = 120
export const AB_LEVEL_GAP = 120

/**
 * A game tree by its shape alone: a leaf has a value, any other node has children, left to right, as many as
 * it likes. Levels alternate MAX (the root), MIN, MAX, …, so the type of a node follows from its depth.
 */
export interface AbShape {
  value?: number
  children?: AbShape[]
}
/** a node of a shape by the child index taken at each level from the root; [] is the root */
export type AbPath = number[]

const isLeafShape = (s: AbShape): boolean => !s.children?.length

/** a full binary shape `depth` levels deep above its leaves, the leaves taking `leafValues` left to right */
export function fullShape(depth: number, leafValues: readonly number[]): AbShape {
  let i = 0
  const grow = (l: number): AbShape => (l === depth ? { value: leafValues[i++] ?? 0 } : { children: [grow(l + 1), grow(l + 1)] })
  return grow(0)
}

/** the leaf values, left to right */
export const shapeLeaves = (s: AbShape): number[] => (isLeafShape(s) ? [s.value ?? 0] : s.children!.flatMap(shapeLeaves))
/** how many levels the deepest leaf sits below the root */
export const shapeDepth = (s: AbShape): number => (isLeafShape(s) ? 0 : 1 + Math.max(...s.children!.map(shapeDepth)))
/** the shape with every leaf value replaced by f(value, index left to right) */
export function mapLeaves(s: AbShape, f: (v: number, i: number) => number): AbShape {
  let i = 0
  const go = (n: AbShape): AbShape => (isLeafShape(n) ? { value: f(n.value ?? 0, i++) } : { children: n.children!.map(go) })
  return go(s)
}

/**
 * Nodes with SVG positions for a shape: leaves AB_LEAF_GAP apart along the bottom in left-to-right order, each
 * at its own level, every parent centred over its first and last child. Inner nodes are named A, B, … breadth
 * first and leaves L1, L2, … left to right; `paths` maps each name back to its node in the shape.
 */
export function layoutAbTree(shape: AbShape): { nodes: Record<string, AbNode>; paths: Record<string, AbPath>; width: number; height: number } {
  // names: inner nodes breadth first, leaves left to right
  const names = new Map<AbShape, string>()
  let inner = 0
  let leaf = 0
  const visitLeaves = (s: AbShape) => (isLeafShape(s) ? names.set(s, `L${++leaf}`) : s.children!.forEach(visitLeaves))
  visitLeaves(shape)
  for (let level = [shape]; level.length; level = level.flatMap((s) => s.children ?? [])) for (const s of level) if (!isLeafShape(s)) names.set(s, nodeName(inner++))

  const nodes: Record<string, AbNode> = {}
  const paths: Record<string, AbPath> = {}
  let slot = 0
  let bottom = 0
  const place = (s: AbShape, l: number, parent: string | null, path: AbPath): AbNode => {
    const id = names.get(s)!
    paths[id] = path
    if (isLeafShape(s)) {
      const y = 60 + l * AB_LEVEL_GAP + 10
      bottom = Math.max(bottom, y)
      return (nodes[id] = { id, type: 'leaf', parent, value: s.value ?? 0, x: 70 + slot++ * AB_LEAF_GAP, y })
    }
    const kids = s.children!.map((c, i) => place(c, l + 1, id, [...path, i]))
    return (nodes[id] = { id, type: l % 2 === 0 ? 'max' : 'min', parent, children: kids.map((k) => k.id), x: (kids[0].x + kids.at(-1)!.x) / 2, y: 60 + l * AB_LEVEL_GAP })
  }
  place(shape, 0, null, [])
  return { nodes, paths, width: 140 + (slot - 1) * AB_LEAF_GAP, height: bottom + 60 }
}

/**
 * A full binary game tree `depth` levels deep above its leaves (MAX at the root, then MIN, MAX, …), with SVG
 * positions: leaves AB_LEAF_GAP apart along the bottom, each parent centred over its children.
 * depth 3 is the lecture's tree: A → B, C → D…G → L1…L8.
 */
export function buildAbTree(depth: number, leafValues: number[]): { nodes: Record<string, AbNode>; width: number; height: number } {
  const { nodes, width, height } = layoutAbTree(fullShape(depth, leafValues))
  return { nodes, width, height }
}

// ---- editing a shape: each returns a new shape and leaves the old one alone ----

/** the node at `path` rebuilt by f; the rest of the tree is shared */
function update(s: AbShape, path: AbPath, f: (n: AbShape) => AbShape): AbShape {
  if (!path.length) return f(s)
  const [i, ...rest] = path
  return { children: s.children!.map((c, j) => (j === i ? update(c, rest, f) : c)) }
}
export const shapeAt = (s: AbShape, path: AbPath): AbShape => path.reduce((n, i) => n.children![i], s)

/** a new leaf as the last child of `path`; a leaf there becomes a MIN / MAX node over its old value and the new one */
export const addChild = (s: AbShape, path: AbPath, value: number): AbShape =>
  update(s, path, (n) => ({ children: isLeafShape(n) ? [{ value: n.value ?? 0 }, { value }] : [...n.children!, { value }] }))

/** the node at `path` and everything under it gone; a parent left with no children becomes a leaf worth `value` */
export function removeNode(s: AbShape, path: AbPath, value: number): AbShape {
  if (!canRemove(s, path)) return s
  const i = path.at(-1)!
  return update(s, path.slice(0, -1), (p) => {
    const rest = p.children!.filter((_, j) => j !== i)
    return rest.length ? { children: rest } : { value }
  })
}
/** every node but the root, and not the root's only child (the root stays a MIN / MAX node) */
export const canRemove = (s: AbShape, path: AbPath): boolean => path.length > 1 || (path.length === 1 && s.children!.length > 1)

/** the node at `path` swapped with its neighbour on the left (-1) or the right (+1) */
export function moveNode(s: AbShape, path: AbPath, dir: -1 | 1): AbShape {
  if (!canMove(s, path, dir)) return s
  const i = path.at(-1)!
  return update(s, path.slice(0, -1), (p) => {
    const kids = [...p.children!]
    ;[kids[i], kids[i + dir]] = [kids[i + dir], kids[i]]
    return { children: kids }
  })
}
export function canMove(s: AbShape, path: AbPath, dir: -1 | 1): boolean {
  if (!path.length) return false
  const j = path.at(-1)! + dir
  return j >= 0 && j < shapeAt(s, path.slice(0, -1)).children!.length
}

export const setLeafValue = (s: AbShape, path: AbPath, value: number): AbShape => update(s, path, () => ({ value }))
