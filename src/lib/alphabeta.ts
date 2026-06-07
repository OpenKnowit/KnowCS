import type { AbNode, AbStep } from '../types'

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
