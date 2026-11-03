import { describe, expect, it } from 'vitest'
import { buildAbTree, generateAbSteps, leavesEvaluated, minimaxValue, treeWithLeaves } from './alphabeta'
import { AB_NODES, AB_PRESETS } from '../data/constants'

describe('generateAbSteps', () => {
  it('剪枝得到的根值与无剪枝 minimax 一致（所有预设）', () => {
    for (const key of Object.keys(AB_PRESETS)) {
      const tree = treeWithLeaves(AB_PRESETS[key])
      const steps = generateAbSteps(tree)
      const rootExit = steps.filter((s) => s.type === 'exit' && s.nodeId === 'A').at(-1)
      expect(rootExit?.returnValue).toBe(minimaxValue(tree))
    }
  })

  it('默认 user 预设的根值为 6', () => {
    // 树：MAX( MIN(MAX(3,5),MAX(2,4)), MIN(MAX(6,1),MAX(0,7)) )
    //  = MAX( MIN(5,4), MIN(6,7) ) = MAX(4,6) = 6 ... 校验 minimax
    const tree = treeWithLeaves(AB_PRESETS.user)
    expect(minimaxValue(tree)).toBe(6)
  })

  it('进入/向下搜索节点时 α < β 区间始终有效', () => {
    // enter / traverse 步的区间继承自父节点，必满足 α < β；
    // update 步在剪枝判定之前发出，α 可能已 ≥ β，不在此不变量内。
    const tree = treeWithLeaves(AB_PRESETS.user)
    for (const s of generateAbSteps(tree)) {
      if (s.type === 'enter' || s.type === 'traverse') {
        expect(s.alpha).toBeLessThan(s.beta)
      }
    }
  })

  it('alpha 预设会触发至少一次剪枝并标记被剪节点', () => {
    const tree = treeWithLeaves(AB_PRESETS.alpha)
    const steps = generateAbSteps(tree)
    const prunes = steps.filter((s) => s.type === 'prune')
    expect(prunes.length).toBeGreaterThan(0)
    const lastPrune = prunes.at(-1)!
    expect(lastPrune.prunedChildren!.length).toBeGreaterThan(0)
    // 被剪节点应出现在后续步骤的 prunedNodes 中
    expect(lastPrune.prunedNodes.length).toBeGreaterThan(0)
  })

  it('被剪节点从不被进入；进入数 + 最终被剪数 = 15（所有预设）', () => {
    for (const key of Object.keys(AB_PRESETS)) {
      const tree = treeWithLeaves(AB_PRESETS[key])
      const steps = generateAbSteps(tree)
      const entered = new Set(steps.filter((s) => s.type === 'enter').map((s) => s.nodeId))
      const finalPruned = new Set(steps.at(-1)!.prunedNodes)
      // 不相交：被剪的子树不会被访问
      for (const id of entered) expect(finalPruned.has(id)).toBe(false)
      // 完备：每个节点要么被进入、要么被剪
      expect(entered.size + finalPruned.size).toBe(15)
    }
  })

  it('第一步是进入根节点，区间为 [-∞, +∞]', () => {
    const tree = treeWithLeaves(AB_PRESETS.user)
    const first = generateAbSteps(tree)[0]
    expect(first.type).toBe('enter')
    expect(first.nodeId).toBe('A')
    expect(first.alpha).toBe(-Infinity)
    expect(first.beta).toBe(Infinity)
  })
})

describe('leavesEvaluated', () => {
  it('counts the leaves alpha-beta looks at', () => {
    expect(leavesEvaluated(generateAbSteps(treeWithLeaves(AB_PRESETS.alpha)))).toBe(5)
    // the same eight values in a bad order: nothing is pruned, the root value is unchanged
    const worst = treeWithLeaves([0, 4, 1, 2, 8, 9, 6, 7])
    expect(leavesEvaluated(generateAbSteps(worst))).toBe(8)
    expect(minimaxValue(worst)).toBe(minimaxValue(treeWithLeaves(AB_PRESETS.alpha)))
  })
})

describe('buildAbTree', () => {
  it('depth 3 is the lecture tree', () => {
    const { nodes } = buildAbTree(3, [3, 5, 2, 4, 6, 1, 0, 7])
    for (const id in AB_NODES) {
      expect(nodes[id].type).toBe(AB_NODES[id].type)
      expect(nodes[id].children).toEqual(AB_NODES[id].children)
      expect(nodes[id].parent).toBe(AB_NODES[id].parent)
    }
    expect(Object.keys(nodes)).toHaveLength(15)
    expect(minimaxValue(nodes)).toBe(minimaxValue(treeWithLeaves([3, 5, 2, 4, 6, 1, 0, 7])))
  })
  it('other depths alternate MAX / MIN and name every node once', () => {
    const { nodes } = buildAbTree(5, Array.from({ length: 32 }, (_, i) => i))
    expect(Object.keys(nodes)).toHaveLength(63)
    expect(nodes.A.type).toBe('max')
    expect(nodes.B.type).toBe('min')
    expect(nodes.L32.parent).toBe(nodes[nodes.L32.parent!].id)
    // alpha-beta returns the plain minimax value at any depth
    expect(generateAbSteps(nodes).at(-1)!.returnValue).toBe(minimaxValue(nodes))
    const { nodes: two } = buildAbTree(2, [4, 9, 2, 7])
    expect(generateAbSteps(two).at(-1)!.returnValue).toBe(4)
  })
})
