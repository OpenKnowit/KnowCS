import { describe, expect, it } from 'vitest'
import { parse, runPython } from './minipy'

const run = (code: string) => runPython(`import numpy as np\n${code}`)

describe('parse', () => {
  it('优先级：& 比 > 绑定更紧（a > 3 & a < 10 是比较链）', () => {
    const [st] = parse('a > 3 & a < 10')
    expect(st.k).toBe('expr')
    if (st.k === 'expr') {
      expect(st.x.k).toBe('compare')
      if (st.x.k === 'compare') expect(st.x.ops).toEqual(['>', '<'])
    }
  })

  it('缩进与不支持的语句给出明确错误', () => {
    expect(() => parse('  x = 1')).toThrow('unexpected indent')
    expect(() => parse('try:\n    x = 1')).toThrow('not supported')
    expect(() => parse('if x:\ny = 1')).toThrow('expected an indented block')
    expect(() => parse('if x:\n    y = 1\n  z = 2')).toThrow('unindent does not match')
  })
})

describe('runPython 输出与 numpy 一致', () => {
  it('最后一个表达式显示 repr，print 写入 stdout', () => {
    const r = run('a = np.arange(6).reshape(2, 3)\nprint(a)\na.T')
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('[[0 1 2]\n [3 4 5]]\n')
    expect(r.out).toBe('array([[0, 3],\n       [1, 4],\n       [2, 5]])')
  })

  it('Python 标量语义：/ 得浮点，// 向下取整，% 与除数同号', () => {
    expect(run('5 / 2, 5 // 2, -7 % 3, 2 ** -1').out).toBe('(2.5, 2, 2, 0.5)')
  })

  it('经典报错：歧义真值 / 越界 / 列表元组下标', () => {
    expect(run('a = np.arange(16).reshape(4, 4)\na[a > 3 & a < 10]').error?.message).toMatch('truth value of an array with more than one element is ambiguous')
    expect(run('a = np.arange(4)\na[4]').error).toMatchObject({ type: 'IndexError', line: 3 })
    expect(run('l = [1, 2, 3]\nl[0, 1]').error?.message).toBe('list indices must be integers or slices, not tuple')
    expect(run('a = np.arange(6)\na /= 2').error?.type).toBe('UFuncTypeError')
  })
})

describe('索引轨迹（IndexTrace）', () => {
  // 源数组为 arange 时，结果值 == 源扁平下标 —— 用来验证溯源映射
  const cases = [
    'a = np.arange(16).reshape(4, 4)\na[1:3, ::2]',
    'a = np.arange(16).reshape(4, 4)\na[[0, 2, 3], [1, 3, 0]]',
    'a = np.arange(16).reshape(4, 4)\na[a % 3 == 0]',
    'a = np.arange(24).reshape(2, 3, 4)\na[0, :, [1, 3]]',
    'a = np.arange(24).reshape(2, 3, 4)\na[:, [0, 2], 1:3]',
    'a = np.arange(24).reshape(2, 3, 4)\na[..., ::-2]',
    'a = np.arange(9).reshape(3, 3)\na[[[0], [2]], [0, 2]]',
  ]
  it.each(cases)('srcFlat 与结果值一致：%s', (code) => {
    const r = run(code)
    expect(r.error).toBeNull()
    const tr = r.traces.at(-1)!
    expect(tr.out.values).toEqual(tr.srcFlat)
  })

  it('区分 view / copy / scalar，并记录掩码', () => {
    const r = run('a = np.arange(16).reshape(4, 4)\nv = a[1:3]\nc = a[[1, 2]]\nm = a[a > 10]\ns = a[1, 2]')
    expect(r.traces.map((t) => t.result)).toEqual(['view', 'copy', 'copy', 'scalar'])
    const mask = r.traces.find((t) => t.hasMask)!
    expect(mask.code).toBe('a[a > 10]')
    expect(mask.mask?.snapshot.values.filter(Boolean)).toHaveLength(5)
  })

  it('通过视图写入：记录被连带修改的变量', () => {
    const r = run('a = np.arange(16).reshape(4, 4)\nb = a[1:3, 1:3]\nb[0, 0] = 99')
    const w = r.traces.at(-1)!
    expect(w.result).toBe('write')
    expect(w.code).toBe('b[0, 0] = 99')
    expect(w.aliases).toEqual([expect.objectContaining({ name: 'a', changed: [5] })])
    expect(r.vars.find((v) => v.name === 'b')).toMatchObject({ isView: true, sharesWith: ['a'] })
  })

  it('fancy 结果是副本：修改不影响原数组', () => {
    const r = run('a = np.arange(16).reshape(4, 4)\nc = a[[1, 2]]\nc[0, 0] = -1\na[1, 0]')
    expect(r.out).toBe('4')
    expect(r.traces.at(-2)!.aliases).toEqual([])
  })

  it('原地运算 b += 100 同样写穿视图', () => {
    const r = run('a = np.arange(6)\nb = a[::2]\nb += 100\na')
    expect(r.out).toBe('array([100,   1, 102,   3, 104,   5])')
    expect(r.traces.at(-1)!.aliases[0]).toMatchObject({ name: 'a', changed: [0, 2, 4] })
  })

  it('增量下标赋值只记录一次写入', () => {
    const r = run('a = np.arange(9).reshape(3, 3)\na[a > 4] += 10')
    // 先读后写的那次读取不单独记录，只留一条写入轨迹
    expect(r.traces.map((t) => t.mode)).toEqual(['write'])
    expect(r.traces[0].srcFlat).toEqual([5, 6, 7, 8])
  })
})
