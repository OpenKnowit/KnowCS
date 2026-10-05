import { describe, expect, it } from 'vitest'
import { formatRoute, parseRoute } from './route'
import type { Route, RouteTable } from './route'

const TABLE: RouteTable = {
  course: ['numpy', 'knn', 'kmeans'],
  package: ['numpy', 'pandas'],
  extend: ['attention'],
}

describe('parseRoute', () => {
  it('损坏的 URI 编码回退默认模块，不让整个应用崩溃', () => {
    expect(parseRoute('#/course/%', TABLE)).toEqual({ mode: 'course', tab: 'numpy' })
    expect(parseRoute('#/package/%E0%A4', TABLE)).toEqual({ mode: 'course', tab: 'numpy' })
  })
  it('空 hash / 未知模式 → 默认课程模块', () => {
    expect(parseRoute('', TABLE)).toEqual({ mode: 'course', tab: 'numpy' })
    expect(parseRoute('#/', TABLE)).toEqual({ mode: 'course', tab: 'numpy' })
    expect(parseRoute('#/nope/x', TABLE)).toEqual({ mode: 'course', tab: 'numpy' })
  })

  it('课程模块：合法 id 命中，非法 id 回退默认', () => {
    expect(parseRoute('#/course/knn', TABLE)).toEqual({ mode: 'course', tab: 'knn' })
    expect(parseRoute('#/course/perceptron', TABLE)).toEqual({ mode: 'course', tab: 'numpy' })
  })

  it('package / extend：可带条目 id，非法条目回到列表页', () => {
    expect(parseRoute('#/package', TABLE)).toEqual({ mode: 'package', item: null })
    expect(parseRoute('#/package/pandas', TABLE)).toEqual({ mode: 'package', item: 'pandas' })
    expect(parseRoute('#/package/attention', TABLE)).toEqual({ mode: 'package', item: null })
    expect(parseRoute('#/extend/attention', TABLE)).toEqual({ mode: 'extend', item: 'attention' })
  })

  it('兼容缺少前导斜杠的写法', () => {
    expect(parseRoute('#course/kmeans', TABLE)).toEqual({ mode: 'course', tab: 'kmeans' })
  })
})

describe('formatRoute', () => {
  it('与 parseRoute 互逆', () => {
    const routes: Route[] = [
      { mode: 'course', tab: 'knn' },
      { mode: 'package', item: null },
      { mode: 'package', item: 'pandas' },
      { mode: 'extend', item: 'attention' },
    ]
    for (const r of routes) expect(parseRoute(formatRoute(r), TABLE)).toEqual(r)
  })

  it('输出格式', () => {
    expect(formatRoute({ mode: 'course', tab: 'knn' })).toBe('#/course/knn')
    expect(formatRoute({ mode: 'extend', item: null })).toBe('#/extend')
  })
})
