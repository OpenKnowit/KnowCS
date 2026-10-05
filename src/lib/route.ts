// --- Hash 路由：#/course/knn、#/package/numpy、#/extend/attention ---
// 单文件 / IPFS 托管没有服务端路由，用 location.hash 保存当前视图，支持刷新、前进后退与分享链接。

export type AppMode = 'course' | 'package' | 'extend'

export type Route =
  | { mode: 'course'; tab: string }
  | { mode: 'package' | 'extend'; item: string | null }

export interface RouteTable {
  course: readonly string[] // 合法的课程模块 id，第一个为默认
  package: readonly string[]
  extend: readonly string[]
}

/** 解析 hash；任何非法片段都回退到最近的合法视图，而不是报错。 */
export const parseRoute = (hash: string, table: RouteTable): Route => {
  let parts: string[]
  try {
    parts = hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent)
  } catch {
    return { mode: 'course', tab: table.course[0] }
  }
  const [mode, id] = parts
  if (mode === 'package' || mode === 'extend') {
    return { mode, item: id && table[mode].includes(id) ? id : null }
  }
  return { mode: 'course', tab: id && table.course.includes(id) ? id : table.course[0] }
}

export const formatRoute = (route: Route): string => {
  if (route.mode === 'course') return `#/course/${encodeURIComponent(route.tab)}`
  return route.item ? `#/${route.mode}/${encodeURIComponent(route.item)}` : `#/${route.mode}`
}
