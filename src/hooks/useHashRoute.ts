import { useCallback, useSyncExternalStore } from 'react'
import { formatRoute, parseRoute } from '../lib/route'
import type { Route, RouteTable } from '../lib/route'

const subscribe = (onChange: () => void) => {
  window.addEventListener('hashchange', onChange)
  return () => window.removeEventListener('hashchange', onChange)
}
const getHash = () => window.location.hash

/** 订阅 location.hash 并解析为 Route；navigate 写入 hash（产生历史记录，浏览器后退可用）。 */
export const useHashRoute = (table: RouteTable): [Route, (route: Route) => void] => {
  const hash = useSyncExternalStore(subscribe, getHash)
  const route = parseRoute(hash, table)
  const navigate = useCallback((next: Route) => {
    const target = formatRoute(next)
    if (window.location.hash !== target) window.location.hash = target
  }, [])
  return [route, navigate]
}
