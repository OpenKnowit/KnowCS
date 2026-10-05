import { useId, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, PointerEvent } from 'react'
import { linearScale, nearestIndex, niceTicks } from '../lib/chart'

export interface LinePoint {
  x: number
  y: number
}

interface LineChartProps {
  data: LinePoint[]
  /** 竖直参考线（如当前 K） */
  referenceX?: number
  /** 额外高亮的点（如最优 K） */
  highlightX?: number
  height?: number
  formatY?: (v: number) => string
  xLabel?: string
  /** 屏幕阅读器用的图表描述 */
  ariaLabel: string
}

const W = 320 // viewBox 宽度；实际宽度随容器自适应
const PAD = { top: 12, right: 12, bottom: 22, left: 34 }

/** 深色面板用的轻量折线图（SVG，无第三方依赖）。支持悬停 / 方向键查看数值。 */
export const LineChart = ({
  data,
  referenceX,
  highlightX,
  height = 180,
  formatY = (v) => String(v),
  xLabel,
  ariaLabel,
}: LineChartProps) => {
  const [active, setActive] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const titleId = useId()

  const { sx, sy, yTicks, path, xs } = useMemo(() => {
    const xs = data.map((d) => d.x)
    const ys = data.map((d) => d.y)
    const yTicks = niceTicks(Math.min(0, ...ys), Math.max(...ys), 4)
    const sx = linearScale([Math.min(...xs), Math.max(...xs)], [PAD.left, W - PAD.right])
    const sy = linearScale([yTicks[0], yTicks.at(-1)!], [height - PAD.bottom, PAD.top])
    const path = data.map((d, i) => `${i ? 'L' : 'M'}${sx(d.x).toFixed(1)},${sy(d.y).toFixed(1)}`).join('')
    return { sx, sy, yTicks, path, xs }
  }, [data, height])

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect || !data.length) return
    const vx = ((e.clientX - rect.left) / rect.width) * W
    const inv = linearScale([PAD.left, W - PAD.right], [xs[0], xs.at(-1)!])
    setActive(nearestIndex(xs, inv(vx)))
  }

  const onKeyDown = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const cur = active ?? 0
    setActive(Math.min(data.length - 1, Math.max(0, cur + (e.key === 'ArrowRight' ? 1 : -1))))
  }

  const tip = active !== null ? data[active] : null
  const tipX = tip ? sx(tip.x) : 0
  const tipLeft = tipX > W / 2

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${height}`}
      className="w-full h-auto select-none touch-none focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded-lg"
      role="img"
      aria-labelledby={titleId}
      tabIndex={0}
      onPointerMove={onPointerMove}
      onPointerLeave={() => setActive(null)}
      onKeyDown={onKeyDown}
      onBlur={() => setActive(null)}
    >
      <title id={titleId}>{ariaLabel}</title>

      {/* 网格 + y 轴刻度 */}
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={PAD.left} x2={W - PAD.right} y1={sy(v)} y2={sy(v)} stroke="#334155" strokeDasharray="3 3" />
          <text x={PAD.left - 6} y={sy(v)} dy="0.32em" textAnchor="end" fontSize={9} fill="#94a3b8">{formatY(v)}</text>
        </g>
      ))}

      {/* x 轴刻度 */}
      {data.map((d) => (
        <text key={d.x} x={sx(d.x)} y={height - 6} textAnchor="middle" fontSize={9} fill="#94a3b8">{d.x}</text>
      ))}

      {referenceX !== undefined && (
        <line x1={sx(referenceX)} x2={sx(referenceX)} y1={PAD.top} y2={height - PAD.bottom} stroke="#fbbf24" strokeDasharray="5 5" strokeWidth={1.5} />
      )}

      <path d={path} fill="none" stroke="#3b82f6" strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" />
      {data.map((d, i) => {
        const isHi = d.x === highlightX
        return (
          <circle
            key={d.x}
            cx={sx(d.x)}
            cy={sy(d.y)}
            r={i === active ? 5 : isHi ? 4.5 : 3}
            fill={isHi ? '#10b981' : '#3b82f6'}
            stroke={i === active ? '#fff' : isHi ? '#d1fae5' : 'none'}
            strokeWidth={1.5}
          />
        )
      })}

      {tip && (
        <g transform={`translate(${tipX + (tipLeft ? -8 : 8)}, ${PAD.top})`} pointerEvents="none">
          <rect x={tipLeft ? -76 : 0} width={76} height={34} rx={6} fill="#1e293b" stroke="#475569" />
          <text x={tipLeft ? -68 : 8} y={14} fontSize={9} fill="#94a3b8">{xLabel ?? 'x'} = {tip.x}</text>
          <text x={tipLeft ? -68 : 8} y={27} fontSize={10} fontWeight={700} fill="#f1f5f9">{formatY(tip.y)}</text>
        </g>
      )}
    </svg>
  )
}
