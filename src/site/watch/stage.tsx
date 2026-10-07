import type { CSSProperties, ReactNode } from 'react'
import { Latex } from '../../components/Latex'

// The stage is a 1600 × 900 coordinate space. SVG draws in it directly; HTML (text, KaTeX) is placed with <At>,
// sized in container-query units so everything scales with the player width.

export const W = 1600
export const H = 900

/** 3Blue1Brown-like palette on a dark stage. */
export const C = {
  bg: '#0e1117',
  grid: '#1f2633',
  axis: '#3b4556',
  text: '#e8ecf2',
  muted: '#8b95a7',
  blue: '#58c4dd',
  yellow: '#ffd866',
  teal: '#5cd0b3',
  red: '#fc6255',
  green: '#83c167',
  purple: '#a78bfa',
  orange: '#ff9f43',
} as const

/** Full-stage SVG layer. */
export const Svg = ({ children, className = '' }: { children?: ReactNode; className?: string }) => (
  <svg viewBox={`0 0 ${W} ${H}`} className={`absolute inset-0 h-full w-full ${className}`} aria-hidden>
    {children}
  </svg>
)

type Anchor = 'c' | 'l' | 'r' | 'tl' | 't' | 'b'
const SHIFT: Record<Anchor, string> = { c: '-50%,-50%', l: '0,-50%', r: '-100%,-50%', tl: '0,0', t: '-50%,0', b: '-50%,-100%' }

/** HTML positioned at stage coordinates (x, y). size is the font size in stage pixels. */
export function At({ x, y, children, size = 36, anchor = 'c', o = 1, scale = 1, color = C.text, w, className = '', style }: {
  x: number
  y: number
  children?: ReactNode
  size?: number
  anchor?: Anchor
  o?: number
  scale?: number
  color?: string
  /** max width in stage pixels (text wraps) */
  w?: number
  className?: string
  style?: CSSProperties
}) {
  if (o <= 0.001) return null
  return (
    <div
      className={`pointer-events-none absolute leading-tight ${className}`}
      style={{
        left: `${(x / W) * 100}%`,
        top: `${(y / H) * 100}%`,
        transform: `translate(${SHIFT[anchor]}) scale(${scale})`,
        transformOrigin: anchor === 'l' || anchor === 'tl' ? 'left center' : anchor === 'r' ? 'right center' : 'center',
        opacity: o,
        color,
        fontSize: `${(size / W) * 100}cqw`,
        width: w ? `${(w / W) * 100}cqw` : undefined,
        whiteSpace: w ? 'normal' : 'nowrap',
        ...style,
      }}
    >
      {children}
    </div>
  )
}

/** KaTeX at stage coordinates. */
export const Tex = ({ f, ...rest }: { f: string } & Omit<Parameters<typeof At>[0], 'children'>) => (
  <At {...rest}>
    <Latex formula={f} />
  </At>
)

/** Faint background grid, the default backdrop. */
export const Grid = ({ o = 1, step = 80 }: { o?: number; step?: number }) => (
  <g opacity={o}>
    {Array.from({ length: Math.floor(W / step) + 1 }, (_, i) => <line key={'v' + i} x1={i * step} x2={i * step} y1={0} y2={H} stroke={C.grid} strokeWidth={1} />)}
    {Array.from({ length: Math.floor(H / step) + 1 }, (_, i) => <line key={'h' + i} x1={0} x2={W} y1={i * step} y2={i * step} stroke={C.grid} strokeWidth={1} />)}
  </g>
)

/** A line drawn progressively: t = 0 nothing, t = 1 complete. */
export const DrawLine = ({ x1, y1, x2, y2, t, color = C.text, width = 3, dash }: { x1: number; y1: number; x2: number; y2: number; t: number; color?: string; width?: number; dash?: string }) =>
  t <= 0 ? null : <line x1={x1} y1={y1} x2={x1 + (x2 - x1) * Math.min(1, t)} y2={y1 + (y2 - y1) * Math.min(1, t)} stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={dash} />

/** Episode title card used as the first frame. */
export const TitleCard = ({ p, kicker, title, sub }: { p: number; kicker: string; title: string; sub: string }) => {
  const o = Math.min(1, p * 6)
  return (
    <>
      <Svg>
        <Grid o={0.5} />
      </Svg>
      <At x={800} y={330} size={28} color={C.blue} o={o} className="font-bold uppercase tracking-[0.2em]">
        {kicker}
      </At>
      <At x={800} y={430} size={76} o={o} className="font-black tracking-tight" w={1300} style={{ textAlign: 'center' }}>
        {title}
      </At>
      <At x={800} y={560} size={32} color={C.muted} o={o} w={1100} style={{ textAlign: 'center' }}>
        {sub}
      </At>
    </>
  )
}
