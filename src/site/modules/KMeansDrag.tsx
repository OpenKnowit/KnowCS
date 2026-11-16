// k-means practice: drag each point onto the centroid it is closest to, checked on drop. Built with Dragology
// (declarative dragging): a point's pin can only land on one of the k centroids, or back on the point, and the library
// previews each landing, snaps and animates. Only this mode uses it: Dragology's SVG takes touches for dragging, so
// the read-only plot (which a finger can scroll past) stays plain SVG.
import { useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { DraggableRenderer, translate } from 'dragology'
import type { Draggable } from 'dragology'
import { niceTicks } from '../../lib/chart'
import { fmt, PALETTE } from '../format'
import { plotFrame } from './kmeansFrame'

export type Guess = (number | null)[]
export interface DragState { guess: Guess }

type Frame = ReturnType<typeof plotFrame>

/** grid and tick labels: a plain component, so Dragology draws it as it is */
function Axes({ f }: { f: Frame }) {
  const { oneD, W, H, P, x0, x1, y0, y1, X, Y } = f
  return (
    <g>
      {niceTicks(x0, x1, 7).map((v) => (
        <g key={'x' + v}>
          {!oneD && <line x1={X(v)} x2={X(v)} y1={P} y2={H - P} stroke="#f1f5f9" />}
          <text x={X(v)} y={H - 10} textAnchor="middle" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>
        </g>
      ))}
      {!oneD && niceTicks(y0, y1, 6).map((v) => (
        <g key={'y' + v}>
          <line x1={P} x2={W - P} y1={Y(v)} y2={Y(v)} stroke="#f1f5f9" />
          <text x={P - 6} y={Y(v) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>
        </g>
      ))}
      {oneD && <line x1={P} x2={W - P} y1={H / 2} y2={H / 2} stroke="#cbd5e1" />}
    </g>
  )
}

/** where each assigned pin sits: in a ring around its centroid, in point order */
function slots(guess: Guess, centroidsPx: [number, number][]): ([number, number] | null)[] {
  return guess.map((g, i) => {
    if (g === null) return null
    const members = guess.flatMap((h, j) => (h === g ? [j] : []))
    const n = members.length
    const r = Math.max(24, (n * 19) / (2 * Math.PI))
    // from straight below, clockwise: the centroid's label is above it
    const a = Math.PI / 2 + (members.indexOf(i) * 2 * Math.PI) / n
    const [cx, cy] = centroidsPx[g]
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)]
  })
}

function makeDraggable(f: Frame, pts: number[][], names: string[], centroids: number[][], next: number[][], answer: number[], revealed: boolean): Draggable<DragState> {
  const ptsPx = pts.map(f.at)
  const cPx = centroids.map(f.at)
  const nPx = next.map(f.at)
  return ({ state, d, draggedId }) => {
    const at = slots(state.guess, cPx)
    return (
      <g>
        <Axes f={f} />
        {revealed && (
          <g id="km-next">
            {nPx.map(([x, y], k) => (
              <g id={`km-next-${k}`}>
                <line x1={cPx[k][0]} y1={cPx[k][1]} x2={x} y2={y} stroke={PALETTE[k]} strokeWidth={2} strokeDasharray="4 3" />
                <rect transform={translate(x, y) + ' rotate(45)'} x={-7} y={-7} width={14} height={14} fill={PALETTE[k]} />
              </g>
            ))}
          </g>
        )}
        {/* the same element in every state (Dragology blends one into the other): an unassigned point's link is empty */}
        {state.guess.map((g, i) => {
          const [x, y] = ptsPx[i]
          const [x2, y2] = at[i] ?? ptsPx[i]
          return <line id={`km-link-${i}`} x1={x} y1={y} x2={x2} y2={y2} stroke={g === null ? '#cbd5e1' : PALETTE[g]} strokeOpacity={g === null ? 0 : 0.5} strokeWidth={2} strokeDasharray={g === null || g === answer[i] ? '0' : '4 3'} />
        })}
        {cPx.map(([x, y], k) => (
          <g id={`km-c-${k}`} transform={translate(x, y)}>
            <circle r={11} fill="#fff" stroke={PALETTE[k]} strokeWidth={3} />
            <text y={-16} textAnchor="middle" className="text-[11px] font-extrabold" fill={PALETTE[k]}>C{k + 1}</text>
          </g>
        ))}
        {ptsPx.map(([x, y], i) => (
          <g id={`km-pt-${i}`} transform={translate(x, y)}>
            <circle r={6.5} fill={state.guess[i] === null ? '#cbd5e1' : PALETTE[state.guess[i]!]} stroke="#fff" strokeWidth={1.5} />
            <text x={9} y={f.oneD ? 20 : 4} className="fill-slate-600 text-[12px] font-bold">{names[i]}</text>
          </g>
        ))}
        {state.guess.map((g, i) => {
          const [x, y] = at[i] ?? ptsPx[i]
          const wrong = g !== null && g !== answer[i]
          const id = `km-pin-${i}`
          // only the centroids are targets: a point often sits on a centroid (seeds are data points), so "back on the
          // point" would win a drop aimed at that centroid. Far from every centroid, nothing changes.
          const options = centroids.map((_, k) => ({ guess: state.guess.map((v, j) => (j === i ? k : v)) }))
          return (
            <g
              id={id}
              transform={translate(x, y)}
              dragologyZIndex={draggedId === id ? '/1' : 1}
              dragologyOnDrag={() => d.closest(options).whenFar(state).withFloating()}
            >
              {/* a bigger target than the pin, for a finger */}
              <circle r={14} fill="transparent" />
              <circle r={g === null ? 8 : 9} fill={g === null ? '#475569' : PALETTE[g]} stroke={wrong ? '#e11d48' : '#fff'} strokeWidth={wrong ? 3 : 2} />
              {g !== null && <text y={3} textAnchor="middle" className="text-[8px] font-extrabold" fill="#fff">{names[i].slice(0, 3)}</text>}
            </g>
          )
        })}
      </g>
    )
  }
}

/** scales a fixed-size Dragology drawing to the width of its box (Dragology reads the scale back from the box) */
function Fit({ w, h, label, children }: { w: number; h: number; label: string; children: ReactNode }) {
  const box = useRef<HTMLDivElement>(null)
  const [s, setS] = useState(1)
  useLayoutEffect(() => {
    const el = box.current
    if (!el) return
    const measure = () => setS(el.clientWidth / w || 1)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [w])
  return (
    <div ref={box} role="group" aria-label={label} className="relative w-full overflow-hidden" style={{ height: h * s }}>
      <div style={{ width: w, height: h, transform: `scale(${s})`, transformOrigin: '0 0' }}>{children}</div>
    </div>
  )
}

export function KMeansPractice({ pts, names, centroids, next, answer, state, revealed, onDrop }: {
  pts: number[][]
  names: string[]
  centroids: number[][]
  next: number[][]
  answer: number[]
  state: DragState
  revealed: boolean
  onDrop: (s: DragState) => void
}) {
  const { t } = useTranslation()
  const f = plotFrame(pts, centroids, next)
  const draggable = makeDraggable(f, pts, names, centroids, next, answer, revealed)
  return (
    <Fit w={f.W} h={f.H} label={t('lab.km.drag_aria')}>
      <DraggableRenderer draggable={draggable} state={state} onDropState={onDrop} width={f.W} height={f.H} />
    </Fit>
  )
}
