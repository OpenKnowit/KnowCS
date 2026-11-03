import { useRef } from 'react'
import type { ComponentProps } from 'react'
import { highlightPython } from '../site/playground/pyHighlight'

// Python with syntax colours wherever a lab shows code: blocks, inline chips, and editable boxes.

/** a code block; tone 'dark' for the slate-900 boxes the labs use, 'light' on white */
export const PyBlock = ({ code, className = '', tone = 'dark', ...rest }: { code: string; className?: string; tone?: 'dark' | 'light' } & Omit<ComponentProps<'pre'>, 'children'>) => (
  <pre className={className} {...rest}>
    {highlightPython(code, tone)}
  </pre>
)

/** inline code */
export const PyInline = ({ code, className = '', tone = 'dark' }: { code: string; className?: string; tone?: 'dark' | 'light' }) => <code className={className}>{highlightPython(code, tone)}</code>

/**
 * A textarea whose text is drawn coloured: the coloured copy sits under a transparent textarea and follows its scroll,
 * so typing, selection and the caret stay native. `className` styles both (padding, font, size must match).
 */
export function PyTextarea({ value, className = '', boxClassName = '', ...rest }: { value: string; className?: string; boxClassName?: string } & Omit<ComponentProps<'textarea'>, 'value'>) {
  const under = useRef<HTMLPreElement>(null)
  return (
    <div className={`relative min-w-0 ${boxClassName}`}>
      <pre ref={under} aria-hidden className={`pointer-events-none absolute inset-0 m-0 overflow-hidden whitespace-pre ${className} text-slate-100`}>
        {highlightPython(value)}
        {'\n'}
      </pre>
      <textarea
        value={value}
        wrap="off"
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        onScroll={(e) => {
          if (!under.current) return
          under.current.scrollTop = e.currentTarget.scrollTop
          under.current.scrollLeft = e.currentTarget.scrollLeft
        }}
        {...rest}
        className={`relative block w-full whitespace-pre bg-transparent text-transparent caret-sky-400 selection:bg-sky-400/30 ${className}`}
      />
    </div>
  )
}
