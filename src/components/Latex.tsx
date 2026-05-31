import { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

// KaTeX 进行 latex 渲染能力
interface LatexProps {
  formula: string
  displayMode?: boolean
  className?: string
}

export const Latex = ({ formula, displayMode = false, className = '' }: LatexProps) => {
  const containerRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      katex.render(formula, containerRef.current, {
        throwOnError: false,
        displayMode,
      })
    }
  }, [formula, displayMode])

  return <span ref={containerRef} className={className} />
}
