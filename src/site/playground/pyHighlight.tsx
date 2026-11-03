import type { ReactNode } from 'react'

// Python syntax colours for the dark code editor. One left-to-right scan: whichever token starts first wins,
// so a # inside a string stays part of the string.

const KEYWORDS = 'and|as|assert|async|await|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield'
const TOKEN = new RegExp(
  [
    String.raw`(?<str>[rbfuRBFU]{0,2}(?:"""[\s\S]*?(?:"""|$(?![\s\S]))|'''[\s\S]*?(?:'''|$(?![\s\S]))|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?))`,
    String.raw`(?<com>#[^\n]*)`,
    String.raw`(?<dec>@[A-Za-z_][\w.]*)`,
    String.raw`(?<kw>\b(?:${KEYWORDS})\b)`,
    String.raw`(?<fn>\b[A-Za-z_]\w*(?=\s*\())`,
    String.raw`(?<num>\b(?:0[xob][\da-fA-F_]+|\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?j?|\.\d+(?:[eE][+-]?\d+)?)\b)`,
  ].join('|'),
  'g',
)
const DARK: Record<string, string> = {
  str: 'text-emerald-300',
  com: 'italic text-slate-500',
  dec: 'text-pink-300',
  kw: 'text-violet-300',
  fn: 'text-sky-300',
  num: 'text-orange-300',
}
// the same roles on a light background (inline code chips on white cards)
const LIGHT: Record<string, string> = {
  str: 'text-emerald-700',
  com: 'italic text-slate-500',
  dec: 'text-pink-700',
  kw: 'text-violet-700',
  fn: 'text-sky-700',
  num: 'text-orange-700',
}

export function highlightPython(code: string, tone: 'dark' | 'light' = 'dark'): ReactNode[] {
  const CLASS = tone === 'dark' ? DARK : LIGHT
  const out: ReactNode[] = []
  let last = 0
  for (const m of code.matchAll(TOKEN)) {
    const kind = Object.entries(m.groups!).find(([, v]) => v !== undefined)?.[0]
    if (!kind || !m[0]) continue
    if (m.index > last) out.push(code.slice(last, m.index))
    out.push(
      <span key={m.index} className={CLASS[kind]}>
        {m[0]}
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < code.length) out.push(code.slice(last))
  return out
}
