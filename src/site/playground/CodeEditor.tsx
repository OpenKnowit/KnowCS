import { useRef } from 'react'
import type { KeyboardEvent } from 'react'

// --- 带行号的 Python 编辑框：Tab / Shift+Tab 缩进 4 格，回车保持缩进（冒号后多缩进一级），Ctrl/⌘+Enter 立即运行，Esc 离开编辑框 ---

interface Props {
  value: string
  onChange: (v: string) => void
  onRun: () => void
  errorLine?: number
  label: string
  minRows?: number
  /** grow to fill the column (full-screen mode) */
  tall?: boolean
}

const INDENT = '    '

export const CodeEditor = ({ value, onChange, onRun, errorLine, label, minRows = 6, tall }: Props) => {
  const ref = useRef<HTMLTextAreaElement>(null)
  const lines = value.split('\n')

  /** replace the selection and put the caret at the end of the inserted text */
  const edit = (start: number, end: number, text: string, caret = start + text.length, caretEnd = caret) => {
    const next = value.slice(0, start) + text + value.slice(end)
    onChange(next)
    requestAnimationFrame(() => ref.current?.setSelectionRange(caret, caretEnd))
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    const ta = e.currentTarget
    const { selectionStart: s, selectionEnd: en } = ta
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      onRun()
      return
    }
    if (e.key === 'Escape') {
      // Tab indents inside the editor, so Esc lets keyboard users leave it (a second Esc closes full screen)
      e.stopPropagation()
      ta.blur()
      return
    }
    if (e.key === 'Tab') {
      e.preventDefault()
      const lineStart = value.lastIndexOf('\n', s - 1) + 1
      if (s === en && !e.shiftKey) return edit(s, en, INDENT)
      // indent / dedent every selected line
      const before = value.slice(lineStart, en).split('\n')
      const after = before.map((l) => (e.shiftKey ? l.replace(/^ {1,4}/, '') : INDENT + l))
      const out = after.join('\n')
      return edit(lineStart, en, out, Math.max(lineStart, s + after[0].length - before[0].length), lineStart + out.length)
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.altKey) {
      e.preventDefault()
      const lineStart = value.lastIndexOf('\n', s - 1) + 1
      const line = value.slice(lineStart, s)
      let indent = line.match(/^ */)![0]
      if (/:\s*(#.*)?$/.test(line)) indent += INDENT
      else if (/^\s*(return|pass|break|continue)\b/.test(line)) indent = indent.slice(0, Math.max(0, indent.length - 4))
      return edit(s, en, '\n' + indent)
    }
    if (e.key === 'Backspace' && s === en && s > 0) {
      const lineStart = value.lastIndexOf('\n', s - 1) + 1
      const before = value.slice(lineStart, s)
      if (before.length > 0 && /^ +$/.test(before) && before.length % 4 === 0) {
        e.preventDefault()
        edit(s - 4, s, '')
      }
    }
  }

  return (
    <div className={`flex overflow-hidden rounded-xl border border-slate-800 bg-slate-900 ${tall ? 'min-h-[18rem] flex-1' : ''}`}>
      <div aria-hidden className="select-none border-r border-slate-700/60 py-3 pl-3 pr-2 text-right font-mono text-[12.5px] leading-6 text-slate-500">
        {lines.map((_, i) => (
          <div key={i} className={errorLine === i + 1 ? 'font-bold text-rose-400' : ''}>{i + 1}</div>
        ))}
      </div>
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKey}
        rows={Math.max(lines.length, minRows)}
        wrap="off"
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        aria-label={label}
        className="min-w-0 flex-1 resize-none overflow-x-auto whitespace-pre bg-transparent px-3 py-3 font-mono text-[12.5px] leading-6 text-slate-100 caret-sky-400 outline-none"
      />
    </div>
  )
}
