import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from 'react'
import { ArrowUp, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Eye, EyeOff, Play } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { planBroadcast, shapeLabel, suggestFix } from '../../lib/broadcast'
import { BROADCAST_STEPS, COURSE_CODE, EXAM_QUESTIONS, PRESETS, QUIZ, SYNTAXES, constructCode, quotedShape, makeQuoted, makePairwise, makeProgram, opSign, quizAnswer, withoutOnes } from '../../lib/broadcastDemo'
import type { QuoteFix, QuotedCode, Op, Syntax } from '../../lib/broadcastDemo'
import { runPython } from '../../lib/minipy'
import { InputView, RuleTable, ShapeEditor, StretchView } from '../../modules/NumpyBroadcast'
import { CallView } from '../../modules/NpCallView'
import { Latex } from '../../components/Latex'
import { PyBlock, PyInline } from '../../components/PyCode'
import { RunInPlayground } from '../../components/RunInPlayground'
import { Motion } from '../../components/Motion'
import { SeniorAdvice } from '../../components/SeniorAdvice'
import { highlightPython } from '../playground/pyHighlight'
import { Ans, LabPage, PageBar, QuizToggle, TableWrap } from '../ui'

// L1 · NumPy broadcasting, two screens the page snaps between. The dashboard: shapes and what to construct on the left,
// the program in the middle, the rule worked through on the right. The notes: concept, the four APIs, the mechanism, the exam.

type Example = { id: string; title: string; note: string; a: number[]; b: number[]; op: Op; program?: boolean; exam?: QuotedCode }

const readNum = (key: string, fallback: number, min: number, max: number) => {
  try {
    const v = Number(localStorage.getItem(key))
    return v >= min && v <= max ? v : fallback
  } catch {
    return fallback
  }
}
const save = (key: string, v: number | string) => {
  try {
    localStorage.setItem(key, String(v))
  } catch {
    // private mode: not remembered
  }
}
const CODE_KEY = 'knowcs-bc-code-width'
const OUT_KEY = 'knowcs-bc-output-height'
const HIDE_KEY = 'knowcs-bc-output-hidden'
const grip = 'hidden shrink-0 touch-none rounded-full bg-slate-200 transition-colors hover:bg-blue-400 focus-visible:bg-blue-500 focus-visible:outline-none lg:block'

// ---------------------------------------------------------------- the Example menu

function ExampleMenu({ groups, value, onPick }: { groups: { title: string; items: Example[] }[]; value: string | null; onPick: (e: Example) => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const away = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', away)
    document.addEventListener('keydown', esc)
    return () => {
      document.removeEventListener('pointerdown', away)
      document.removeEventListener('keydown', esc)
    }
  }, [open])
  return (
    <div ref={box} className="relative">
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-haspopup="true" className="flex items-center gap-1 rounded-full border border-slate-300 bg-white px-3 py-1 text-[13px] font-bold text-slate-700 hover:border-slate-400">
        {t('lab.common.example')}
        <ChevronDown size={14} aria-hidden className={open ? 'rotate-180' : ''} />
      </button>
      {open && (
        <div className="absolute left-0 top-full z-40 mt-1.5 max-h-[min(70dvh,560px)] w-[min(400px,calc(100vw-48px))] overflow-y-auto rounded-xl border border-slate-200 bg-white p-2 shadow-xl">
          {groups.map((g) => (
            <div key={g.title} className="mb-1 last:mb-0">
              <div className="px-2 pb-1 pt-1.5 text-[11px] font-bold uppercase tracking-wide text-slate-400">{g.title}</div>
              {g.items.map((e) => (
                <button
                  key={e.id}
                  type="button"
                  onClick={() => {
                    onPick(e)
                    setOpen(false)
                  }}
                  aria-current={value === e.id}
                  className={`block w-full rounded-lg px-2 py-1.5 text-left ${value === e.id ? 'bg-blue-50' : 'hover:bg-slate-100'}`}
                >
                  <span className={`block text-[13px] font-semibold ${value === e.id ? 'text-blue-700' : 'text-slate-800'}`}>{e.title}</span>
                  <span className="block text-xs leading-snug text-slate-500">{e.note}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------- the notes screen

const rich = { b: <b className="font-bold text-slate-900" />, c: <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.92em] text-slate-800" /> }
const T = ({ k, values }: { k: string; values?: Record<string, string> }) => <Trans i18nKey={k} values={values} components={rich} />

function Panel({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <section className="flex min-h-0 min-w-0 flex-col rounded-[18px] border border-slate-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-2.5">
        <h2 className="text-[15px] font-extrabold text-slate-900">{title}</h2>
        {action}
      </div>
      <div className="relative min-h-0 flex-1 space-y-2.5 px-4 py-3 text-[13.5px] leading-relaxed text-slate-700 lg:overflow-y-auto">{children}</div>
    </section>
  )
}

/** shapes lined up on the right, the way the rule reads them; try loads the pair into the dashboard */
function Aligned({ rows, onTry }: { rows: [string, (number | string)[]][]; onTry?: () => void }) {
  const { t } = useTranslation()
  const cols = Math.max(...rows.map(([, d]) => d.length))
  return (
    <div className="flex items-start justify-between gap-3 rounded-lg bg-slate-50 px-3 py-2">
      <div className="grid items-baseline gap-x-2 gap-y-0.5 text-[13px]" style={{ gridTemplateColumns: `auto repeat(${cols}, 2.2ch)` }}>
        {rows.map(([label, dims]) => (
          <div key={label} className="contents">
            <span className="pr-2 text-slate-500">{label}</span>
            {Array.from({ length: cols }, (_, i) => dims[i - (cols - dims.length)]).map((d, i) => (
              <span key={i} className={`text-right font-mono ${d === 1 ? 'text-blue-600' : 'text-slate-800'}`}>
                {d ?? ''}
              </span>
            ))}
          </div>
        ))}
      </div>
      {onTry && (
        <button type="button" onClick={onTry} className="shrink-0 text-xs font-semibold text-blue-600 hover:underline">
          {t('lab.bc.try')}
        </button>
      )}
    </div>
  )
}

function Notes({ load, onBack }: { load: (a: number[], b: number[], op: Op) => void; onBack: () => void }) {
  const { t } = useTranslation()
  const LABEL = { a: 'A', b: 'B', pad: t('lab.bc.mech.padded'), out: t('lab.bc.mech.result') }
  const API = [
    ['reshape', 'a.reshape(1, 3)', 'a.reshape(3, 1)'],
    ['None', 'a[None, :]', 'a[:, None]'],
    ['np.newaxis', 'a[np.newaxis, :]', 'a[:, np.newaxis]'],
    ['expand_dims', 'np.expand_dims(a, axis=0)', 'np.expand_dims(a, axis=1)'],
  ]
  return (
    <div className="grid min-h-0 gap-3 lg:h-full lg:grid-cols-2 lg:grid-rows-2">
      <Panel title={t('lab.bc.concept.title')}>
        <p><T k="lab.bc.concept.p1" /></p>
        <p><T k="lab.bc.concept.p2" /></p>
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.concept.why')}</h3>
        <ol className="list-decimal space-y-1 pl-5">
          <li><b className="font-bold text-slate-900">{t('lab.bc.concept.why_1')}</b></li>
          <li><b className="font-bold text-slate-900">{t('lab.bc.concept.why_2')}</b></li>
          <li><b className="font-bold text-slate-900">{t('lab.bc.concept.why_3')}</b></li>
        </ol>
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.concept.how')}</h3>
        <p><T k="lab.bc.concept.how_p" /></p>
      </Panel>

      <Panel title={t('lab.bc.api.title')}>
        <p><T k="lab.bc.api.lead" /></p>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-[12.5px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-1 pr-3 font-semibold">{t('lab.bc.api.method')}</th>
                <th className="py-1 pr-3 font-semibold">{t('lab.bc.api.row')} <code className="font-mono">(1, 3)</code></th>
                <th className="py-1 font-semibold">{t('lab.bc.api.col')} <code className="font-mono">(3, 1)</code></th>
              </tr>
            </thead>
            <tbody>
              {API.map(([name, row, col]) => (
                <tr key={name} className="border-b border-slate-100">
                  <td className="py-1 pr-3 font-mono font-semibold text-slate-800">{name}</td>
                  <td className="py-1 pr-3"><PyInline tone="light" code={row} className="whitespace-nowrap font-mono" /></td>
                  <td className="py-1"><PyInline tone="light" code={col} className="whitespace-nowrap font-mono" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-slate-500"><T k="lab.bc.api.base" /></p>
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.api.none_t')}</h3>
        <p><T k="lab.bc.api.none_p" /></p>
        <PyBlock tone="light" code={'a = np.zeros((2, 3))\n\na[None, :, :].shape  # (1, 2, 3)\na[:, None, :].shape  # (2, 1, 3)\na[:, :, None].shape  # (2, 3, 1)'} className="overflow-x-auto rounded-lg bg-slate-50 px-3 py-2 font-mono text-[12px] leading-5" />
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.api.expand_t')}</h3>
        <p><T k="lab.bc.api.expand_p" /></p>
        <PyBlock tone="light" code={'np.expand_dims(a, axis=0).shape   # (1, 2, 3)\nnp.expand_dims(a, axis=1).shape   # (2, 1, 3)\nnp.expand_dims(a, axis=2).shape   # (2, 3, 1)\nnp.expand_dims(a, axis=-1).shape  # (2, 3, 1)'} className="overflow-x-auto rounded-lg bg-slate-50 px-3 py-2 font-mono text-[12px] leading-5" />
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.api.reshape_t')}</h3>
        <p><T k="lab.bc.api.reshape_p" /></p>
        <PyBlock tone="light" code={'a = np.arange(6)       # (6,)\n\na.reshape(2, 3)        # (2, 3)\na.reshape(1, 6, 1)     # (1, 6, 1)\na.reshape(-1, 1)       # (6, 1)'} className="overflow-x-auto rounded-lg bg-slate-50 px-3 py-2 font-mono text-[12px] leading-5" />
        <p><T k="lab.bc.api.memory" /></p>
        <p className="rounded-lg bg-blue-50 px-3 py-2 text-blue-950"><T k="lab.bc.api.pick" /></p>
      </Panel>

      <Panel title={t('lab.bc.mech.title')}>
        <p><T k="lab.bc.mech.rule" /></p>
        <ul className="list-disc space-y-0.5 pl-5">
          <li>{t('lab.bc.mech.cond_1')}</li>
          <li><T k="lab.bc.mech.cond_2" /></li>
          <li><T k="lab.bc.mech.cond_3" /></li>
        </ul>
        <p><T k="lab.bc.mech.then" /></p>
        <Aligned rows={[[LABEL.a, [3, 4]], [LABEL.b, [4]], [LABEL.pad, [1, 4]], [LABEL.out, [3, 4]]]} onTry={() => load([3, 4], [4], '+')} />
        <p className="text-slate-600"><T k="lab.bc.mech.ex1" /></p>
        <Aligned rows={[[LABEL.a, [2, 1, 4]], [LABEL.b, [3, 1]], [LABEL.pad, [1, 3, 1]], [LABEL.out, [2, 3, 4]]]} onTry={() => load([2, 1, 4], [3, 1], '+')} />
        <p className="text-slate-600"><T k="lab.bc.mech.ex2" /></p>
        <Aligned rows={[[LABEL.a, [3, 4]], [LABEL.b, [3]]]} onTry={() => load([3, 4], [3], '+')} />
        <p className="text-slate-600"><T k="lab.bc.mech.ex3" /></p>
        <p className="rounded-lg bg-blue-50 px-3 py-2 font-semibold text-blue-950">{t('lab.bc.mech.remember')}</p>
        {/* SeniorAdvice animates with framer-motion's lazy features, which only <Motion> provides */}
        <Motion>
          <SeniorAdvice content={<Trans i18nKey="numpy_module.broadcast.advice" components={{ 1: <Latex formula="1" /> }} />} />
        </Motion>
      </Panel>

      <Panel title={t('lab.bc.exam.title')} action={<QuizToggle />}>
        <p><T k="lab.bc.exam.lead" /></p>
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.exam.t1')}</h3>
        <p><T k="lab.bc.exam.t1_p" /></p>
        <TableWrap>
          <tr>
            <th>A</th>
            <th>B</th>
            <th>{t('lab.bc.exam.result')}</th>
            <th />
          </tr>
          {QUIZ.map(([qa, qb], i) => (
            <tr key={i}>
              <td>{shapeLabel(qa)}</td>
              <td>{shapeLabel(qb)}</td>
              <Ans k={`bc${i}`} v={quizAnswer(qa, qb)} />
              <td>
                <button type="button" onClick={() => load(qa, qb, '+')} className="text-xs font-semibold text-blue-600 hover:underline">
                  {t('lab.bc.try')}
                </button>
              </td>
            </tr>
          ))}
        </TableWrap>
        <p className="rounded-lg bg-amber-50 px-3 py-2 text-amber-950"><T k="lab.bc.exam.trap" /></p>
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.exam.t2')}</h3>
        <p><T k="lab.bc.exam.t2_p" /></p>
        <PyBlock tone="light" code={'d = np.sqrt(np.sum((X_test[:, None] - X_train[None]) ** 2, axis=-1))\n# (m, 1, d) - (1, n, d) → (m, n, d), sum → (m, n)'} className="overflow-x-auto rounded-lg bg-slate-50 px-3 py-2 font-mono text-[12px] leading-5" />
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.exam.t3')}</h3>
        <p><T k="lab.bc.exam.t3_p" /></p>
        <h3 className="pt-1 text-[13px] font-extrabold text-slate-900">{t('lab.bc.exam.marks')}</h3>
        <ul className="list-disc space-y-0.5 pb-10 pl-5">
          <li><T k="lab.bc.exam.mark_1" /></li>
          <li><T k="lab.bc.exam.mark_2" /></li>
          <li><T k="lab.bc.exam.mark_3" /></li>
          <li><T k="lab.bc.exam.mark_4" /></li>
          <li><T k="lab.bc.exam.mark_5" /></li>
        </ul>
      </Panel>

      <button type="button" onClick={onBack} aria-label={t('lab.bc.back')} title={t('lab.bc.back')} className="absolute bottom-5 right-5 z-20 hidden items-center gap-1.5 rounded-full bg-blue-600 px-3.5 py-2 text-sm font-bold text-white shadow-lg hover:bg-blue-700 lg:flex">
        <ArrowUp size={15} aria-hidden />
        <Play size={13} aria-hidden />
        {t('lab.bc.run')}
      </button>
    </div>
  )
}

// ---------------------------------------------------------------- the page

export default function NumpyBroadcasting() {
  const { t } = useTranslation()
  const first = PRESETS.find((p) => p.id === 'outer')!
  const [a, setA] = useState<number[]>(first.a)
  const [b, setB] = useState<number[]>(first.b)
  const [op, setOp] = useState<Op>(first.op)
  const [syntax, setSyntax] = useState<Syntax>('none')
  /** the pairwise-distance program instead of the generated one */
  const [pairwise, setPairwise] = useState(false)
  /** code quoted from a paper, the slides or a lab: its own code and data instead of the generated program */
  const [exam, setExam] = useState<QuotedCode | null>(null)
  const [quoteFix, setQuoteFix] = useState<QuoteFix>(null)
  const [step, setStep] = useState(3)
  const [pwStep, setPwStep] = useState(0)
  const [codeW, setCodeW] = useState(() => readNum(CODE_KEY, 35, 25, 65))
  const [outH, setOutH] = useState(() => readNum(OUT_KEY, 150, 70, 460))
  const [outHidden, setOutHidden] = useState(() => {
    try {
      return localStorage.getItem(HIDE_KEY) === '1'
    } catch {
      return false
    }
  })
  const scroller = useRef<HTMLDivElement>(null)
  const dash = useRef<HTMLDivElement>(null)
  const notes = useRef<HTMLDivElement>(null)
  const row = useRef<HTMLDivElement>(null)
  const right = useRef<HTMLDivElement>(null)
  const smooth = () => (window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth') as ScrollBehavior
  const toDash = () => dash.current?.scrollIntoView({ behavior: smooth(), block: 'start' })
  const toNotes = () => notes.current?.scrollIntoView({ behavior: smooth(), block: 'start' })

  const plan = useMemo(() => planBroadcast(a, b), [a, b])
  const program = useMemo(() => (pairwise ? makePairwise(syntax) : exam ? makeQuoted(exam, syntax, op, quoteFix) : makeProgram(a, b, op, syntax)), [pairwise, exam, quoteFix, a, b, op, syntax])
  const run = useMemo(() => runPython(program.code), [program.code])
  const traced = useMemo(() => (pairwise ? run.calls.filter((c) => c.line >= 4 && c.line <= 10) : []), [pairwise, run])
  // a clash stops at the rule: there is nothing to stretch
  const st = plan.ok ? step : Math.min(step, 1)
  const fix = useMemo(() => suggestFix(a, b), [a, b])

  const change = (na: number[], nb: number[], nop: Op = op) => {
    setA(na)
    setB(nb)
    setOp(nop)
    setPairwise(false)
    setExam(null)
    setQuoteFix(null)
    setStep(3)
  }
  const pickFrom = (na: number[], nb: number[], nop: Op) => {
    change(na, nb, nop)
    toDash()
  }

  const groups = useMemo(() => {
    const basic: Example[] = PRESETS.map((p) => ({ ...p, title: t(`numpy_module.broadcast.preset.${p.id}`), note: t(`numpy_module.broadcast.story.${p.id}`) }))
    const paper = (id: string) => ({ title: t(`lab.bc.papers.${id}.title`), note: t(`lab.bc.papers.${id}.note`) })
    const quoted = (q: QuotedCode): Example => ({ id: q.id, a: quotedShape(q.l, 'l', null), b: quotedShape(q.r, 'r', null), op: q.op, exam: q, ...paper(q.id) })
    const course = COURSE_CODE.map(quoted)
    const exam: Example[] = [
      { id: 'pairwise', a: [2, 1, 2], b: [1, 4, 2], op: '-', program: true, ...paper('pairwise') },
      ...EXAM_QUESTIONS.map((q) => ({ id: q.id, a: quotedShape(q.l, 'l', null), b: quotedShape(q.r, 'r', null), op: q.op, exam: q, ...paper(q.id) })),
    ]
    return [
      { title: t('lab.bc.group_basic'), items: basic },
      { title: t('lab.bc.group_course'), items: course },
      { title: t('lab.bc.group_exam'), items: exam },
    ]
  }, [t])
  const current = pairwise
    ? 'pairwise'
    : exam
      ? exam.id
      : (groups.flatMap((g) => g.items).find((e) => !e.program && !e.exam && e.a.join() === a.join() && e.b.join() === b.join() && e.op === op)?.id ?? null)
  const pick = (e: Example) => {
    change(e.a, e.b, e.op)
    if (e.exam?.syntax) setSyntax(e.exam.syntax)
    if (e.exam) setExam(e.exam)
    if (e.program) {
      setSyntax('none')
      setPairwise(true)
      setPwStep(0)
    }
  }
  const applyFix = (side: 'A' | 'B', shape: number[]) => {
    if (side === 'A') setA(shape)
    else setB(shape)
    if (exam) setQuoteFix({ side: side === 'A' ? 'l' : 'r', shape })
    setStep(3)
  }
  // a question's own variables and values; the demo's otherwise
  const data = exam ? { a: exam.l.values, b: exam.r.values, nameA: exam.l.name, nameB: exam.r.name } : undefined
  const nameA = data?.nameA ?? 'A'
  const nameB = data?.nameB ?? 'B'
  const nameOf = (side: 'A' | 'B') => (side === 'A' ? nameA : nameB)
  const rawName = (side: 'A' | 'B') => {
    const o = side === 'A' ? exam?.l : exam?.r
    return o ? (o.variable ?? o.name) : side.toLowerCase()
  }

  // the code and the visual follow each other: clicking a line picks its step, a step lights its lines
  const lineOn = (i: number) => {
    if (pairwise) return traced[pwStep]?.line === i + 1
    return program.lines[i].steps.includes(st)
  }
  const lineTarget = (i: number): number | null => {
    if (pairwise) {
      const at = traced.findIndex((c) => c.line === i + 1)
      return at >= 0 ? at : null
    }
    const s = program.lines[i].steps
    if (!s.length) return null
    return plan.ok ? s[0] : Math.min(s[0], 1)
  }
  const goLine = (i: number) => {
    const to = lineTarget(i)
    if (to === null) return
    if (pairwise) setPwStep(to)
    else setStep(to)
  }

  const rail = pairwise
    ? traced.map((c, i) => ({ key: i, label: `L${c.line}`, disabled: false }))
    : BROADCAST_STEPS.map((s, i) => ({ key: i, label: t(`lab.bc.steps.${s}`), disabled: !plan.ok && i > 1 }))
  const at = pairwise ? pwStep : st
  const goStep = (i: number) => {
    if (pairwise) setPwStep(Math.max(0, Math.min(traced.length - 1, i)))
    else setStep(Math.max(0, Math.min(plan.ok ? 3 : 1, i)))
  }
  const last = pairwise ? traced.length - 1 : plan.ok ? 3 : 1

  const dragCode = (e: ReactPointerEvent) => {
    const box = row.current?.getBoundingClientRect()
    if (!box || !(e.buttons & 1)) return
    const v = Math.max(25, Math.min(65, ((e.clientX - box.left) / box.width) * 100))
    setCodeW(v)
    save(CODE_KEY, v)
  }
  const keyCode = (e: ReactKeyboardEvent) => {
    const d = e.key === 'ArrowLeft' ? -3 : e.key === 'ArrowRight' ? 3 : 0
    if (!d) return
    e.preventDefault()
    const v = Math.max(25, Math.min(65, codeW + d))
    setCodeW(v)
    save(CODE_KEY, v)
  }
  const dragOut = (e: ReactPointerEvent) => {
    const box = right.current?.getBoundingClientRect()
    if (!box || !(e.buttons & 1)) return
    const v = Math.max(70, Math.min(Math.min(460, box.height - 160), box.bottom - e.clientY))
    setOutH(v)
    save(OUT_KEY, v)
  }
  const keyOut = (e: ReactKeyboardEvent) => {
    const d = e.key === 'ArrowUp' ? 24 : e.key === 'ArrowDown' ? -24 : 0
    if (!d) return
    e.preventDefault()
    const v = Math.max(70, Math.min(460, outH + d))
    setOutH(v)
    save(OUT_KEY, v)
  }
  const hideOutput = (hidden: boolean) => {
    setOutHidden(hidden)
    save(HIDE_KEY, hidden ? '1' : '0')
  }

  const failAt = plan.steps.findIndex((s) => s.verdict === 'mismatch')
  const summary = plan.ok ? t('numpy_module.broadcast.narr.done_ok', { shape: shapeLabel(plan.outShape!) }) : t('numpy_module.broadcast.narr.done_fail', { axis: plan.steps[failAt].axis })
  const fixTo = fix?.shape ?? []
  const padded = [
    { name: nameA, shape: a, to: plan.steps.map((s) => s.a).reverse() },
    { name: nameB, shape: b, to: plan.steps.map((s) => s.b).reverse() },
  ].filter((p) => p.shape.length !== p.to.length && p.shape.length > 0)

  return (
    <LabPage id="numpy-broadcast" quiz dashboard>
      <main className="lg:-mb-10">
        <div ref={scroller} className="lg:h-[calc(100dvh-66px)] lg:snap-y lg:snap-mandatory lg:overflow-y-auto">
          {/* ---------- the dashboard ---------- */}
          <div ref={dash} className="relative flex flex-col gap-3 p-3 lg:h-full lg:snap-start lg:snap-always lg:flex-row">
            <aside className="relative z-30 flex shrink-0 flex-col rounded-[18px] border border-slate-200 bg-white lg:w-[300px]">
              <div className="space-y-2 border-b border-slate-100 p-3">
                <PageBar id="numpy-broadcast" bare />
                <ExampleMenu groups={groups} value={current} onPick={pick} />
              </div>
              <div className="min-h-0 flex-1 space-y-2 p-3 lg:overflow-y-auto">
                <ShapeEditor name="A" label={nameA} shape={a} onChange={(s) => change(s, b)} />
                <div className="flex justify-center gap-1" role="group" aria-label={t('numpy_module.broadcast.op')}>
                  {(['+', '-', '*', '/'] as Op[]).map((o) => (
                    <button key={o} type="button" onClick={() => setOp(o)} aria-pressed={op === o} className={`h-7 w-7 rounded-lg font-mono text-sm font-bold ${op === o ? 'bg-slate-800 text-white' : 'border border-gray-200 bg-white text-gray-500'}`}>
                      {opSign(o)}
                    </button>
                  ))}
                </div>
                <ShapeEditor name="B" label={nameB} shape={b} onChange={(s) => change(a, s)} />

                {/* what to construct, and how, in the spelling picked under the code */}
                <div className="rounded-xl bg-slate-50 p-3 text-[13px] leading-snug text-slate-700">
                  <div className="mb-1.5 text-xs font-extrabold uppercase tracking-wide text-slate-500">{t('lab.bc.help.title')}</div>
                  {!plan.ok && !pairwise ? (
                    <>
                      <p className="text-rose-700">{t('lab.bc.help.clash', { axis: plan.steps[failAt].axis, a: plan.steps[failAt].a, b: plan.steps[failAt].b })}</p>
                      {fix ? (
                        <div className="mt-2 space-y-1.5">
                          <p>{t('lab.bc.help.make', { name: nameOf(fix.side), from: shapeLabel(withoutOnes(fixTo)), to: shapeLabel(fixTo) })}</p>
                          <PyInline tone="light" code={constructCode(rawName(fix.side), fixTo, syntax) ?? ''} className="block overflow-x-auto whitespace-nowrap rounded bg-white px-2 py-1 font-mono text-xs" />
                          <button type="button" onClick={() => applyFix(fix.side, fixTo)} className="rounded-lg bg-emerald-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-emerald-500">
                            {t('numpy_module.broadcast.fix_btn')}
                          </button>
                        </div>
                      ) : (
                        <p className="mt-2">{t('lab.bc.help.no_fix')}</p>
                      )}
                    </>
                  ) : (
                    <div className="space-y-2">
                      {program.constructs.length === 0 ? (
                        <p>{t('lab.bc.help.none')}</p>
                      ) : (
                        program.constructs.map((c) => (
                          <div key={c.name} className="space-y-1">
                            <p>{t('lab.bc.help.make', { name: c.name, from: shapeLabel(c.from), to: shapeLabel(c.to) })}</p>
                            <PyInline tone="light" code={c.code} className="block overflow-x-auto whitespace-nowrap rounded bg-white px-2 py-1 font-mono text-xs" />
                          </div>
                        ))
                      )}
                      {!pairwise && padded.map((p) => <p key={p.name} className="text-slate-500">{t('lab.bc.help.pad', { name: p.name, from: shapeLabel(p.shape), to: shapeLabel(p.to) })}</p>)}
                      <p>{t('lab.bc.help.then', { code: pairwise ? 'A - B' : `${nameA} ${op} ${nameB}` })}</p>
                    </div>
                  )}
                </div>
              </div>
            </aside>

            <div ref={row} className="flex min-h-0 min-w-0 flex-1 flex-col gap-3 lg:flex-row lg:gap-0" style={{ ['--bc-code' as string]: `${codeW}%` }}>
              {/* ---------- the code ---------- */}
              <section className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-[18px] bg-slate-900 lg:w-[var(--bc-code)] lg:shrink-0">
                <div className="flex items-center justify-between gap-2 border-b border-slate-700/70 px-3 py-2 text-[12px] font-semibold text-slate-300">
                  {t('lab.bc.code')}
                  <span className="truncate text-[11px] font-normal text-slate-500">{t('lab.bc.code_hint')}</span>
                </div>
                <ol className="min-h-0 flex-1 overflow-auto py-2 font-mono text-[12.5px] leading-6">
                  {program.lines.map((line, i) => {
                    const on = lineOn(i)
                    const can = lineTarget(i) !== null
                    return (
                      <li key={i}>
                        <button type="button" disabled={!can} onClick={() => goLine(i)} className={`flex w-full min-w-max items-center gap-3 pr-3 text-left ${on ? 'bg-sky-400/15 shadow-[inset_2px_0_0_#38bdf8]' : can ? 'hover:bg-white/5' : 'cursor-default'}`}>
                          <span className={`w-8 shrink-0 select-none text-right ${on ? 'font-bold text-sky-300' : 'text-slate-500'}`}>{i + 1}</span>
                          <span className="whitespace-pre text-slate-100">{highlightPython(line.text)}</span>
                        </button>
                      </li>
                    )
                  })}
                </ol>
                <div className="flex justify-end px-3 pb-2">
                  <RunInPlayground lib="numpy" code={program.code} />
                </div>
                <div className="border-t border-slate-700/70 px-3 py-2">
                  <div className="mb-1.5 text-[11px] text-slate-400">{t('lab.bc.syntax')}</div>
                  <div className="flex flex-wrap gap-1" role="group" aria-label={t('lab.bc.syntax')}>
                    {SYNTAXES.map((s) => (
                      <button key={s.id} type="button" onClick={() => setSyntax(s.id)} aria-pressed={syntax === s.id} className={`rounded-md px-2 py-1 font-mono text-[11.5px] ${syntax === s.id ? 'bg-sky-400 font-bold text-slate-900' : 'bg-white/10 text-slate-300 hover:bg-white/15'}`}>
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              <div role="separator" aria-orientation="vertical" aria-label={t('lab.bc.resize_code')} tabIndex={0} onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)} onPointerMove={dragCode} onKeyDown={keyCode} className={`${grip} mx-1.5 my-auto h-16 w-1.5 cursor-col-resize`} />

              {/* ---------- the visual and the output ---------- */}
              <div ref={right} className="flex min-h-0 min-w-0 flex-1 flex-col">
                <section className="flex min-h-0 min-w-0 flex-1 rounded-[18px] border border-slate-200 bg-white">
                  <div className="min-h-0 min-w-0 flex-1 space-y-3 p-4 lg:overflow-y-auto">
                    {pairwise ? (
                      <>
                        <h2 className="text-[15px] font-extrabold">{t('lab.bc.pw_title')}</h2>
                        <p className="rounded-lg bg-sky-50 px-3 py-2 text-sm leading-relaxed text-sky-950" aria-live="polite">{t(`lab.bc.pw_step.${pwStep}`)}</p>
                        {traced[pwStep] && <CallView key={`${pwStep}-${syntax}`} call={traced[pwStep]} />}
                      </>
                    ) : (
                      <>
                        <h2 className="text-[15px] font-extrabold">{t(`lab.bc.steps.${BROADCAST_STEPS[st]}`)}</h2>
                        <p className="text-[13px] leading-relaxed text-slate-600">{t(`lab.bc.step_note.${BROADCAST_STEPS[st]}`, { shape: plan.ok ? shapeLabel(plan.outShape!) : '' })}</p>
                        {st === 0 && <InputView a={a} b={b} op={op} data={data} />}
                        {st === 1 && (
                          <>
                            <RuleTable a={a} b={b} nameA={nameA} nameB={nameB} steps={plan.steps} phase={plan.ndim + 1} />
                            <p aria-live="polite" className={`rounded-lg px-3 py-2 text-sm ${plan.ok ? 'bg-emerald-50 text-emerald-800' : 'bg-rose-50 text-rose-700'}`}>{summary}</p>
                          </>
                        )}
                        {st >= 2 && plan.ok && <StretchView key={`${exam?.id}|${a.join()}|${b.join()}|${st}`} a={a} b={b} op={op} out={plan.outShape!} withResult={st === 3} data={data} />}
                      </>
                    )}
                  </div>
                  {/* the step rail */}
                  <nav aria-label={t('lab.bc.step_rail')} className="flex w-[74px] shrink-0 flex-col items-stretch gap-1 overflow-y-auto border-l border-slate-100 px-1.5 py-3">
                    <button type="button" onClick={() => goStep(at - 1)} disabled={at === 0} aria-label={t('numpy_module.broadcast.back')} className="mx-auto rounded-md p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30">
                      <ChevronLeft size={15} aria-hidden className="rotate-90" />
                    </button>
                    {rail.map((r) => (
                      <button key={r.key} type="button" disabled={r.disabled} onClick={() => goStep(r.key)} aria-current={at === r.key ? 'step' : undefined} className={`flex flex-col items-center gap-0.5 rounded-lg py-1 text-center text-[10.5px] leading-tight ${at === r.key ? 'bg-blue-50 font-bold text-blue-700' : r.disabled ? 'text-slate-300' : 'text-slate-500 hover:bg-slate-100'}`}>
                        <span className={`grid h-5 w-5 place-items-center rounded-full text-[10px] font-bold ${at === r.key ? 'bg-blue-600 text-white' : r.disabled ? 'bg-slate-100 text-slate-300' : r.key < at ? 'bg-blue-100 text-blue-700' : 'bg-slate-200 text-slate-600'}`}>{r.key + 1}</span>
                        {r.label}
                      </button>
                    ))}
                    <button type="button" onClick={() => goStep(at + 1)} disabled={at === last} aria-label={t('numpy_module.broadcast.next')} className="mx-auto rounded-md p-1 text-slate-500 hover:bg-slate-100 disabled:opacity-30">
                      <ChevronRight size={15} aria-hidden className="rotate-90" />
                    </button>
                    {outHidden && (
                      <button type="button" onClick={() => hideOutput(false)} aria-label={t('lab.bc.show_output')} title={t('lab.bc.show_output')} className="mx-auto mt-auto rounded-md p-1.5 text-slate-500 hover:bg-slate-100">
                        <Eye size={15} aria-hidden />
                      </button>
                    )}
                  </nav>
                </section>
                {!outHidden && (
                  <>
                    <div role="separator" aria-orientation="horizontal" aria-label={t('lab.bc.resize_output')} tabIndex={0} onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)} onPointerMove={dragOut} onKeyDown={keyOut} className={`${grip} mx-auto my-1.5 h-1.5 w-16 cursor-row-resize`} />
                    <section className="mt-3 flex min-h-0 shrink-0 flex-col rounded-[18px] border border-slate-200 bg-white lg:mt-0 lg:h-[var(--bc-out)]" style={{ ['--bc-out' as string]: `${outH}px` }}>
                      <div className="flex items-center justify-between gap-2 px-4 pt-2 text-xs font-bold text-slate-500">
                        {t('numpy_module.broadcast.output_title')}
                        <button type="button" onClick={() => hideOutput(true)} aria-label={t('lab.bc.hide_output')} title={t('lab.bc.hide_output')} className="rounded-md p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600">
                          <EyeOff size={14} aria-hidden />
                        </button>
                      </div>
                      <div className="min-h-0 flex-1 overflow-auto px-4 pb-2 pr-12" aria-live="polite">
                        {(run.stdout || run.out) && <pre className="whitespace-pre font-mono text-[12px] leading-5 text-sky-700">{[run.stdout.trimEnd(), run.out].filter(Boolean).join('\n')}</pre>}
                        {run.error && (
                          <p className="whitespace-pre-wrap break-words font-mono text-[12px] leading-5 text-rose-600">
                            <b>{run.error.type}</b>: {run.error.message}
                          </p>
                        )}
                      </div>
                    </section>
                  </>
                )}
              </div>
            </div>

            <button type="button" onClick={toNotes} aria-label={t('lab.bc.to_notes')} title={t('lab.bc.to_notes')} className="absolute bottom-5 right-5 z-20 hidden h-9 w-9 place-items-center rounded-full bg-blue-600 text-white shadow-lg hover:bg-blue-700 lg:grid">
              <CircleHelp size={19} aria-hidden />
            </button>
          </div>

          {/* ---------- the notes ---------- */}
          <div ref={notes} id="concept" className="relative p-3 lg:h-full lg:snap-start lg:snap-always">
            <Notes load={pickFrom} onBack={toDash} />
          </div>
        </div>
      </main>
    </LabPage>
  )
}
