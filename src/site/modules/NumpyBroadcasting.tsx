import { useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight, Footprints } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { planBroadcast, shapeLabel } from '../../lib/broadcast'
import { ML_USES, PRESETS, QUIZ, makeCode, quizAnswer } from '../../lib/broadcastDemo'
import type { Op } from '../../lib/broadcastDemo'
import { runPython } from '../../lib/minipy'
import { FailView, RuleTable, ShapeEditor, StretchView } from '../../modules/NumpyBroadcast'
import { CallView } from '../../modules/NpCallView'
import { Latex } from '../../components/Latex'
import { PyBlock } from '../../components/PyCode'
import { RunInPlayground } from '../../components/RunInPlayground'
import { Motion } from '../../components/Motion'
import { SeniorAdvice } from '../../components/SeniorAdvice'
import { highlightPython } from '../playground/pyHighlight'
import { Ans, Btn, Card, LabPage, Note, Presets, TableWrap, Workspace } from '../ui'

// L1 · NumPy broadcasting. One pair of shapes drives the page: ① check the rule axis by axis, ② see the stretch (or the
// clash and its fix) with the same thing in code, ③ the exam drill and the (n,) vs (n, 1) trap, ④ the course's classic
// use, pairwise distances, walked through one traced line at a time.

/** every example in one list: the teaching pairs, then where the course uses broadcasting */
const EXAMPLES = [...PRESETS.map((p) => ({ ...p, kind: 'pair' as const })), ...ML_USES.map((u) => ({ id: `use-${u.id}`, a: u.a, b: u.b, op: u.op, kind: 'use' as const, use: u }))]

// ---------------------------------------------------------------- ④ pairwise distances, step by step

const PAIR_CODE = `import numpy as np
X_train = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
X_test = np.array([[5, 6], [7, 8]])
A = X_test[:, None, :]
B = X_train[None, :, :]
diff = A - B
sq = diff ** 2
d2 = sq.sum(axis=-1)
D = np.sqrt(d2)
nearest = np.argsort(D, axis=1)[:, :2]`
const PAIR_LINES = PAIR_CODE.split('\n')

function PairwiseWalkthrough() {
  const { t } = useTranslation()
  const run = useMemo(() => runPython(PAIR_CODE), [])
  // the traced calls from the reshaping on (the two np.array lines only create the data)
  const steps = useMemo(() => run.calls.filter((c) => c.line >= 4), [run])
  const [k, setK] = useState(0)
  const step = steps[k]
  const go = (i: number) => setK(Math.max(0, Math.min(steps.length - 1, i)))
  return (
    <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
      {/* the program: the current step's line is marked; click any line to jump to it */}
      <div className="min-w-0 overflow-hidden rounded-xl bg-slate-900">
        <div className="flex items-center justify-between gap-2 border-b border-slate-700/70 px-3 py-2 text-[11px] text-slate-400">
          {t('lab.bc.pw_code')}
          <RunInPlayground lib="numpy" code={`${PAIR_CODE}\nprint(D)\nprint(nearest)`} />
        </div>
        <ol className="overflow-x-auto py-2 font-mono text-[12.5px] leading-6">
          {PAIR_LINES.map((line, i) => {
            const n = i + 1
            const at = steps.findIndex((s) => s.line === n)
            const on = step?.line === n
            return (
              <li key={i}>
                <button
                  type="button"
                  disabled={at < 0}
                  onClick={() => go(at)}
                  className={`flex w-full min-w-max items-center gap-3 pr-3 text-left ${on ? 'bg-sky-400/15 shadow-[inset_2px_0_0_#38bdf8]' : at >= 0 ? 'hover:bg-white/5' : 'cursor-default'}`}
                >
                  <span className={`w-8 shrink-0 select-none text-right ${on ? 'font-bold text-sky-300' : 'text-slate-500'}`}>{n}</span>
                  <span className="whitespace-pre text-slate-100">{highlightPython(line)}</span>
                </button>
              </li>
            )
          })}
        </ol>
        <p className="border-t border-slate-700/70 px-3 py-2 text-[11px] leading-relaxed text-slate-400">
          {t('lab.bc.pw_oneliner')} <code className="text-slate-200">{highlightPython('np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=-1))')}</code>
        </p>
      </div>

      {/* the step: what this line does, then its inputs and output (hover any cell to see where it came from) */}
      <div className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => go(k - 1)} disabled={k === 0} aria-label={t('numpy_module.broadcast.back')} className="rounded-lg border border-slate-200 p-1.5 disabled:opacity-40">
            <ChevronLeft size={16} aria-hidden />
          </button>
          <span className="font-mono text-xs text-slate-500">
            {k + 1} / {steps.length}
          </span>
          <button type="button" onClick={() => go(k + 1)} disabled={k === steps.length - 1} aria-label={t('numpy_module.broadcast.next')} className="rounded-lg border border-slate-200 p-1.5 disabled:opacity-40">
            <ChevronRight size={16} aria-hidden />
          </button>
          <span className="ml-1 rounded-full bg-sky-100 px-2.5 py-0.5 text-xs font-bold text-sky-800">L{step?.line}</span>
        </div>
        <p className="rounded-lg bg-sky-50 px-3 py-2 text-sm leading-relaxed text-sky-950" aria-live="polite">
          {t(`lab.bc.pw_step.${k}`)}
        </p>
        {step && <CallView key={k} call={step} />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- the page

export default function NumpyBroadcasting() {
  const { t } = useTranslation()
  const [a, setA] = useState<number[]>(PRESETS[2].a)
  const [b, setB] = useState<number[]>(PRESETS[2].b)
  const [op, setOp] = useState<Op>(PRESETS[2].op)
  // the rule walk: null = all axes at once; 0 = shapes lined up; k = axis -k checked
  const [phase, setPhase] = useState<number | null>(null)

  const plan = useMemo(() => planBroadcast(a, b), [a, b])
  const code = makeCode(a, b, op)
  const run = useMemo(() => runPython(code), [code])
  const load = (na: number[], nb: number[], nop: Op = op) => {
    setA(na)
    setB(nb)
    setOp(nop)
    setPhase(null)
  }
  const example = EXAMPLES.find((p) => p.a.join() === a.join() && p.b.join() === b.join() && p.op === op)?.id ?? null

  const nd = plan.ndim
  const failAt = plan.steps.findIndex((s) => s.verdict === 'mismatch')
  const lastPhase = failAt >= 0 ? failAt + 1 : nd + 1
  const ph = phase ?? lastPhase
  const done = ph >= lastPhase
  const summary = plan.ok ? t('numpy_module.broadcast.narr.done_ok', { shape: shapeLabel(plan.outShape!) }) : t('numpy_module.broadcast.narr.done_fail', { axis: plan.steps[failAt].axis })
  const narration =
    phase === null
      ? summary
      : ph === 0
        ? t('numpy_module.broadcast.narr.align', { a: shapeLabel(a), b: shapeLabel(b) })
        : ph <= nd
          ? (() => {
              const s = plan.steps[ph - 1]
              return t(`numpy_module.broadcast.narr.${s.verdict}`, { axis: s.axis, a: s.a, b: s.b, out: s.out })
            })()
          : summary

  return (
    <LabPage id="numpy-broadcast" quiz>
      <Workspace
        controls={
          <>
            <Card title={t('lab.bc.shapes')}>
              <div className="grid gap-2.5">
                <ShapeEditor name="A" shape={a} onChange={(s) => load(s, b)} />
                <div className="flex justify-center gap-1" role="group" aria-label={t('numpy_module.broadcast.op')}>
                  {(['+', '-', '*'] as Op[]).map((o) => (
                    <button key={o} type="button" onClick={() => setOp(o)} aria-pressed={op === o} className={`h-9 w-9 rounded-lg font-mono text-lg font-bold ${op === o ? 'bg-slate-800 text-white' : 'border border-gray-200 bg-white text-gray-500'}`}>
                      {o === '*' ? '×' : o === '-' ? '−' : '+'}
                    </button>
                  ))}
                </div>
                <ShapeEditor name="B" shape={b} onChange={(s) => load(a, s)} />
              </div>
            </Card>
            <Presets
              title={t('lab.common.example')}
              items={EXAMPLES.map((p) =>
                p.kind === 'pair'
                  ? { id: p.id, title: t(`numpy_module.broadcast.preset.${p.id}`), note: t(`numpy_module.broadcast.story.${p.id}`) }
                  : { id: p.id, title: t(`numpy_module.broadcast.cards.ml_${p.use.id}`), note: `${p.use.code}   ${p.use.shapes}` },
              )}
              value={example}
              onPick={(id) => {
                const p = EXAMPLES.find((x) => x.id === id)!
                load(p.a, p.b, p.op)
              }}
            />
          </>
        }
      >
        <Card step={1} title={t('lab.bc.check', { a: shapeLabel(a), b: shapeLabel(b) })}>
          <p className="text-sm leading-relaxed text-slate-700">{t('lab.bc.idea')}</p>
          <div className="mt-2.5 flex flex-wrap items-center gap-2 text-[13px]">
            <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-slate-700">
              <b className="text-blue-700">①</b> {t('lab.bc.rule_1')}
            </span>
            <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-slate-700">
              <b className="text-blue-700">②</b> {t('lab.bc.rule_2')} <span className="text-emerald-700">{t('lab.bc.rule_eq')}</span> · <span className="text-emerald-700">{t('lab.bc.rule_one')}</span> · <span className="text-rose-700">{t('lab.bc.rule_bad')}</span>
            </span>
          </div>
          <div className="mt-4 flex flex-wrap items-start gap-4">
            <div className="min-w-0 flex-1">
              <RuleTable a={a} b={b} steps={plan.steps} phase={ph} />
            </div>
            <div className="flex items-center gap-1.5">
              {phase === null ? (
                <button type="button" onClick={() => setPhase(0)} className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-500">
                  <Footprints size={14} aria-hidden /> {t('numpy_module.broadcast.walk')}
                </button>
              ) : (
                <>
                  <button type="button" onClick={() => setPhase(Math.max(0, ph - 1))} disabled={ph === 0} aria-label={t('numpy_module.broadcast.back')} className="rounded-lg border border-gray-200 p-1.5 disabled:opacity-40">
                    <ChevronLeft size={16} aria-hidden />
                  </button>
                  <span className="w-10 text-center font-mono text-xs text-gray-500">
                    {ph}/{lastPhase}
                  </span>
                  <button type="button" onClick={() => setPhase(Math.min(lastPhase, ph + 1))} disabled={done} aria-label={t('numpy_module.broadcast.next')} className="rounded-lg border border-gray-200 p-1.5 disabled:opacity-40">
                    <ChevronRight size={16} aria-hidden />
                  </button>
                  <button type="button" onClick={() => setPhase(null)} className="rounded-lg px-2 py-1.5 text-xs text-gray-500 hover:text-gray-800">
                    {t('numpy_module.broadcast.show_all')}
                  </button>
                </>
              )}
            </div>
          </div>
          <p
            aria-live="polite"
            className={`mt-3 rounded-lg px-3 py-2 text-sm ${done && !plan.ok ? 'bg-rose-50 text-rose-700' : done ? 'bg-emerald-50 text-emerald-800' : 'bg-yellow-50 text-yellow-900'}`}
          >
            {narration}
          </p>
        </Card>

        <Card step={2} title={plan.ok ? t('lab.bc.stretch_title', { shape: shapeLabel(plan.outShape!) }) : t('lab.bc.fail_title')} sub={plan.ok ? t('lab.bc.stretch_sub') : undefined}>
          {plan.ok ? <StretchView key={`${a.join()}|${b.join()}`} a={a} b={b} op={op} out={plan.outShape!} /> : <FailView a={a} b={b} op={op} onFix={(na, nb) => load(na, nb)} />}
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="min-w-0 rounded-xl bg-slate-900 p-3">
              <div className="mb-1 flex items-center justify-between gap-2 text-[11px] text-slate-400">
                {t('numpy_module.broadcast.code_title')}
                <RunInPlayground lib="numpy" code={code} />
              </div>
              <PyBlock code={code} className="overflow-x-auto whitespace-pre font-mono text-[12.5px] leading-5 text-slate-100" />
            </div>
            <div className="min-w-0 rounded-xl border border-gray-200 bg-white p-3">
              <div className="mb-1 text-[11px] text-gray-500">{t('numpy_module.broadcast.output_title')}</div>
              {run.error ? (
                <p className="whitespace-pre-wrap break-words font-mono text-[12.5px] leading-5 text-rose-600">
                  <b>{run.error.type}</b>: {run.error.message}
                </p>
              ) : (
                <pre className="overflow-x-auto whitespace-pre font-mono text-[12.5px] leading-5 text-sky-700">{run.out}</pre>
              )}
            </div>
          </div>
        </Card>

        <Card step={3} title={t('lab.bc.quiz_title')} sub={t('lab.common.quiz_sub')}>
          <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
            <div>
              <p className="mb-2 text-sm text-slate-600">{t('lab.bc.quiz_lead')}</p>
              <TableWrap>
                <tr>
                  <th>A</th>
                  <th>B</th>
                  <th>{t('lab.bc.quiz_result')}</th>
                  <th />
                </tr>
                {QUIZ.map(([qa, qb], i) => (
                  <tr key={i}>
                    <td>{shapeLabel(qa)}</td>
                    <td>{shapeLabel(qb)}</td>
                    <Ans k={`bc${i}`} v={quizAnswer(qa, qb)} />
                    <td>
                      <button type="button" onClick={() => load(qa, qb, '+')} className="text-xs font-semibold text-blue-600 hover:underline">
                        {t('lab.bc.check_it')}
                      </button>
                    </td>
                  </tr>
                ))}
              </TableWrap>
            </div>
            <Note tone="warn" title={t('numpy_module.broadcast.cards.pit_t')}>
              {t('numpy_module.broadcast.cards.pit')}
              <span className="mt-2 flex flex-wrap gap-2">
                <Btn onClick={() => load([3], [3, 1], '+')}>{t('lab.bc.pit_try')}</Btn>
                <Btn onClick={() => load([3, 4], [3], '+')}>{t('lab.bc.mismatch_try')}</Btn>
              </span>
            </Note>
          </div>
        </Card>

        <Card step={4} title={t('lab.bc.pw_title')} sub={t('lab.bc.pw_sub')}>
          <PairwiseWalkthrough />
        </Card>

        {/* SeniorAdvice animates with framer-motion's lazy features, which only <Motion> provides */}
        <Motion>
          <SeniorAdvice content={<Trans i18nKey="numpy_module.broadcast.advice" components={{ 1: <Latex formula="1" /> }} />} />
        </Motion>
      </Workspace>
    </LabPage>
  )
}
