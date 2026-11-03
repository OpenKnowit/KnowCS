import { useMemo, useState } from 'react'
import { m } from 'framer-motion'
import { ChevronLeft, ChevronRight, Footprints } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { planBroadcast, shapeLabel } from '../../lib/broadcast'
import { ML_USES, PRESETS, QUIZ, makeCode, quizAnswer } from '../../lib/broadcastDemo'
import type { Op } from '../../lib/broadcastDemo'
import { runPython } from '../../lib/minipy'
import { FailView, RuleTable, ShapeEditor, StretchView } from '../../modules/NumpyBroadcast'
import { NumpyPairwiseVideo } from '../../modules/NumpyPairwiseVideo'
import { Latex } from '../../components/Latex'
import { PyBlock, PyInline } from '../../components/PyCode'
import { RunInPlayground } from '../../components/RunInPlayground'
import { SeniorAdvice } from '../../components/SeniorAdvice'
import { Ans, Btn, Card, LabPage, Presets, TableWrap, Workspace } from '../ui'

// L1 · NumPy broadcasting, taught in order: the idea (a smaller array is stretched, not copied), the rule (line the
// shapes up on the right, then compare axis by axis), the stretch itself, the code, where the course uses it, the exam
// question (which shape comes out?), the traps, and the pairwise-distance trick.

/** [1 2 3] + 10: the scalar's ghost copies under each element, then the sum */
function IdeaPicture() {
  const cell = 'grid h-9 w-9 place-items-center rounded border font-mono text-[12px]'
  return (
    <div className="flex flex-wrap items-center justify-center gap-3 py-2">
      <div className="flex gap-1">{[1, 2, 3].map((v) => <span key={v} className={`${cell} border-blue-600 bg-blue-500 text-white`}>{v}</span>)}</div>
      <span className="text-xl font-bold text-slate-300">+</span>
      <div className="flex gap-1">
        <span className={`${cell} border-orange-600 bg-orange-500 text-white`}>10</span>
        <span className={`${cell} border-dashed border-orange-300 bg-orange-50 text-orange-400`}>10</span>
        <span className={`${cell} border-dashed border-orange-300 bg-orange-50 text-orange-400`}>10</span>
      </div>
      <span className="text-xl font-bold text-slate-300">=</span>
      <div className="flex gap-1">{[11, 12, 13].map((v) => <span key={v} className={`${cell} border-emerald-600 bg-emerald-500 text-white`}>{v}</span>)}</div>
    </div>
  )
}

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
  const preset = PRESETS.find((p) => p.a.join() === a.join() && p.b.join() === b.join() && p.op === op)?.id ?? null

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
            <Presets
              title={t('lab.common.example')}
              items={PRESETS.map((p) => ({ id: p.id, title: t(`numpy_module.broadcast.preset.${p.id}`), note: t(`numpy_module.broadcast.story.${p.id}`) }))}
              value={preset}
              onPick={(id) => {
                const p = PRESETS.find((x) => x.id === id)!
                load(p.a, p.b, p.op)
              }}
            />
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
          </>
        }
      >
        <Card step={1} title={t('lab.bc.idea_title')}>
          <p className="text-sm leading-relaxed text-slate-700">{t('lab.bc.idea')}</p>
          <IdeaPicture />
          <p className="text-xs text-slate-500">{t('lab.bc.idea_note')}</p>
        </Card>

        <Card step={2} title={t('lab.bc.rule_title')} sub={t('lab.bc.rule_sub')}>
          <ol className="mb-4 grid gap-2 text-sm text-slate-700 sm:grid-cols-2">
            <li className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <b className="mr-1 text-blue-700">①</b>
              {t('lab.bc.rule_1')}
              <pre className="mt-2 font-mono text-[12px] leading-5 text-slate-600">{'A:  3 × 4\nB:      4   →  1 × 4'}</pre>
            </li>
            <li className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <b className="mr-1 text-blue-700">②</b>
              {t('lab.bc.rule_2')}
              <ul className="mt-1.5 space-y-0.5 text-[13px]">
                <li>✓ {t('lab.bc.rule_eq')}</li>
                <li>✓ {t('lab.bc.rule_one')}</li>
                <li className="text-rose-700">✗ {t('lab.bc.rule_bad')}</li>
              </ul>
            </li>
          </ol>
          <div className="rounded-xl border border-slate-200 bg-white p-3">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-semibold text-slate-700">{t('lab.bc.check', { a: shapeLabel(a), b: shapeLabel(b) })}</span>
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
                    <span className="w-12 text-center font-mono text-xs text-gray-500">
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
            <RuleTable a={a} b={b} steps={plan.steps} phase={ph} />
            <m.p
              key={`${ph}-${a.join()}-${b.join()}`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              aria-live="polite"
              className={`mt-3 rounded-lg px-3 py-2 text-sm ${done && !plan.ok ? 'bg-rose-50 text-rose-700' : done ? 'bg-emerald-50 text-emerald-800' : 'bg-yellow-50 text-yellow-900'}`}
            >
              {narration}
            </m.p>
          </div>
        </Card>

        <Card step={3} title={plan.ok ? t('lab.bc.stretch_title', { shape: shapeLabel(plan.outShape!) }) : t('lab.bc.fail_title')} sub={plan.ok ? t('lab.bc.stretch_sub') : undefined}>
          {plan.ok ? <StretchView key={`${a.join()}|${b.join()}`} a={a} b={b} op={op} out={plan.outShape!} /> : <FailView a={a} b={b} op={op} onFix={(na, nb) => load(na, nb)} />}
        </Card>

        <Card step={4} title={t('lab.bc.code_title')} sub={t('lab.bc.code_sub')}>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-xl bg-slate-900 p-3">
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

        <Card step={5} title={t('numpy_module.broadcast.cards.ml_t')} sub={t('lab.bc.uses_sub')}>
          <ul className="grid gap-2 md:grid-cols-2">
            {ML_USES.map((u) => (
              <li key={u.id} className="flex items-start justify-between gap-3 rounded-xl border border-slate-200 p-3">
                <div className="min-w-0">
                  <PyInline tone="light" code={u.code} className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[12px] text-slate-800" />
                  <span className="mt-1 block text-xs text-slate-500">
                    <span className="font-mono">{u.shapes}</span> · {t(`numpy_module.broadcast.cards.ml_${u.id}`)}
                  </span>
                </div>
                <span className="shrink-0 whitespace-nowrap">
                  <Btn onClick={() => load(u.a, u.b, u.op)}>{t('lab.bc.try')}</Btn>
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card step={6} title={t('lab.bc.quiz_title')} sub={t('lab.common.quiz_sub')}>
          <p className="mb-3 text-sm text-slate-600">{t('lab.bc.quiz_lead')}</p>
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
        </Card>

        <Card step={7} title={t('numpy_module.broadcast.cards.pit_t')}>
          <p className="text-sm leading-relaxed text-slate-700">{t('numpy_module.broadcast.cards.pit')}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Btn onClick={() => load([3], [3, 1], '+')}>{t('lab.bc.pit_try')}</Btn>
            <Btn onClick={() => load([3, 4], [3], '+')}>{t('lab.bc.mismatch_try')}</Btn>
          </div>
          <p className="mt-3 text-xs text-slate-500">{t('numpy_module.broadcast.cards.mem')}</p>
        </Card>

        <Card step={8} title={t('numpy_module.pairwise.title')}>
          <NumpyPairwiseVideo />
          <SeniorAdvice content={t('numpy_module.pairwise.advice')} />
        </Card>

        <SeniorAdvice content={<Trans i18nKey="numpy_module.broadcast.advice" components={{ 1: <Latex formula="1" /> }} />} />
      </Workspace>
    </LabPage>
  )
}
