import { useMemo, useState } from 'react'
import { classificationMetrics, confusionFromLabels } from '../../lib/metrics'
import { fmt } from '../format'
import { Ans, Card, LabPage, Note, Presets, Seg, Stat, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  title: string
  note: string
  labels?: [string, string]
  counts?: { classes: string[]; m: number[][] }
}

const PRESETS: Preset[] = [
  { id: 'three', title: 'Exam-style: 3 classes', note: 'Pattern of 2024 Spring Q3 (labels changed)', labels: ['0 1 2 2 0 1 1 0 2 1', '0 2 2 1 0 1 0 0 2 2'] },
  { id: 'disease', title: 'Exam-style: rare disease', note: 'Pattern of Final 2022 Q3(a): accuracy vs F1', counts: { classes: ['yes', 'no'], m: [[2, 18], [6, 100]] } },
  { id: 'mcc', title: 'Lecture 3: MCC example', note: 'TP 90 · FP 4 · TN 1 · FN 5', counts: { classes: ['pos', 'neg'], m: [[90, 5], [4, 1]] } },
  { id: 'lazy', title: 'Lazy model: always “no”', note: '95% accuracy, finds nothing', counts: { classes: ['yes', 'no'], m: [[0, 5], [0, 95]] } },
]

type Role = 'TP' | 'FN' | 'FP' | 'TN'
const ROLE_STYLE: Record<Role, string> = {
  TP: 'bg-emerald-100 border-emerald-500 text-emerald-700',
  FN: 'bg-rose-100 border-rose-400 text-rose-700',
  FP: 'bg-amber-100 border-amber-500 text-amber-700',
  TN: 'bg-slate-100 border-slate-200 text-slate-500',
}
const frac = (n: number, d: number) => (d ? `${n}/${d}` : '—')

export default function Evaluation() {
  const [preset, setPreset] = useState<string | null>('three')
  const [labels, setLabels] = useState<[string, string]>(PRESETS[0].labels!)
  const [manual, setManual] = useState<{ classes: string[]; m: number[][] } | null>(null)
  const [rows, setRows] = useState<'actual' | 'pred'>('actual')
  const [sel, setSel] = useState(0)

  const parsed = useMemo(() => {
    if (manual) return { ...manual, error: null }
    try {
      return { ...confusionFromLabels(labels[0], labels[1]), error: null }
    } catch (e) {
      return { classes: [], m: [], error: (e as Error).message }
    }
  }, [labels, manual])

  const { classes, m } = parsed
  const s = Math.min(sel, Math.max(0, classes.length - 1))
  const R = classes.length ? classificationMetrics(m) : null
  const c = R?.perClass[s]
  const rowsActual = rows === 'actual'
  const role = (a: number, p: number): Role => (a === s && p === s ? 'TP' : a === s ? 'FN' : p === s ? 'FP' : 'TN')
  const setCell = (a: number, p: number, v: number) => {
    const next = m.map((r) => [...r])
    next[a][p] = Math.max(0, Math.round(v || 0))
    setManual({ classes, m: next })
    setPreset(null)
  }

  return (
    <LabPage id="evaluation" quiz lead="Accuracy hides which class the model fails on. Pick a class and the matrix splits into its true positives, false negatives, false positives and true negatives.">
      <Workspace
        controls={
          <>
            <Card title="Example">
              <Presets
                items={PRESETS}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((q) => q.id === id)!
                  setPreset(id)
                  setSel(0)
                  if (p.counts) {
                    setManual({ classes: [...p.counts.classes], m: p.counts.m.map((r) => [...r]) })
                    setLabels(['', ''])
                  } else {
                    setManual(null)
                    setLabels(p.labels!)
                  }
                }}
              />
            </Card>
            <Card title="Labels" sub={manual ? 'matrix edited by hand' : undefined}>
              <div className={`grid gap-2.5 ${manual ? 'opacity-50' : ''}`}>
                {(['Actual', 'Predicted'] as const).map((name, i) => (
                  <label key={name} className="grid gap-1">
                    <span className="text-xs font-bold text-slate-500">{name}</span>
                    <input
                      type="text"
                      spellCheck={false}
                      value={labels[i]}
                      onChange={(e) => {
                        const next: [string, string] = [...labels]
                        next[i] = e.target.value
                        setLabels(next)
                        setManual(null)
                        setPreset(null)
                      }}
                      className="rounded-[10px] border border-slate-300 px-2.5 py-1.5 font-mono text-sm outline-none focus:ring-2 focus:ring-blue-300"
                    />
                  </label>
                ))}
              </div>
              <p className={`mt-1.5 text-xs ${parsed.error ? 'text-rose-600' : 'text-slate-400'}`}>{parsed.error ?? 'Comma or space separated — or type straight into the matrix cells.'}</p>
            </Card>
            <Card>
              <Seg label="Matrix rows are" value={rows} onChange={setRows} options={[{ v: 'actual', label: 'Actual' }, { v: 'pred', label: 'Predicted' }]} />
              <p className="mt-2 text-xs text-slate-400">
                The 2024 Spring midterm put <b>actual</b> on rows; the 2023 Fall midterm put <b>predicted</b> on rows. Read the axis labels before you sum.
              </p>
            </Card>
          </>
        }
      >
        {!R || !c ? (
          <Note tone="bad" title="Nothing to score yet.">{parsed.error}</Note>
        ) : (
          <>
            <div className="grid gap-5 xl:grid-cols-2">
              <Card step={1} title="Confusion matrix" sub="click a class name">
                <div className="overflow-x-auto">
                  <table className="mx-auto border-separate border-spacing-1">
                    <tbody>
                      <tr>
                        <td colSpan={2} />
                        <th colSpan={classes.length} className="pb-1 text-xs font-extrabold uppercase tracking-widest text-indigo-600">{rowsActual ? 'Predicted' : 'Actual'} →</th>
                      </tr>
                      <tr>
                        <td colSpan={2} />
                        {classes.map((name, j) => (
                          <th key={j}>
                            <button type="button" onClick={() => setSel(j)} aria-pressed={j === s} className={`rounded-lg px-2.5 py-1 text-xs font-extrabold ${j === s ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-blue-50'}`}>{name}</button>
                          </th>
                        ))}
                      </tr>
                      {classes.map((name, i) => (
                        <tr key={i}>
                          {i === 0 && (
                            <th rowSpan={classes.length} className="pr-1 text-xs font-extrabold uppercase tracking-widest text-indigo-600 [writing-mode:vertical-rl] rotate-180">{rowsActual ? 'Actual' : 'Predicted'} →</th>
                          )}
                          <th>
                            <button type="button" onClick={() => setSel(i)} aria-pressed={i === s} className={`rounded-lg px-2.5 py-1 text-xs font-extrabold ${i === s ? 'bg-blue-600 text-white' : 'text-slate-600 hover:bg-blue-50'}`}>{name}</button>
                          </th>
                          {classes.map((_, j) => {
                            const a = rowsActual ? i : j
                            const p = rowsActual ? j : i
                            const r = role(a, p)
                            return (
                              <td key={j} className={`relative h-[58px] w-[74px] rounded-xl border-2 text-center ${ROLE_STYLE[r]}`}>
                                <span className="absolute left-1.5 top-0.5 text-[10px] font-extrabold">{r}</span>
                                <input
                                  type="number"
                                  min={0}
                                  value={m[a][p]}
                                  onChange={(e) => setCell(a, p, Number(e.target.value))}
                                  aria-label={`actual ${classes[a]}, predicted ${classes[p]}`}
                                  className="w-14 bg-transparent text-center font-mono text-xl font-extrabold text-slate-900 outline-none"
                                />
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-3 flex flex-wrap justify-center gap-4 text-xs text-slate-600">
                  <span>■ <b className="text-emerald-600">TP</b> correct</span>
                  <span>■ <b className="text-rose-600">FN</b> missed</span>
                  <span>■ <b className="text-amber-600">FP</b> false alarm</span>
                  <span>■ <b className="text-slate-500">TN</b> everything else</span>
                </div>
              </Card>

              <Card step={2} title={`Class “${classes[s]}” in one picture`}>
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                  <Stat k="TP" v={c.TP} d={`said ${classes[s]}, right`} tone="good" />
                  <Stat k="FN" v={c.FN} d={`was ${classes[s]}, missed`} tone="bad" />
                  <Stat k="FP" v={c.FP} d={`said ${classes[s]}, wrong`} tone="warn" />
                  <Stat k="TN" v={c.TN} d="everything else" />
                </div>
                <div className="formula mt-3">
                  {`Precision = TP / (TP + FP) = ${c.TP} / ${c.TP + c.FP} = `}<b className="text-blue-700">{fmt(c.precision, 3)}</b>
                  {`\nRecall    = TP / (TP + FN) = ${c.TP} / ${c.TP + c.FN} = `}<b className="text-blue-700">{fmt(c.recall, 3)}</b>
                  {`\nF1        = 2·TP / (2·TP + FP + FN) = ${2 * c.TP} / ${2 * c.TP + c.FP + c.FN} = `}<b className="text-rose-600">{fmt(c.f1, 3)}</b>
                </div>
                <p className="mt-2 text-[13px] text-slate-400">
                  Precision asks “when I say {classes[s]}, am I right?” Recall asks “of all real {classes[s]}, how many did I find?”
                </p>
              </Card>
            </div>

            <Card step={3} title="Per-class table" sub="Turn on “Quiz me” to fill it in">
              <TableWrap>
                <tr>
                  <th className="left">Class</th><th>TP</th><th>TN</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th>
                </tr>
                {R.perClass.map((p, i) => (
                  <tr key={i} className={i === s ? 'cur' : ''}>
                    <td className="left">{classes[i]}</td>
                    <Ans k={`tp${i}`} v={p.TP} />
                    <Ans k={`tn${i}`} v={p.TN} />
                    <Ans k={`fp${i}`} v={p.FP} />
                    <Ans k={`fn${i}`} v={p.FN} />
                    <Ans k={`pr${i}`} v={frac(p.TP, p.TP + p.FP)} />
                    <Ans k={`re${i}`} v={frac(p.TP, p.TP + p.FN)} />
                    <Ans k={`f1${i}`} v={fmt(p.f1, 3)} />
                  </tr>
                ))}
              </TableWrap>
            </Card>

            <Card step={4} title="Summary scores">
              <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4 xl:grid-cols-5">
                <Stat k="Accuracy" v={fmt(R.accuracy, 3)} d={`${R.perClass.reduce((a, p) => a + p.TP, 0)} / ${R.total} on the diagonal`} tone="blue" />
                <Stat k="Macro-F1" v={fmt(R.macroF1, 3)} d="mean of per-class F1" tone={R.macroF1 < R.accuracy - 0.15 ? 'bad' : 'none'} />
                <Stat k="Weighted-F1" v={fmt(R.weightedF1, 3)} d="F1 weighted by support" />
                <Stat k="Micro-F1" v={fmt(R.accuracy, 3)} d="= accuracy (one label each)" />
                {R.mcc !== null && <Stat k="MCC" v={fmt(R.mcc, 3)} d="−1 … 0 random … 1" tone={Math.abs(R.mcc) < 0.3 ? 'bad' : 'good'} />}
              </div>
              <div className="mt-3">
                {R.accuracy - R.macroF1 > 0.15 ? (
                  <Note tone="bad" title="Accuracy is flattering this model.">
                    It is {fmt((R.accuracy - R.macroF1) * 100, 1)} points above macro-F1 because one class dominates.{' '}
                    {R.mcc !== null && <>MCC = {fmt(R.mcc, 2)} says it is {Math.abs(R.mcc) < 0.3 ? 'close to a random guess' : 'still informative'}.</>}
                  </Note>
                ) : (
                  <Note tone="good">Accuracy and F1 agree here, so the classes are reasonably balanced.</Note>
                )}
              </div>
            </Card>

            <div className="grid gap-5 md:grid-cols-2">
              <Card title="Same thing in NumPy" sub="2024 Spring Q3(e) pattern">
                <div className="formula">
                  <span className="text-slate-400"># cm rows = {rowsActual ? 'actual' : 'predicted'}, columns = {rowsActual ? 'predicted' : 'actual'}</span>
                  {'\naccuracy  = cm.diagonal().sum() / cm.sum()\nprecision = cm.diagonal() / cm.sum(axis='}<b className="text-blue-700">{rowsActual ? 0 : 1}</b>
                  {')  '}<span className="text-slate-400"># ÷ predicted totals</span>
                  {'\nrecall    = cm.diagonal() / cm.sum(axis='}<b className="text-blue-700">{rowsActual ? 1 : 0}</b>
                  {')  '}<span className="text-slate-400"># ÷ actual totals</span>
                  {'\nmacro_f1  = np.mean(2 * precision * recall / (precision + recall))'}
                </div>
              </Card>
              <Note tone="warn" title="Exam traps.">
                (1) Which axis is actual? Precision divides by the <i>predicted</i> total, recall by the <i>actual</i> total. (2) Macro-F1 is the plain average of per-class F1 — not F1 of the averaged precision and recall. (3) On imbalanced data accuracy can be high while the rare class is never found; MCC and F1 expose it.
              </Note>
            </div>
          </>
        )}
      </Workspace>
    </LabPage>
  )
}
