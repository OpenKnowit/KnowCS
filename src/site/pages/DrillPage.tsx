import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { DRILL } from '../../data/drill'
import type { DrillItem } from '../../data/drill'
import { LECTURES, lectureScope, moduleUrl, watchUrl } from '../../lib/sitemap'

const STORE = 'knowcs-drill'
const load = (): Record<string, boolean> => {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? '{}') as Record<string, boolean>
  } catch {
    return {}
  }
}
const save = (v: Record<string, boolean>) => {
  try {
    localStorage.setItem(STORE, JSON.stringify(v))
  } catch {
    /* storage blocked: answers just are not remembered */
  }
}

function Item({ d, chosen, onChoose }: { d: DrillItem; chosen: boolean | undefined; onChoose: (v: boolean) => void }) {
  const { t } = useTranslation()
  const answered = chosen !== undefined
  const right = answered && chosen === d.answer
  const btn = (v: boolean) => {
    const picked = chosen === v
    const tone = !answered ? 'border-slate-300 bg-white text-slate-700 hover:border-blue-400 hover:text-blue-700' : v === d.answer ? 'border-emerald-500 bg-emerald-50 text-emerald-800' : picked ? 'border-rose-400 bg-rose-50 text-rose-700' : 'border-slate-200 bg-white text-slate-500'
    return (
      <button type="button" onClick={() => onChoose(v)} aria-pressed={picked} className={`min-w-[84px] rounded-full border-2 px-4 py-1.5 text-sm font-black transition ${tone}`}>
        {t(v ? 'drill.true' : 'drill.false')}
      </button>
    )
  }
  return (
    <li className={`rounded-2xl border bg-white p-4 shadow-sm sm:p-5 ${answered ? (right ? 'border-emerald-200' : 'border-rose-200') : 'border-slate-200'}`}>
      <p className="text-[15px] font-semibold leading-relaxed text-slate-900">{t(`drill.items.${d.id}.q`)}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {btn(true)}
        {btn(false)}
        {answered && (
          <span className={`text-sm font-black ${right ? 'text-emerald-700' : 'text-rose-600'}`} role="status">
            {right ? t('drill.correct') : t('drill.wrong')} · {t('drill.answer_is', { a: t(d.answer ? 'drill.true' : 'drill.false') })}
          </span>
        )}
      </div>
      {answered && (
        <div className="mt-3 rounded-xl bg-slate-50 p-3 text-[14px] leading-relaxed text-slate-700">
          {t(`drill.items.${d.id}.why`)}{' '}
          <a href={d.link.kind === 'watch' ? watchUrl(d.link.id) : moduleUrl(d.link.id)} className="whitespace-nowrap font-bold text-blue-700 hover:underline">
            {d.link.kind === 'watch' ? t('drill.watch', { title: t(`watch.${d.link.id}.title`) }) : t('drill.practise', { title: t(`site.modules.${d.link.id}.title`) })} →
          </a>
        </div>
      )}
    </li>
  )
}

/** /drill/: original true/false statements grouped by lecture, with explanations and links. */
export default function DrillPage() {
  const { t } = useTranslation()
  const [answers, setAnswers] = useState<Record<string, boolean>>(load)
  const [scope, setScope] = useState<'all' | 'mid' | 'final'>('all')
  const [onlyWrong, setOnlyWrong] = useState(false)
  useEffect(() => {
    document.title = `${t('drill.title')} · KnowCS`
  }, [t])
  useEffect(() => save(answers), [answers])

  const done = Object.keys(answers).filter((id) => DRILL.some((d) => d.id === id)).length
  const right = DRILL.filter((d) => answers[d.id] === d.answer).length
  const visible = useMemo(
    () => DRILL.filter((d) => (scope === 'all' || lectureScope(d.lec) === scope) && (!onlyWrong || (answers[d.id] !== undefined && answers[d.id] !== d.answer))),
    [scope, onlyWrong, answers],
  )

  return (
    <main className="mx-auto max-w-[920px] px-4 pb-12 pt-6 sm:px-8">
      <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600">{t('drill.eyebrow')}</p>
      <h1 className="mt-2 text-[clamp(30px,4vw,44px)] font-black leading-tight tracking-tight">{t('drill.title')}</h1>
      <p className="mt-2 text-[16px] leading-relaxed text-slate-600">{t('drill.lead')}</p>

      <div className="sticky top-0 z-30 -mx-4 mt-5 border-b border-slate-200 bg-[#f1f5f9]/90 px-4 py-3 backdrop-blur sm:-mx-8 sm:px-8">
        <div className="flex flex-wrap items-center gap-2">
          <div className="grid grid-cols-3 gap-1 rounded-xl bg-white p-1 text-sm font-bold shadow-sm" role="group" aria-label={t('site.home.scope')}>
            {(['all', 'mid', 'final'] as const).map((v) => (
              <button key={v} type="button" onClick={() => setScope(v)} aria-pressed={scope === v} className={`rounded-lg px-3 py-1 ${scope === v ? 'bg-slate-900 text-white' : 'text-slate-500'}`}>
                {t(`site.home.scope_${v}`)}
              </button>
            ))}
          </div>
          <button type="button" onClick={() => setOnlyWrong(!onlyWrong)} aria-pressed={onlyWrong} className={`rounded-full border px-3 py-1.5 text-sm font-bold ${onlyWrong ? 'border-rose-400 bg-rose-50 text-rose-700' : 'border-slate-300 bg-white text-slate-600'}`}>
            {t(onlyWrong ? 'drill.show_all' : 'drill.only_wrong')}
          </button>
          <button type="button" onClick={() => setAnswers({})} className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm font-bold text-slate-600 hover:border-slate-400">
            {t('drill.reset')}
          </button>
          <span className="ml-auto text-sm font-bold text-slate-600" aria-live="polite">
            {t('drill.score', { right, done, total: DRILL.length })}
          </span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200">
          <div className="h-full bg-emerald-500 transition-all" style={{ width: `${(right / DRILL.length) * 100}%` }} />
        </div>
      </div>

      {LECTURES.filter((n) => visible.some((d) => d.lec === n)).map((n) => (
        <section key={n} className="mt-7" aria-labelledby={`drill-${n}`}>
          <h2 id={`drill-${n}`} className="mb-3 text-sm font-black uppercase tracking-wider text-slate-500">
            {t('site.lecture_long', { n })} · {t(`site.lectures.${n}`)}
          </h2>
          <ol className="grid gap-3">
            {visible
              .filter((d) => d.lec === n)
              .map((d) => (
                <Item key={d.id} d={d} chosen={answers[d.id]} onChoose={(v) => setAnswers((a) => ({ ...a, [d.id]: v }))} />
              ))}
          </ol>
        </section>
      ))}
    </main>
  )
}
