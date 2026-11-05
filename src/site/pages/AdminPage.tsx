import { useEffect, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowDown, ArrowUp, ExternalLink, ImagePlus, LogOut, Pencil, Plus, Trash2, X } from 'lucide-react'
import { RESOURCE_CATS, RESOURCE_KINDS, inLang } from '../../lib/resources'
import type { ResourceData, ResourceItem } from '../../lib/resources'
import { previewSrc } from '../resourcePreviews'

// /admin/: the Resources list behind a passphrase (server/admin.mjs). Not linked from the site, not indexed.

type Draft = Pick<ResourceItem, 'url' | 'cat' | 'kind'> & { title: { en: string; zh: string }; desc: { en: string; zh: string } }
const EMPTY: Draft = { url: 'https://', cat: 'study', kind: 'web', title: { en: '', zh: '' }, desc: { en: '', zh: '' } }

async function api(path: string, init: RequestInit = {}): Promise<unknown> {
  const res = await fetch(path, { ...init, headers: init.body && !(init.body instanceof Blob) ? { 'Content-Type': 'application/json', ...init.headers } : init.headers })
  const body: unknown = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error((body as { error?: string }).error || `http_${res.status}`)
  return body
}

const input = 'w-full rounded-[10px] border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100'
const Field = ({ label, children }: { label: string; children: ReactNode }) => (
  <label className="grid gap-1">
    <span className="text-xs font-bold text-slate-500">{label}</span>
    {children}
  </label>
)

function Editor({ item, onClose, onSaved }: { item: ResourceItem | null; onClose: () => void; onSaved: (d: ResourceData) => void }) {
  const { t, i18n } = useTranslation()
  const [d, setD] = useState<Draft>(() => (item ? { url: item.url, cat: item.cat, kind: item.kind, title: { en: item.title.en, zh: item.title.zh }, desc: { en: item.desc.en, zh: item.desc.zh } } : EMPTY))
  const [file, setFile] = useState<File | null>(null)
  const [dropPreview, setDropPreview] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const fileUrl = file ? URL.createObjectURL(file) : null
  useEffect(() => () => void (fileUrl && URL.revokeObjectURL(fileUrl)), [fileUrl])
  const shown = fileUrl ?? (dropPreview || !item ? undefined : previewSrc(item))
  const save = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      let data = (await api(item ? `/api/resources/${item.id}` : '/api/resources', { method: item ? 'PATCH' : 'POST', body: JSON.stringify(d) })) as ResourceData
      const id = item?.id ?? data.resources[0].id
      if (file) data = (await api(`/api/resources/${id}/preview`, { method: 'PUT', body: file })) as ResourceData
      else if (dropPreview) data = (await api(`/api/resources/${id}/preview`, { method: 'DELETE' })) as ResourceData
      onSaved(data)
    } catch (err) {
      setError(t(`site.admin.err.${(err as Error).message}`, { defaultValue: (err as Error).message }))
      setBusy(false)
    }
  }
  const set = (patch: Partial<Draft>) => setD((x) => ({ ...x, ...patch }))
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-slate-900/40 p-4 sm:p-8" role="dialog" aria-modal="true" aria-label={t(item ? 'site.admin.edit' : 'site.admin.add')}>
      <form onSubmit={save} className="grid w-full max-w-[640px] gap-4 rounded-[20px] border border-slate-200 bg-white p-5 shadow-2xl sm:p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-extrabold">{t(item ? 'site.admin.edit' : 'site.admin.add')}</h2>
          <button type="button" onClick={onClose} aria-label={t('site.ui.close')} className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100">
            <X size={18} aria-hidden />
          </button>
        </div>
        <Field label={t('site.admin.f.url')}>
          <input className={`${input} font-mono`} value={d.url} onChange={(e) => set({ url: e.target.value })} required type="url" />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t('site.admin.f.cat')}>
            <select className={input} value={d.cat} onChange={(e) => set({ cat: e.target.value as Draft['cat'] })}>
              {RESOURCE_CATS.map((c) => (
                <option key={c} value={c}>
                  {t(`site.resources.cats.${c}`)}
                </option>
              ))}
            </select>
          </Field>
          <Field label={t('site.admin.f.kind')}>
            <select className={input} value={d.kind} onChange={(e) => set({ kind: e.target.value as Draft['kind'] })}>
              {RESOURCE_KINDS.map((k) => (
                <option key={k} value={k}>
                  {t(`site.resources.kinds.${k}`)}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={t('site.admin.f.title_en')}>
            <input className={input} value={d.title.en} onChange={(e) => set({ title: { ...d.title, en: e.target.value } })} maxLength={120} />
          </Field>
          <Field label={t('site.admin.f.title_zh')}>
            <input className={input} value={d.title.zh} onChange={(e) => set({ title: { ...d.title, zh: e.target.value } })} maxLength={120} />
          </Field>
          <Field label={t('site.admin.f.desc_en')}>
            <textarea className={`${input} min-h-[96px]`} value={d.desc.en} onChange={(e) => set({ desc: { ...d.desc, en: e.target.value } })} maxLength={600} />
          </Field>
          <Field label={t('site.admin.f.desc_zh')}>
            <textarea className={`${input} min-h-[96px]`} value={d.desc.zh} onChange={(e) => set({ desc: { ...d.desc, zh: e.target.value } })} maxLength={600} />
          </Field>
        </div>
        <p className="-mt-1 text-xs text-slate-500">{t('site.admin.hk_note')}</p>
        <Field label={t('site.admin.f.preview')}>
          <div className="flex flex-wrap items-center gap-3">
            {shown ? <img src={shown} alt="" className="h-[75px] w-[120px] rounded-lg border border-slate-200 object-cover object-top" /> : <span className="grid h-[75px] w-[120px] place-items-center rounded-lg border border-dashed border-slate-300 text-xs text-slate-400">{t('site.admin.no_preview')}</span>}
            <span className="flex flex-col gap-1.5">
              <span className="relative flex cursor-pointer items-center gap-1.5 rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-bold text-slate-700 hover:border-slate-400">
                <ImagePlus size={14} aria-hidden />
                {t('site.admin.upload')}
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  className="absolute inset-0 cursor-pointer opacity-0"
                  onChange={(e) => {
                    const f = e.target.files?.[0] ?? null
                    if (f && f.size > 3 * 1024 * 1024) return setError(t('site.admin.err.too_large'))
                    setFile(f)
                    setDropPreview(false)
                  }}
                />
              </span>
              {shown && (
                <button type="button" onClick={() => (setFile(null), setDropPreview(true))} className="rounded-lg px-3 py-1 text-left text-xs font-semibold text-rose-600 hover:bg-rose-50">
                  {t('site.admin.remove_preview')}
                </button>
              )}
            </span>
            <span className="text-xs text-slate-500">{t('site.admin.preview_hint')}</span>
          </div>
        </Field>
        {error && <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700">{error}</p>}
        <div className="flex items-center justify-between gap-3 border-t border-slate-100 pt-4">
          <span className="truncate text-xs text-slate-500">{inLang({ en: d.title.en, zh: d.title.zh, 'zh-HK': '' }, i18n.language)}</span>
          <div className="flex gap-2">
            <button type="button" onClick={onClose} className="rounded-[10px] border border-slate-300 px-4 py-2 text-sm font-bold text-slate-700 hover:border-slate-400">
              {t('site.admin.cancel')}
            </button>
            <button type="submit" disabled={busy} className="rounded-[10px] bg-blue-600 px-4 py-2 text-sm font-bold text-white hover:bg-blue-700 disabled:opacity-50">
              {t('site.admin.save')}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}

export default function AdminPage() {
  const { t, i18n } = useTranslation()
  const [state, setState] = useState<'loading' | 'missing' | 'out' | 'in'>('loading')
  const [data, setData] = useState<ResourceData | null>(null)
  const [pass, setPass] = useState('')
  const [error, setError] = useState('')
  const [editing, setEditing] = useState<ResourceItem | 'new' | null>(null)
  const [confirmDel, setConfirmDel] = useState<string | null>(null)
  const passRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    document.title = `${t('site.admin.title')} · KnowCS`
  }, [t])
  useEffect(() => {
    void (async () => {
      try {
        const s = (await api('/api/session')) as { authenticated: boolean }
        setData((await api('/api/resources')) as ResourceData)
        setState(s.authenticated ? 'in' : 'out')
      } catch {
        setState('missing')
      }
    })()
  }, [])
  const errText = (e: unknown) => t(`site.admin.err.${(e as Error).message}`, { defaultValue: (e as Error).message })
  const run = async (f: () => Promise<unknown>) => {
    setError('')
    try {
      const d = await f()
      if (d && typeof d === 'object' && 'resources' in d) setData(d as ResourceData)
    } catch (e) {
      if ((e as Error).message === 'signed_out') setState('out')
      setError(errText(e))
    }
  }
  const signIn = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      await api('/api/session', { method: 'POST', body: JSON.stringify({ passphrase: pass }) })
      setPass('')
      setState('in')
    } catch (err) {
      setError(errText(err))
      passRef.current?.select()
    }
  }
  const move = (i: number, j: number) =>
    run(() => {
      const ids = data!.resources.map((r) => r.id)
      const next = [...ids]
      ;[next[i], next[j]] = [next[j], next[i]]
      return api('/api/resources/order', { method: 'PATCH', body: JSON.stringify({ ids: next, baseIds: ids }) })
    })

  return (
    <main className={state === 'in' ? 'px-[clamp(1rem,9vw,13rem)] pb-16 pt-6 lg:pt-8' : 'flex min-h-[calc(100dvh-154px)] flex-col justify-center px-4 pb-[74px] sm:min-h-[calc(100dvh-106px)] sm:pb-[26px]'}>
      {/* signed out: title and sign-in in the middle of the screen; signed in: title left, actions right */}
      <div className={state === 'in' ? 'flex flex-wrap items-center justify-between gap-3' : 'flex justify-center'}>
        <h1 className="text-[clamp(22px,2.4vw,28px)] font-extrabold tracking-tight text-slate-900">{t('site.admin.title')}</h1>
        {state === 'in' && (
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => setEditing('new')} className="flex items-center gap-1.5 rounded-[10px] bg-blue-600 px-3.5 py-2 text-sm font-bold text-white hover:bg-blue-700">
              <Plus size={16} aria-hidden />
              {t('site.admin.add')}
            </button>
            <a href="/resources/" className="flex items-center gap-1.5 rounded-[10px] border border-slate-300 bg-white px-3.5 py-2 text-sm font-bold text-slate-700 hover:border-slate-400">
              <ExternalLink size={15} aria-hidden />
              {t('site.admin.view')}
            </a>
            <button type="button" onClick={() => run(async () => (await api('/api/session', { method: 'DELETE' }), setState('out')))} className="flex items-center gap-1.5 rounded-[10px] border border-slate-300 bg-white px-3.5 py-2 text-sm font-bold text-slate-700 hover:border-slate-400">
              <LogOut size={15} aria-hidden />
              {t('site.admin.sign_out')}
            </button>
          </div>
        )}
      </div>

      {state === 'loading' && <p className="mt-8 text-center text-sm text-slate-500">…</p>}
      {state === 'missing' && <p className="mx-auto mt-8 max-w-[380px] rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">{t('site.admin.missing')}</p>}
      {state === 'out' && (
        <form onSubmit={signIn} className="mx-auto mt-6 grid w-full max-w-[380px] gap-3 rounded-[20px] border border-slate-200 bg-white p-5 shadow-sm">
          <Field label={t('site.admin.passphrase')}>
            <input ref={passRef} type="password" autoComplete="current-password" autoFocus className={input} value={pass} onChange={(e) => setPass(e.target.value)} />
          </Field>
          {error && <p className="text-sm font-semibold text-rose-700">{error}</p>}
          <button type="submit" disabled={!pass} className="rounded-[10px] bg-blue-600 px-4 py-2 text-sm font-bold text-white hover:bg-blue-700 disabled:opacity-50">
            {t('site.admin.sign_in')}
          </button>
        </form>
      )}
      {state === 'in' && data && (
        <>
          {error && <p className="mt-4 rounded-lg bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700">{error}</p>}
          <p className="mt-2 text-sm text-slate-500">{t('site.admin.count', { count: data.resources.length })}</p>
          <ul className="mt-4 grid gap-2">
            {data.resources.map((r, i) => {
              const img = previewSrc(r)
              return (
                <li key={r.id} className="flex items-center gap-3 rounded-[14px] border border-slate-200 bg-white p-2.5">
                  <span className="flex shrink-0 flex-col">
                    <button type="button" aria-label={t('lab.cnn.move_up')} disabled={i === 0} onClick={() => move(i, i - 1)} className="grid h-6 w-6 place-items-center rounded text-slate-500 hover:bg-slate-100 disabled:opacity-30">
                      <ArrowUp size={14} aria-hidden />
                    </button>
                    <button type="button" aria-label={t('lab.cnn.move_down')} disabled={i === data.resources.length - 1} onClick={() => move(i, i + 1)} className="grid h-6 w-6 place-items-center rounded text-slate-500 hover:bg-slate-100 disabled:opacity-30">
                      <ArrowDown size={14} aria-hidden />
                    </button>
                  </span>
                  {img ? <img src={img} alt="" className="h-[50px] w-[80px] shrink-0 rounded-md border border-slate-200 object-cover object-top" /> : <span className="h-[50px] w-[80px] shrink-0 rounded-md border border-dashed border-slate-300" />}
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <b className="truncate text-[15px] text-slate-900">{inLang(r.title, i18n.language)}</b>
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-semibold text-slate-600">{t(`site.resources.cats.${r.cat}`)}</span>
                    </div>
                    <a href={r.url} target="_blank" rel="noopener noreferrer" className="block truncate font-mono text-xs text-slate-500 hover:text-blue-700">
                      {r.url}
                    </a>
                  </div>
                  <button type="button" onClick={() => setEditing(r)} className="flex shrink-0 items-center gap-1 rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-bold text-slate-700 hover:border-slate-300">
                    <Pencil size={13} aria-hidden />
                    {t('site.admin.edit')}
                  </button>
                  <button
                    type="button"
                    onClick={() => (confirmDel === r.id ? (setConfirmDel(null), run(() => api(`/api/resources/${r.id}`, { method: 'DELETE' }))) : setConfirmDel(r.id))}
                    onBlur={() => setConfirmDel((c) => (c === r.id ? null : c))}
                    className={`flex shrink-0 items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-bold ${confirmDel === r.id ? 'border-rose-600 bg-rose-600 text-white' : 'border-slate-200 text-rose-600 hover:border-rose-300'}`}
                  >
                    <Trash2 size={13} aria-hidden />
                    {t(confirmDel === r.id ? 'site.admin.confirm_delete' : 'site.admin.delete')}
                  </button>
                </li>
              )
            })}
          </ul>
        </>
      )}
      {editing && (
        <Editor
          item={editing === 'new' ? null : editing}
          onClose={() => setEditing(null)}
          onSaved={(d) => {
            setData(d)
            setEditing(null)
          }}
        />
      )}
    </main>
  )
}
