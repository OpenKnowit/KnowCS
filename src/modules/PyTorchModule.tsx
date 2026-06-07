import { useMemo, useState } from 'react'
import { Code2, FlameKindling, GitCommitHorizontal, Network } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from '../components/Latex'
import { SeniorAdvice } from '../components/SeniorAdvice'
import { PYTORCH_CHEATSHEET, PYTORCH_CNN_CODE, PYTORCH_TRAIN_LOOP } from '../data/constants'
import { Value } from '../lib/autograd'

const CodeBlock = ({ code }: { code: string }) => (
  <pre className="bg-slate-950 text-slate-100 rounded-xl p-4 overflow-x-auto text-[11px] leading-relaxed font-mono border border-slate-800">
    <code>{code}</code>
  </pre>
)

// 计算图布局：a,b,c → d=a*b → e=d+c → L=relu(e)
const LAYOUT: Record<string, { x: number; y: number }> = {
  a: { x: 70, y: 40 }, b: { x: 70, y: 110 }, c: { x: 70, y: 200 },
  d: { x: 240, y: 75 }, e: { x: 410, y: 130 }, L: { x: 560, y: 130 },
}
const GRAPH_EDGES: [string, string][] = [['a', 'd'], ['b', 'd'], ['d', 'e'], ['c', 'e'], ['e', 'L']]

export const PyTorchModule = () => {
  const { t } = useTranslation()
  const [a, setA] = useState(2)
  const [b, setB] = useState(3)
  const [c, setC] = useState(-1)

  // 前向 + 反向，取出每个节点的 data 与 grad
  const nodes = useMemo(() => {
    const av = new Value(a, 'a')
    const bv = new Value(b, 'b')
    const cv = new Value(c, 'c')
    const dv = av.mul(bv, 'd')
    const ev = dv.add(cv, 'e')
    const Lv = ev.relu('L')
    Lv.backward()
    const map: Record<string, Value> = { a: av, b: bv, c: cv, d: dv, e: ev, L: Lv }
    return map
  }, [a, b, c])

  return (
    <div className="space-y-8">
      {/* 1. Autograd 交互式计算图 */}
      <div className="bg-white border border-slate-200 rounded-[2rem] p-8 shadow-sm">
        <h4 className="text-sm font-bold text-slate-700 flex items-center gap-2 uppercase tracking-wide mb-1"><Network size={16} className="text-indigo-500" /> {t('pytorch.autograd_title')}</h4>
        <p className="text-[11px] text-slate-400 mb-5">{t('pytorch.autograd_subtitle')} <Latex formula="L = \mathrm{relu}(a \cdot b + c)" /></p>
        <div className="bg-slate-900 rounded-2xl p-4">
          <svg viewBox="0 0 640 250" className="w-full h-auto">
            {/* 边 */}
            {GRAPH_EDGES.map(([f, to], i) => {
              const p = LAYOUT[f], q = LAYOUT[to]
              return <line key={i} x1={p.x + 30} y1={p.y} x2={q.x - 30} y2={q.y} stroke="#475569" strokeWidth={2.5} markerEnd="url(#pt-arrow)" />
            })}
            <defs>
              <marker id="pt-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#475569" /></marker>
            </defs>
            {/* 节点 */}
            {Object.entries(LAYOUT).map(([id, pos]) => {
              const v = nodes[id]
              const isInput = ['a', 'b', 'c'].includes(id)
              const isOut = id === 'L'
              return (
                <g key={id}>
                  <rect x={pos.x - 30} y={pos.y - 22} width={60} height={44} rx={10}
                    fill={isOut ? '#3730a3' : isInput ? '#0f172a' : '#1e293b'} stroke={isOut ? '#818cf8' : '#334155'} strokeWidth={2} />
                  <text x={pos.x} y={pos.y - 6} textAnchor="middle" fontSize={11} fontWeight={800} fill="#e2e8f0" fontFamily="monospace">{v.label || id}{v.op ? `=${v.op}` : ''}</text>
                  <text x={pos.x} y={pos.y + 8} textAnchor="middle" fontSize={10} fill="#38bdf8" fontFamily="monospace">{v.data.toFixed(2)}</text>
                  {/* 梯度徽章 */}
                  <g transform={`translate(${pos.x}, ${pos.y + 32})`}>
                    <rect x={-32} y={-9} width={64} height={17} rx={5} fill="#7f1d1d" opacity={0.85} />
                    <text textAnchor="middle" dy={3} fontSize={9} fill="#fecaca" fontFamily="monospace">grad {v.grad.toFixed(2)}</text>
                  </g>
                </g>
              )
            })}
          </svg>
          <div className="flex flex-wrap items-center justify-center gap-4 mt-3 text-[10px]">
            <span className="flex items-center gap-1.5 text-sky-400"><GitCommitHorizontal size={12} /> {t('pytorch.forward')} (data)</span>
            <span className="flex items-center gap-1.5 text-rose-300"><FlameKindling size={12} /> {t('pytorch.backward')} (.grad)</span>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mt-5">
          {[['a', a, setA], ['b', b, setB], ['c', c, setC]].map(([name, val, setter]) => (
            <div key={name as string} className="flex flex-col items-center gap-1">
              <span className="text-[10px] font-mono font-bold text-slate-500">{name as string}</span>
              <input type="range" min={-4} max={4} step={0.5} value={val as number} onChange={(e) => (setter as (n: number) => void)(parseFloat(e.target.value))} className="w-full accent-indigo-600" />
              <span className="text-sm font-black text-indigo-600 font-mono">{(val as number).toFixed(1)}</span>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-slate-400 mt-3 text-center">{t('pytorch.autograd_hint')}</p>
      </div>

      {/* 2. Cheatsheet */}
      <div>
        <h4 className="text-sm font-bold text-slate-700 flex items-center gap-2 uppercase tracking-wide mb-4"><Code2 size={16} className="text-emerald-500" /> {t('pytorch.cheatsheet_title')}</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {PYTORCH_CHEATSHEET.map((group) => (
            <div key={group.titleKey} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
              <p className="text-[10px] font-black uppercase tracking-widest text-indigo-500 mb-3">{t(group.titleKey)}</p>
              <div className="space-y-3">
                {group.items.map((item) => (
                  <div key={item.api}>
                    <div className="flex items-baseline justify-between gap-2 mb-1">
                      <code className="text-[11px] font-bold text-slate-800 font-mono">{item.api}</code>
                    </div>
                    <code className="block text-[10px] bg-slate-50 border border-slate-100 rounded px-2 py-1 font-mono text-slate-600 overflow-x-auto">{item.code}</code>
                    <p className="text-[10px] text-slate-400 mt-1">{t(item.descKey)}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 3. 标准训练循环 + CNN */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <p className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2">{t('pytorch.train_loop_title')}</p>
          <CodeBlock code={PYTORCH_TRAIN_LOOP} />
          <p className="text-[10px] text-slate-400 mt-2">{t('pytorch.train_loop_hint')}</p>
        </div>
        <div>
          <p className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-2">{t('pytorch.cnn_title')}</p>
          <CodeBlock code={PYTORCH_CNN_CODE} />
          <p className="text-[10px] text-slate-400 mt-2">{t('pytorch.cnn_hint')}</p>
        </div>
      </div>

      <SeniorAdvice content={<Trans i18nKey="pytorch.advice" components={{
        1: <code className="font-mono font-bold text-amber-800 bg-amber-100 px-1 rounded" />,
        3: <strong className="font-bold text-amber-700" />,
      }} />} />
    </div>
  )
}
