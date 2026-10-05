import { describe, expect, it } from 'vitest'

// 资料包 / 拓展内容的双语一致性守卫：en 与 zh 必须成对出现（zh-HK 由 zh 在构建期生成）
const notes = import.meta.glob<string>('./notes/*.md', { query: '?raw', import: 'default', eager: true })
const pages = import.meta.glob<string>('./attention.*.html', { query: '?raw', import: 'default', eager: true })
const read = (f: string) => notes[`./notes/${f}`]
const CJK = /[\u4e00-\u9fff]/
const stripCode = (md: string) => md.replace(/```[\s\S]*?```/g, '').replace(/`[^`\n]*`/g, '')
const images = (md: string) => [...md.matchAll(/\]\((images\/[^)]+)\)/g)].map((m) => m[1])

const files = Object.keys(notes).map((k) => k.replace('./notes/', ''))
const ids = [...new Set(files.map((f) => f.replace(/\.(en|zh)\.md$/, '')))]

describe('Package 笔记', () => {
  it.each(ids)('%s：en 与 zh 成对存在且引用相同的图片', (id) => {
    expect(files).toContain(`${id}.en.md`)
    expect(files).toContain(`${id}.zh.md`)
    expect(images(read(`${id}.zh.md`))).toEqual(images(read(`${id}.en.md`)))
  })

  it.each(ids)('%s：英文版不含中文', (id) => {
    expect(read(`${id}.en.md`)).not.toMatch(CJK)
  })

  it.each(files)('%s：不含 Obsidian 专有语法（callout / 双链）', (f) => {
    const prose = stripCode(read(f))
    expect(prose).not.toMatch(/^>\s*\[!\w+\]/m)
    expect(prose).not.toMatch(/\[\[[^\]]+\]\]/)
  })
})

describe('Extend 页面', () => {
  const page = (lang: string) => pages[`./attention.${lang}.html`]

  it('英文版不含中文，且 lang 正确', () => {
    expect(page('en')).not.toMatch(CJK)
    expect(page('en')).toMatch(/<html lang="en">/)
    expect(page('zh')).toMatch(/<html lang="zh-CN">/)
  })

  it('两个版本的脚本结构一致（只翻译文案）', () => {
    const shape = (html: string) => (html.match(/getElementById\('[\w-]+'\)/g) ?? []).join()
    expect(shape(page('en'))).toBe(shape(page('zh')))
  })
})
