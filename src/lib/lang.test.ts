import { describe, expect, it } from 'vitest'
import { normalizeLang } from './lang'

describe('normalizeLang', () => {
  it('站点自身的三种语言原样返回', () => {
    expect(normalizeLang('en')).toBe('en')
    expect(normalizeLang('zh')).toBe('zh')
    expect(normalizeLang('zh-HK')).toBe('zh-HK')
  })

  it('繁体地区 / Hant 脚本 → zh-HK（大小写、下划线不敏感）', () => {
    for (const tag of ['zh-TW', 'zh-MO', 'zh-Hant', 'zh-Hant-HK', 'zh_tw', 'ZH-HK']) {
      expect(normalizeLang(tag)).toBe('zh-HK')
    }
  })

  it('简体地区 / Hans 脚本 → zh', () => {
    for (const tag of ['zh-CN', 'zh-SG', 'zh-Hans', 'zh-Hans-CN']) {
      expect(normalizeLang(tag)).toBe('zh')
    }
  })

  it('其他语言与空值回退 en', () => {
    for (const tag of ['en-US', 'fr', 'ja-JP', '', undefined, null]) {
      expect(normalizeLang(tag)).toBe('en')
    }
  })
})
