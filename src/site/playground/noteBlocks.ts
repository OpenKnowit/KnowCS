// 笔记里的代码块：找出能在该笔记实验台里直接运行的那些，给它们加「试一试」按钮
import { runIn } from '../../data/playgrounds/run'
import type { PlayConfig } from '../../data/playgrounds/types'

const PRE = /<pre><code class="language-python">([\s\S]*?)<\/code><\/pre>/g

const decode = (html: string): string =>
  html.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&amp;/g, '&')

/** the Python code blocks of a note, in order */
export const pythonBlocks = (html: string): string[] => [...html.matchAll(PRE)].map((m) => decode(m[1]).replace(/\n$/, ''))

/** a block as the playground opens it: the library's imports in front when the block has none */
export const withPrelude = (config: PlayConfig, code: string): string => (/^\s*(import|from)\s/m.test(code) ? code : config.prelude + code)

/**
 * Which blocks run without an error (a block that only runs out of time still counts: the playground gives it longer).
 * Yields between blocks so a long note does not freeze the page.
 */
export async function runnableBlocks(config: PlayConfig, codes: string[], isLive: () => boolean): Promise<boolean[]> {
  const out: boolean[] = []
  for (const code of codes) {
    await new Promise((r) => setTimeout(r, 0))
    if (!isLive()) return out
    const r = runIn(config, withPrelude(config, code), 300)
    out.push(!r.error || r.error.type === 'TimeoutError')
  }
  return out
}

/** add a run button after every runnable block (data-block = its position among the Python blocks) */
export const decorateBlocks = (html: string, ok: boolean[], label: string, hint: string): string => {
  let k = -1
  return html.replace(PRE, (block) => {
    k++
    return ok[k] ? `<div class="np-block">${block}<button type="button" class="np-run" data-block="${k}" title="${hint.replace(/"/g, '&quot;')}">▶ ${label}</button></div>` : block
  })
}
