// Capture live pages into raw .pen trees: node scripts/pencil-sync/capture.mjs <outDir> <baseUrl> <key=path>...
// Writes <outDir>/<key>.json and <outDir>/assets/<key>/<id>.png. Needs Google Chrome; set PROXY=host:port if required.
import { spawn } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const [outDir, base, ...pages] = process.argv.slice(2)
const EXTRACT = readFileSync(join(here, 'extract.js'), 'utf8')
const FONTS = `(() => {
  const l = document.createElement('link'); l.rel = 'stylesheet'
  l.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@100..900&family=JetBrains+Mono:wght@100..800&display=block'
  document.head.append(l)
  const s = document.createElement('style')
  s.textContent = ':root,html{--font-sans:Inter,sans-serif!important;--default-font-family:Inter,sans-serif!important;--font-mono:"JetBrains Mono",monospace!important;--default-mono-font-family:"JetBrains Mono",monospace!important}body{font-family:Inter,sans-serif}'
  document.head.append(s)
  return new Promise((r) => l.onload = () => document.fonts.ready.then(() => r(true)))
})()`

mkdirSync(outDir, { recursive: true })
const port = 9334
const args = ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${join(outDir, '.prof')}`, '--hide-scrollbars', '--window-size=1440,900', 'about:blank']
if (process.env.PROXY) args.splice(1, 0, `--proxy-server=${process.env.PROXY}`)
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args, { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

let ws
for (let i = 0; i < 60 && !ws; i++) {
  try {
    const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page')
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl)
  } catch { await sleep(250) }
}
await new Promise((r) => ws.addEventListener('open', r))
let seq = 0
const pending = new Map()
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) } })
const send = (method, params = {}) => new Promise((r) => { const i = ++seq; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })) })
const evaluate = async (expression) => {
  const m = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
  if (m.result?.exceptionDetails) throw new Error(JSON.stringify(m.result.exceptionDetails).slice(0, 600))
  return m.result.result.value
}
const viewport = (h) => send('Emulation.setDeviceMetricsOverride', { width: 1440, height: h, deviceScaleFactor: 1, mobile: false })

await send('Page.enable')
await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }, { name: 'prefers-color-scheme', value: 'light' }] })
for (const spec of pages) {
  const [key, path] = spec.split('=')
  await viewport(900)
  await send('Page.navigate', { url: `${base}/${path}${path.includes('?') ? '&' : '?'}lang=en` })
  await sleep(3500)
  await evaluate(FONTS)
  const h0 = await evaluate('document.documentElement.scrollHeight')
  for (let y = 0; y < h0; y += 600) { await evaluate(`scrollTo(0,${y})`); await sleep(150) }
  await evaluate('scrollTo(0,0)')
  const h = Math.min(16000, await evaluate('document.documentElement.scrollHeight'))
  await viewport(h)
  await sleep(1500)
  const { tree, rasters, palette } = await evaluate(EXTRACT)
  const adir = join(outDir, 'assets', key)
  mkdirSync(adir, { recursive: true })
  for (const r of rasters) {
    const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true, clip: { x: r.x, y: r.y, width: r.w, height: r.h, scale: 2 } })
    if (shot.result?.data) writeFileSync(join(adir, `${r.id}.png`), Buffer.from(shot.result.data, 'base64'))
  }
  const full = await send('Page.captureScreenshot', { format: 'png' })
  writeFileSync(join(outDir, `${key}.ref.png`), Buffer.from(full.result.data, 'base64'))
  writeFileSync(join(outDir, `${key}.json`), JSON.stringify({ key, path, tree, rasters: rasters.length, palette }))
  console.log(key, 'h', h, 'rasters', rasters.length, 'bytes', JSON.stringify(tree).length)
}
ws.close()
chrome.kill()
