// Gallery thumbnails for /playground/: open every example in a headless Chrome and photograph its Visual pane.
//   npm run build && npx vite preview --port 5180 &
//   node scripts/playground-thumbs.mjs http://127.0.0.1:5180
// Writes src/assets/playground/<library>/<example>.webp, then scripts/playground-thumbs-fit.py frames them at 16:10. Examples whose pane stays empty get no picture
// (the gallery shows their code instead). Rebuild afterwards so the pictures are bundled.
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const base = process.argv[2] ?? 'http://127.0.0.1:5180'
const assets = join(dirname(fileURLToPath(import.meta.url)), '../src/assets/playground')
// ONLY_NEW=/some/folder photographs just the examples with no picture yet, into that folder
const out = process.env.ONLY_NEW || assets
const LIBS = process.env.LIBS ? process.env.LIBS.split(',') : ['numpy', 'matplotlib', 'pandas', 'pytorch', 'keras', 'tensorflow', 'python', 'sklearn', 'cv2']
const port = 9337
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ['--headless=new', '--no-proxy-server', `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'thumbs-'))}`, '--hide-scrollbars', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

let ws
for (let i = 0; i < 60 && !ws; i++) {
  try {
    const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page')
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl)
  } catch {
    await sleep(250)
  }
}
await new Promise((r) => ws.addEventListener('open', r))
let seq = 0
const pending = new Map()
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data)
  if (m.id && pending.has(m.id)) (pending.get(m.id)(m), pending.delete(m.id))
})
const send = (method, params = {}) => new Promise((r) => { const i = ++seq; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })) })
const js = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result.result?.value

await send('Page.enable')
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 2, mobile: false })
let saved = 0
for (const lib of LIBS) {
  await send('Page.navigate', { url: 'about:blank' })
  await send('Page.navigate', { url: `${base}/playground/?lang=en#${lib}` })
  await sleep(2500)
  // step-through engine, a wide results pane, and every group in this library open
  await js(`localStorage.setItem('knowcs-py-engine','mini'); localStorage.setItem('knowcs-playground-split','50'); true`)
  await send('Page.reload')
  await sleep(3500)
  const ids = await js(`(async () => {
    const nav = document.querySelector('aside nav')
    for (let k = 0; k < 4; k++) { nav.querySelectorAll('button[aria-expanded="false"]').forEach((b) => b.click()); await new Promise((r) => setTimeout(r, 150)) }
    return [...nav.querySelectorAll('[data-entry^="${lib}:"]')].map((b) => b.dataset.entry.slice(${lib.length + 1}))
  })()`)
  mkdirSync(join(out, lib), { recursive: true })
  for (const id of ids) {
    if (process.env.ONLY_NEW && existsSync(join(assets, lib, `${id}.webp`))) continue
    await js(`document.querySelector('[data-entry="${lib}:${id}"]').click(); true`)
    await sleep(lib === 'numpy' ? 450 : 900)
    const box = await js(`(() => {
      const b = document.querySelector('[data-visual-body]')
      if (!b) return { skip: 'no pane' }
      if (b.classList.contains('opacity-60')) return { skip: 'still running' }
      if (!b.querySelector('svg, canvas, table, img, [class*="grid"]')) return { skip: 'empty: ' + b.innerText.slice(0, 60).replace(/\\s+/g, ' ') }
      b.scrollTop = 0
      // the picture is the biggest framed thing in the pane (the array grids, the figure, the graph), not the prose above it
      const pane = b.getBoundingClientRect()
      let best = null
      for (const el of b.querySelectorAll('*')) {
        const cs = getComputedStyle(el)
        const framed = parseFloat(cs.borderTopWidth) > 0 || ['svg', 'canvas', 'table', 'img'].includes(el.tagName.toLowerCase())
        if (!framed || el.closest('svg') !== null && el.tagName.toLowerCase() !== 'svg') continue
        const r = el.getBoundingClientRect()
        const area = Math.min(r.width, pane.width) * Math.min(r.height, pane.bottom - r.top)
        if (r.width > 60 && r.height > 40 && (!best || area > best.area)) best = { r, area }
      }
      const r = best ? best.r : pane
      const pad = 10
      const x = Math.max(pane.left, r.left - pad), y = Math.max(pane.top, r.top - pad)
      return { x, y, w: Math.min(r.right + pad, pane.right) - x, h: Math.min(r.bottom + pad, pane.bottom, y + 520) - y }
    })()`)
    if (box.skip) {
      console.log('skip', lib, id, box.skip)
      continue
    }
    const shot = await send('Page.captureScreenshot', { format: 'webp', quality: 80, clip: { x: box.x, y: box.y, width: box.w, height: box.h, scale: 1 } })
    writeFileSync(join(out, lib, `${id}.webp`), Buffer.from(shot.result.data, 'base64'))
    saved++
  }
  console.log(lib, ids.length, 'examples')
}
console.log('saved', saved)
ws.close()
chrome.kill()
