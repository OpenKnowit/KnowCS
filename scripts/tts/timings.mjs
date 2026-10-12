// Dumps every explainer episode's scene timing (id, dur, cues, ponder) as JSON, for scripts/tts/narrate.py.
// Usage: node scripts/tts/timings.mjs > work/dubbing/timings.json
import { createServer } from 'vite'

const server = await createServer({ configFile: false, logLevel: 'error', server: { middlewareMode: true, hmr: false }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } })
try {
  const { EPISODE_CODE } = await server.ssrLoadModule('/src/site/watch/episodes/index.ts')
  const out = Object.fromEntries(Object.entries(EPISODE_CODE).map(([id, ep]) => [id, ep.scenes.map((s) => ({ id: s.id, dur: s.dur, cues: s.cues, ponder: !!s.ponder }))]))
  process.stdout.write(JSON.stringify(out, null, 1) + '\n')
} finally {
  await server.close()
}
