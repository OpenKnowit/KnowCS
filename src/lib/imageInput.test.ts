import { describe, expect, it } from 'vitest'
import { MAX_FILE_BYTES, MAX_PIXELS, MAX_REMOTE_BYTES, ImageInputError, checkImageBytes, fetchRandomImage, randomImageUrl, sniffImage } from './imageInput'

const pad = (head: number[], n = 40) => Uint8Array.from([...head, ...new Array(Math.max(0, n - head.length)).fill(0)])
const be32 = (v: number) => [(v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255]
const le16 = (v: number) => [v & 255, (v >> 8) & 255]
const le32 = (v: number) => [v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255]
const text = (s: string) => [...s].map((c) => c.charCodeAt(0))
const png = (w: number, h: number) => pad([0x89, ...text('PNG'), 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, ...text('IHDR'), ...be32(w), ...be32(h)])
const jpeg = (w: number, h: number) => pad([0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0, 0xff, 0xc0, 0, 17, 8, h >> 8, h & 255, w >> 8, w & 255])

describe('sniffImage', () => {
  it('reads the size from each format', () => {
    expect(sniffImage(png(640, 480))).toEqual({ type: 'png', width: 640, height: 480 })
    expect(sniffImage(jpeg(1024, 768))).toEqual({ type: 'jpeg', width: 1024, height: 768 })
    expect(sniffImage(pad([...text('GIF89a'), ...le16(32), ...le16(16)]))).toEqual({ type: 'gif', width: 32, height: 16 })
    expect(sniffImage(pad([...text('BM'), ...new Array(16).fill(0), ...le32(100), ...le32(-50)]))).toEqual({ type: 'bmp', width: 100, height: 50 })
    const vp8x = pad([...text('RIFF'), 0, 0, 0, 0, ...text('WEBPVP8X'), 0, 0, 0, 0, 0, 0, 0, 0, 99, 0, 0, 49, 0, 0])
    expect(sniffImage(vp8x)).toEqual({ type: 'webp', width: 100, height: 50 })
  })

  it('goes by the bytes, so a script named .png is not an image', () => {
    expect(sniffImage(Uint8Array.from(text('<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"></svg>')))).toBeNull()
    expect(sniffImage(Uint8Array.from(text('<html><script>alert(1)</script></html>')))).toBeNull()
    expect(sniffImage(Uint8Array.from(text('MZ this is an executable, not a picture')))).toBeNull()
    expect(sniffImage(new Uint8Array(3))).toBeNull()
    // a JPEG start with no frame header never gets a size
    expect(sniffImage(pad([0xff, 0xd8, 0x00, 0x00]))).toBeNull()
  })

  it('does not hang on a JPEG whose segments lead nowhere', () => {
    expect(sniffImage(pad([0xff, 0xd8, 0xff, 0xe0, 0xff, 0xff]))).toBeNull()
    expect(sniffImage(new Uint8Array(4096).fill(0xff))).toBeNull()
  })
})

describe('checkImageBytes', () => {
  const code = (f: () => unknown) => {
    try {
      f()
    } catch (e) {
      return (e as ImageInputError).code
    }
    return 'accepted'
  }
  it('refuses what should not be decoded', () => {
    expect(code(() => checkImageBytes(png(100, 100), MAX_FILE_BYTES))).toBe('accepted')
    expect(code(() => checkImageBytes(Uint8Array.from(text('not an image at all')), MAX_FILE_BYTES))).toBe('bad_type')
    expect(code(() => checkImageBytes(new Uint8Array(100), 10))).toBe('too_big')
    // a small file that claims a huge picture: a decompression bomb
    expect(code(() => checkImageBytes(png(50_000, 50_000), MAX_FILE_BYTES))).toBe('too_many_pixels')
    expect(code(() => checkImageBytes(png(1, MAX_PIXELS + 1), MAX_FILE_BYTES))).toBe('too_many_pixels')
    expect(code(() => checkImageBytes(png(0, 10), MAX_FILE_BYTES))).toBe('too_many_pixels')
  })
})

describe('fetchRandomImage', () => {
  const respond = (body: Uint8Array, url: string, headers: Record<string, string> = { 'content-type': 'image/png' }, status = 200) => {
    const res = new Response(body as BodyInit, { status, headers })
    Object.defineProperty(res, 'url', { value: url })
    return (async () => res) as unknown as typeof fetch
  }
  const OK = 'https://fastly.picsum.photos/id/1/256/256.jpg'
  const code = (p: Promise<unknown>) => p.then(() => 'accepted', (e) => (e as ImageInputError).code)

  it('builds a fixed address and sends no cookies or referrer', async () => {
    expect(randomImageUrl(7.9)).toBe('https://picsum.photos/256/256?random=7')
    let init: RequestInit | undefined
    const spy = (async (_: string, i: RequestInit) => {
      init = i
      const res = new Response(png(256, 256) as BodyInit, { headers: { 'content-type': 'image/png' } })
      Object.defineProperty(res, 'url', { value: OK })
      return res
    }) as unknown as typeof fetch
    await fetchRandomImage(1, undefined, spy)
    expect(init).toMatchObject({ mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer' })
  })

  it('accepts a small image from the expected hosts', async () => {
    expect(await code(fetchRandomImage(1, undefined, respond(png(256, 256), OK)))).toBe('accepted')
  })

  it('refuses another host, another type, an oversize body and a bad status', async () => {
    expect(await code(fetchRandomImage(1, undefined, respond(png(256, 256), 'https://evil.example/x.png')))).toBe('bad_source')
    expect(await code(fetchRandomImage(1, undefined, respond(png(256, 256), 'https://picsum.photos.evil.example/x.png')))).toBe('bad_source')
    expect(await code(fetchRandomImage(1, undefined, respond(png(256, 256), OK, { 'content-type': 'text/html' })))).toBe('bad_type')
    expect(await code(fetchRandomImage(1, undefined, respond(Uint8Array.from(text('<html>not an image, whatever it says</html>')), OK)))).toBe('bad_type')
    expect(await code(fetchRandomImage(1, undefined, respond(png(256, 256), OK, { 'content-type': 'image/png', 'content-length': String(MAX_REMOTE_BYTES + 1) })))).toBe('too_big')
    expect(await code(fetchRandomImage(1, undefined, respond(new Uint8Array(MAX_REMOTE_BYTES + 1), OK)))).toBe('too_big')
    expect(await code(fetchRandomImage(1, undefined, respond(png(256, 256), OK, { 'content-type': 'image/png' }, 500)))).toBe('network')
  })

  it('reports a failed request as a network error', async () => {
    const down = (async () => {
      throw new TypeError('Failed to fetch')
    }) as unknown as typeof fetch
    expect(await code(fetchRandomImage(1, undefined, down))).toBe('network')
  })
})
