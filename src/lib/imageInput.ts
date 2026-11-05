// Getting a picture into the browser safely. Whatever comes in, an upload or a remote image, is only ever decoded and
// turned into a grid of grey numbers: it is never put in the page, stored or sent anywhere.

export const MAX_FILE_BYTES = 10 * 1024 * 1024
export const MAX_REMOTE_BYTES = 5 * 1024 * 1024
export const MAX_PIXELS = 40_000_000
/** the square every picture is kept at; the resolution picker averages down from here */
export const WORK_SIZE = 256
export const RANDOM_HOSTS = ['picsum.photos', 'fastly.picsum.photos']
const RANDOM_TIMEOUT_MS = 12_000

export type ImageError = 'too_big' | 'bad_type' | 'too_many_pixels' | 'decode' | 'network' | 'timeout' | 'bad_source'
export class ImageInputError extends Error {
  code: ImageError
  constructor(code: ImageError) {
    super(code)
    this.code = code
  }
}

export interface ImageInfo {
  type: 'png' | 'jpeg' | 'gif' | 'webp' | 'bmp'
  width: number
  height: number
}

/**
 * What the bytes really are, whatever the file claims, and how large they will decode to. Reading the size from the
 * header lets a tiny file that expands to a huge picture be refused before it is decoded. SVG is deliberately absent:
 * it is a document that can carry scripts and links, not pixels.
 */
export const sniffImage = (b: Uint8Array): ImageInfo | null => {
  const be16 = (i: number) => (b[i] << 8) | b[i + 1]
  const le16 = (i: number) => b[i] | (b[i + 1] << 8)
  const be32 = (i: number) => ((b[i] << 24) | (b[i + 1] << 16) | (b[i + 2] << 8) | b[i + 3]) >>> 0
  const le24 = (i: number) => b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)
  const le32 = (i: number) => (b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24)) | 0
  const ascii = (i: number, n: number) => String.fromCharCode(...b.subarray(i, i + n))
  if (b.length < 12) return null
  if (b[0] === 0x89 && ascii(1, 3) === 'PNG' && b[4] === 0x0d && b[5] === 0x0a && b[6] === 0x1a && b[7] === 0x0a && b.length >= 24) {
    return { type: 'png', width: be32(16), height: be32(20) }
  }
  if (ascii(0, 6) === 'GIF87a' || ascii(0, 6) === 'GIF89a') return { type: 'gif', width: le16(6), height: le16(8) }
  if (ascii(0, 2) === 'BM' && b.length >= 26) return { type: 'bmp', width: Math.abs(le32(18)), height: Math.abs(le32(22)) }
  if (ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP' && b.length >= 30) {
    const chunk = ascii(12, 4)
    if (chunk === 'VP8 ') return { type: 'webp', width: le16(26) & 0x3fff, height: le16(28) & 0x3fff }
    if (chunk === 'VP8L') {
      const n = (b[21] | (b[22] << 8) | (b[23] << 16) | (b[24] << 24)) >>> 0
      return { type: 'webp', width: (n & 0x3fff) + 1, height: ((n >>> 14) & 0x3fff) + 1 }
    }
    if (chunk === 'VP8X') return { type: 'webp', width: le24(24) + 1, height: le24(27) + 1 }
    return null
  }
  if (b[0] === 0xff && b[1] === 0xd8) {
    // walk the JPEG's segments to its frame header
    let i = 2
    while (i + 9 < b.length) {
      if (b[i] !== 0xff) return null
      const marker = b[i + 1]
      if (marker === 0xff) {
        i++
        continue
      }
      const isFrame = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc
      if (isFrame) return { type: 'jpeg', width: be16(i + 7), height: be16(i + 5) }
      i += 2 + be16(i + 2)
    }
  }
  return null
}

/** the checks every picture passes before it is decoded */
export const checkImageBytes = (bytes: Uint8Array, maxBytes: number): ImageInfo => {
  if (bytes.length > maxBytes) throw new ImageInputError('too_big')
  const info = sniffImage(bytes)
  if (!info) throw new ImageInputError('bad_type')
  if (info.width < 1 || info.height < 1 || info.width * info.height > MAX_PIXELS) throw new ImageInputError('too_many_pixels')
  return info
}

/** the fixed address of a random photo; nothing typed by anyone goes into it */
export const randomImageUrl = (nonce: number) => `https://picsum.photos/${WORK_SIZE}/${WORK_SIZE}?random=${Math.floor(Math.abs(nonce))}`

/** download a random photo, refusing anything but a small image from the expected hosts */
export const fetchRandomImage = async (nonce: number, signal?: AbortSignal, fetcher: typeof fetch = fetch): Promise<Uint8Array> => {
  const timeout = new AbortController()
  const timer = setTimeout(() => timeout.abort(), RANDOM_TIMEOUT_MS)
  const onAbort = () => timeout.abort()
  signal?.addEventListener('abort', onAbort)
  try {
    const res = await fetcher(randomImageUrl(nonce), { mode: 'cors', credentials: 'omit', referrerPolicy: 'no-referrer', redirect: 'follow', signal: timeout.signal })
    if (!res.ok) throw new ImageInputError('network')
    if (!RANDOM_HOSTS.includes(new URL(res.url).hostname)) throw new ImageInputError('bad_source')
    if (!(res.headers.get('content-type') ?? '').startsWith('image/')) throw new ImageInputError('bad_type')
    if (Number(res.headers.get('content-length') ?? 0) > MAX_REMOTE_BYTES) throw new ImageInputError('too_big')
    // the header can lie: count what actually arrives
    const reader = res.body?.getReader()
    if (!reader) throw new ImageInputError('network')
    const parts: Uint8Array[] = []
    let total = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.length
      if (total > MAX_REMOTE_BYTES) {
        await reader.cancel()
        throw new ImageInputError('too_big')
      }
      parts.push(value)
    }
    const bytes = new Uint8Array(total)
    let at = 0
    for (const p of parts) {
      bytes.set(p, at)
      at += p.length
    }
    checkImageBytes(bytes, MAX_REMOTE_BYTES)
    return bytes
  } catch (e) {
    if (e instanceof ImageInputError) throw e
    throw new ImageInputError(timeout.signal.aborted ? 'timeout' : 'network')
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

/** checked bytes → a WORK_SIZE square of grey values: centre-cropped, on white, weighted as in the course's formula */
export const decodeToGrey = async (bytes: Uint8Array, info: ImageInfo): Promise<number[][]> => {
  let bitmap: ImageBitmap
  try {
    bitmap = await createImageBitmap(new Blob([bytes as BlobPart], { type: `image/${info.type}` }))
  } catch {
    throw new ImageInputError('decode')
  }
  try {
    const side = Math.min(bitmap.width, bitmap.height)
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = WORK_SIZE
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new ImageInputError('decode')
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, WORK_SIZE, WORK_SIZE)
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, WORK_SIZE, WORK_SIZE)
    const { data } = ctx.getImageData(0, 0, WORK_SIZE, WORK_SIZE)
    return Array.from({ length: WORK_SIZE }, (_, r) =>
      Array.from({ length: WORK_SIZE }, (_, c) => {
        const i = (r * WORK_SIZE + c) * 4
        return Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2])
      }),
    )
  } finally {
    bitmap.close()
  }
}

export const readUpload = async (file: File): Promise<number[][]> => {
  // refuse by the stated size first, so a huge file is not read into memory at all
  if (file.size > MAX_FILE_BYTES) throw new ImageInputError('too_big')
  const bytes = new Uint8Array(await file.arrayBuffer())
  return decodeToGrey(bytes, checkImageBytes(bytes, MAX_FILE_BYTES))
}

export const readRandom = async (nonce: number, signal?: AbortSignal): Promise<number[][]> => {
  const bytes = await fetchRandomImage(nonce, signal)
  return decodeToGrey(bytes, checkImageBytes(bytes, MAX_REMOTE_BYTES))
}
